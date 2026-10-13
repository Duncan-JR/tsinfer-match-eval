# Performance testing for sample matching

## Scope and the meeting's timing boundary

Implement a benchmark harness for the existing full HMM, independent regional
reduced HMM, and exact greedy focal cover plus full-HMM gaps. Measure elapsed,
user and system time, memory, preparation costs, and the work each method did.
This document plans implementation; no production code is changed by this task.

Jerome's instruction in the [1 October meeting notes](../../../exo/documents/meeting_notes/2026-10-01_gil_yan_jerome.md)
was: **“Do not include the cost of extracting them from Zarr in the first timing
experiment.”** This applies to the initial algorithm experiment, not to an
ultimate claim about pipeline speed. He also requested a substantially longer
sequence than 10 Mb, warned about simultaneous Zarr decompression and cache
pressure, and advised measuring the simple implementation before redesigning
storage or the HMM.

Use three explicitly labelled comparisons:

| Mode | Starting point and timed work | Purpose |
| --- | --- | --- |
| `resident_match` | Canonical sample calls, reference, shared full indexes, candidates, intervals and masks are resident. Time the complete copying-path computation, including method-specific per-haplotype simplification, reduced indexing, matcher construction, full-HMM gaps and path/mutation construction. | Jerome's first comparison: exclude Zarr extraction and archive loading. |
| `method_from_reference` | Start with the same saved ancestor panel/reference and annotated sample source. Include candidate lookup, interval generation, mask generation, loading, matching, native sample assembly and output writing for the focal methods. Include corresponding loading, matching, assembly and writing for full HMM. | Practical cost of obtaining sample matches from a common reference. |
| `reference_preparation` | Sample-store copying/count annotation, native configuration, ancestor inference and ancestor matching. | Separate common setup and sizing costs; these are not a focal-method speedup. |

Also report the focal kernel on its own as a diagnostic: prepared reduced-HMM
regional calls versus greedy selection, excluding reduced preparation. Do not
use this small kernel comparison as the headline speedup. Charge simplification
and reduced-index construction to `resident_match`: these occur for each
haplotype and are part of the reduced algorithm. Charge candidate/interval/mask
computation to `method_from_reference`; loading a saved mask is not its
construction cost. Record compilation separately and exclude Numba compilation
from steady-state resident comparisons.

The first three-way comparison must use `max_mismatches: [0]`. Greedy currently
rejects positive budgets. Use AC 10 and the same scalar recombination/mismatch
parameters, called-site domain, reference, haplotypes and inference intervals.
Here `reduced_hmm` means the existing sample-specific regional reduced HMM plus
full-reference gaps, not a globally truncated reference or a whole-chromosome
reduced pass. `greedy` also includes its full-HMM gap work.

## Measurement design

### Process elapsed, user and system time

Add a small optional collector in `lib/performance.py`, using a dataclass for a
measurement record and a context manager for a stage. Snapshot
`time.perf_counter_ns()` and `resource.getrusage(resource.RUSAGE_SELF)` at the
start/end of a whole run and each non-overlapping process phase. Store seconds
as floating-point numbers. The process counters include all its threads; they
are not main-thread CPU time. Keep user and system seconds separately and derive:

- `cpu_s = user_s + system_s`;
- `effective_cores = cpu_s / wall_s`;
- `speedup = wall_s_at_1_worker / wall_s_at_p_workers` for the same method/mode;
- `parallel_efficiency = speedup / p`;
- CPU work inflation relative to the one-worker run.

On this Mac, `resource.RUSAGE_THREAD` is absent but
`time.thread_time_ns()` is available. Use that thread clock for parallel worker
phase CPU attribution. Do **not** put `RUSAGE_SELF` snapshots around concurrent
haplotypes and add their deltas: each would include other workers' CPU, causing
double counting. Worker-phase `user_s` and `system_s` must be blank; store their
combined `thread_cpu_s` instead. To obtain a stage's separate user/system costs,
run the corresponding resident operation serially with reader background work
stopped, using process counters.

Worker elapsed sums are **service time**, not elapsed critical-path time. They
can exceed total elapsed time. Keep them distinct from process wall time and
do not add them to get a multithreaded runtime. For overlap between background
reading and matching, the whole matching window's process counters are the
accounting total; worker and reader measurements explain that total rather than
forming additive process phases. Native codec/helper threads may also consume
CPU outside the measured Python worker, so worker CPU sums need not equal
process CPU.

Use a fresh subprocess for each trial, with preparation and compilation outside
the resident measurement window. The harness launches trials sequentially;
there should be one measured trial active on the machine. A new interpreter is
not a cold filesystem cache. Label cache conditions honestly and record order.
Also record launcher elapsed time and waited-child user/system deltas for the
entire subprocess, using `RUSAGE_CHILDREN` around launch/wait while no other
child trial is running. Keep these as `process_wall_s`, `process_user_s` and
`process_system_s`, distinct from the mode's timed-window fields. They include
interpreter imports and input preparation. Practical first-process results
include startup/compilation; resident steady-state results exclude them. Store
the startup policy explicitly. An external `time -l` measurement corresponds
to the whole subprocess, not just the inner resident window.

### Memory, I/O and cache pressure

Record process peak RSS and sample RSS at a configurable interval, initially
0.1 seconds, outside workers. On macOS `ru_maxrss` is bytes; normalise the stored
value to bytes and retain the platform in the manifest. It is a process-lifetime
high-water mark: never subtract two values and call the difference a stage peak.
Separate preparation and trial subprocesses make lifetime peaks interpretable.
Sampling gives approximate stage peaks; short allocation spikes may be missed.
For resident trials distinguish the preparation lifetime peak from the sampled
matching-window peak. Excluding extraction from time does not remove its peak
from `ru_maxrss`; that value cannot be labelled algorithm-only memory.

Record machine available memory and swap usage before/after the run, page-fault
and context-switch deltas where meaningful, and RSS before/after each stage.
Keep unsupported macOS disk-I/O counters blank, not zero. Whole-machine swap
changes are a pressure diagnostic, not proof that this process caused them.

Expose the existing HaplotypeReader/ScheduledCache statistics as structured
records: sample chunks loaded, physical Zarr chunks read, loads/hits/misses,
evictions, forced evictions, decoded bytes, cache high-water bytes, summed
load/wait service time and duplicate/stalled loads. Retain the existing logs as
a cross-check. Distinguish physical source-block decoding from the size of the
selected cohort and the retained cached array. Resident measurements must have
no active reader background loading in their timed window.

Snakemake's `benchmark:` output is useful as an additional rule-level resource
check, but its standard `cpu_time` combines user and system time and it samples
resources. It cannot replace the explicit counters or the stage records. Its
short sampling interval is 0.5 seconds in the installed implementation, so
sub-second masks and greedy kernels especially need explicit measurement.

### Stage boundaries to implement

Keep recording optional and off in ordinary runs. Put hooks at leaf operations;
do not replicate matching or interval algorithms in a profiler. Accumulate
records in memory and write them once after the timed computation. No worker
writes shared CSVs or logs an event for every site.

| File/function | Stages and quantities |
| --- | --- |
| `lib/utils.py::annotate_derived_counts` | `sample_store_copy`, `count_input_read`, `derived_count_compute`, `count_array_write`; selected individuals, source/selected rows, physical chunks and decoded bytes. |
| `lib/utils.py::make_inference_config` | `config_write`; persist the exact resulting TOML. |
| `lib/ancestors.py::infer_ancestors` | `ancestor_inference_total`; preserve native internal timing logs. This is common preparation, not sample matching. |
| `lib/matching.py::match_ancestors` | `ancestor_match_total`, `reference_write`; reference sites/nodes/edges, groups and serialized size. |
| `lib/haplotypes.py::find_focal_ancestors` | `focal_metadata`, `seed_call_extract`, `candidate_select`, `candidate_archive_write`; seeds, candidate associations and cohort identities. |
| `lib/haplotypes.py::_prepare`, `_block_bounds`, `_collect_bounds` | `interval_sample_extract`, `interval_association_prepare`, `ancestor_block_read`, `interval_sweep`, `interval_collect`, `interval_archive_write`; blocks, seeds, budgets and associations. Read/sweep events overlap across blocks, so use worker service/CPU records and one process total. |
| `lib/haplotypes.py::_write_focal_mask` | `mask_select_and_union`, `mask_archive_write`; cutoff/budget, association count, covered sites, focal/gap components and their lengths. Keep batched mask preparation as one measured job with identifiable per-mask work. |
| `lib/matching.py::match_stitched_samples`, `_prepare_stitched_inputs` | `reference_load`, `jobs_metadata`, `reader_prepare`, `focal_archive_load` or `interval_archive_load`, `mask_archive_load`, `ancestor_id_join`, `full_index_build`; split the currently combined setup measurements. |
| `_worker_stitched_haplotype`, `match_stitched_haplotype` | `full_matcher_init`, `sample_read`, `focal_select`, `reduced_simplify`, `reduced_node_map`, `reduced_index_build`, `reduced_matcher_init`, `focal_hmm`, `gap_hmm`, `greedy_select`, `path_merge`; aggregate regional calls by phase for each haplotype. |
| `match_stitched_samples` and the full-match benchmark runner | `result_collect`, `native_assembly`, `trees_write`, `jsonl_write`, `reader_shutdown`; matching and writing are separate. |

Instrument `find_path` calls separately from `_match_region`'s dense-parent
materialisation where that distinction matters; include both in method totals.
For greedy, initially time its selection/sorting/sweep as one per-haplotype
phase. Only split those further if it is a material cost. Do not refactor
production algorithms solely to obtain prettier timing buckets.

For the resident full baseline use native `tsinfer.matching.Matcher.match` with
the same native jobs and an in-memory reader. It must retain the native
per-haplotype matcher-construction policy. For reduced/greedy use the existing
stitched worker with that same resident reader. Include all composite gap calls.
This needs no reimplementation of the HMM or edits to its recurrence. TGP's
present-day samples form one native sample group; do not silently extend this
shortcut to ancient samples or dependent multi-source groups.

For practical full-HMM timing also wrap the existing `match_samples` leaf. Its
native `pipeline.match` total includes reader/index preparation, native matching
and assembly. If further native phase attribution is needed, add a minimal
optional timing hook to the editable tsinfer wrapper; do not infer exact CPU
phase totals from existing wall-time log lines. Retain both the deployed leaf
and the resident driver labels in results.

## Result files and accounting

Write benchmark output below `data/performance/{experiment_id}/`, with lowercase
underscore filenames and columns. Canonical analysis tables are plain CSV;
metadata and exact configurations are JSON/TOML/YAML. Keep raw records rather
than just medians.

| File | Grain and required fields |
| --- | --- |
| `manifest.json` | Machine/OS/Python/package versions; both repository commits, tracked diff/source hashes and extension build hash; core-group names/counts; power/cache conditions; original input paths and metadata/chunk layout; cohort and configuration hashes; workload order; compilation and codec-thread settings. |
| `datasets.csv` | One dataset/reference: `dataset_id`, `cohort_id`, requested BP bounds/width, actual inference interval BP, individuals/haplotypes, inference sites, ancestors, reference nodes/edges/bytes, genotype chunk shapes, common preparation IDs. |
| `runs.csv` | One trial: `run_id`, `dataset_id`, `method`, `mode`, `startup_policy`, `ac_cutoff`, `max_mismatches`, `workers_requested`, `workers_granted`, `matching_workers`, `read_workers`, `repeat`, `execution_order`, `status`, mode-window elapsed/user/system/CPU seconds, whole-process elapsed/user/system seconds, peak RSS bytes, preparation IDs, output/checksum paths and failure reason. |
| `stages.csv` | One run/phase/scope/haplotype: `run_id`, `stage`, `scope` (`process` or `worker_haplotype`), nullable `haplotype_index`, call count, elapsed seconds, nullable user/system seconds, nullable thread CPU seconds, start/end RSS and sampled peak where available. |
| `haplotypes.csv` | One run/haplotype: sample ID, ploidy index, population, called domain/sites, eligible interval count, reduced nodes/edges, focal/gap run count, focal/gap covered sites, region-call counts, final switches and mismatches, path checksum. |
| `cache.csv` | One run/source: cache/read-worker settings, block sizes, loads/hits/misses/evictions/stalls, decoded and peak retained bytes, load/wait service time. |
| `preparation.csv` | One common or heuristic preparation operation with elapsed/user/system/RSS, `preparation_id`, cohort/reference/cutoff/budget range, consumer runs, and whether it was reused. |

`run_id` identifies exactly one measurement, not just a dataset/method. Missing
or inapplicable values are empty CSV fields. Record failed/timed-out runs as
such; do not turn missing times into zeros. Sort output deterministically after
worker collection. Population joins use the saved pedigree and sample IDs.

Keep common reference construction outside three-way sample-method speedups.
For a standalone heuristic run, its prerequisite candidate/interval/mask work
must be charged once; full HMM has no such charge. Reusing preparation across
methods, cutoffs or repeats is useful for evaluation but does not make it free
in deployment. Link consumers with `preparation_id` and report both standalone
and explicitly amortised costs. Do not divide preprocessing by the number of
benchmark repetitions. Batch outputs across AC/budgets share real work; never
present their total as an independent single-cutoff measurement.

Compute an inclusive whole-run row directly. Do not reconstruct elapsed time
by summing overlapping worker or reader durations. An additive phase chart can
use exclusive process phases around the parallel-matching window; detail
within that window should use thread CPU/service time, separately labelled.

## Harness, workflow and verification

Add an explicit benchmark configuration and target; keep normal pipeline outputs
and `rule all` unchanged. A small driver in `lib` should launch one trial
subprocess at a time with a JSON configuration, collect its records, and assemble
the tables. Preparation is done before timed matching trials and shared through
immutable input paths. Snakemake grants the suite's maximum CPU allocation;
the driver passes the selected matching worker count to each trial. Do not let
Snakemake run competing benchmark jobs in parallel. Add rule-level benchmark
sidecars to the normal preparation rules as supplementary measurements.

Settings belong in the benchmark config: workloads, seed, populations, cutoffs,
budgets, repetitions, worker counts, cache capacity/read workers, memory-sampling
interval, time/RSS limits, output retention and run-order seed. Existing
`threads: workflow.cores` stitched scheduling must not override a benchmark's
requested worker count. Record granted and actual worker counts, not just CLI
`--cores`. Native background readers and Zarr/Blosc threads are additional
workers; record them and hold their settings fixed for the resident comparison.

Use AC 10/k 0 for the main comparison. Build/reference-match once per
cohort/window and generate one focal preparation per operating point. Use a
Numba warm-up with the production dtypes/signature before steady-state timing.
Record its separate compilation duration; do not use precision-zero HMM in the
baseline without creating a distinct fourth method.

Run three measured repeats per cell, with deterministically shuffled/interleaved
method/worker order for the main matrix; smaller scaling controls can use two.
Report medians and min/max or IQR. Run on mains power in a
stable power mode, without other CPU-intensive jobs, and record thermal/power
conditions where obtainable. A fresh reader means an empty application cache;
filesystem cache may remain warm. Do not use privileged cache purges or claim a
cold-disk comparison. For deployed I/O scaling, either hold `read_workers` fixed
or explicitly benchmark the native default policy and record it; the default
changes with matching-worker count.

Before the full matrix, check with ad hoc runs:

1. Timed and untimed execution produce identical path/mutation checksums.
2. Resident full matching agrees with the deployed native full baseline; use
   fresh results, not the saved pre-regional-matcher stitched outputs.
3. One- and multi-worker scientific outputs agree within each method. Decode
   called alleles outside timing and compare with the selected source calls.
4. Serial process-phase user/system sums approximately reconcile with the
   process total; parallel worker records are not incorrectly added to it.
5. Duplicate/shared preprocessing is charged once and unsupported counters
   remain missing. Sanity-check process CPU with `/usr/bin/time -l` on a pilot.
6. Measure instrumentation-on/off overhead. Keep per-haplotype phase aggregates;
   if timing tiny region calls materially changes runtime, move detailed tracing
   into a diagnostic run rather than the main benchmark.

Quality comparison is separate untimed analysis using native TS/JSONL outputs:
parent agreement, focal/gap/seam switches, mismatches and path scores. Different
references have different `n` in native transition probabilities; record that
fact and evaluate paths with one common full-reference model when comparing
their scores. Keep the performance experiment focused:
no new unit-test suite, storage redesign, boundary conditioning or algorithm
changes are part of this implementation.

## Initial sizing probes on this Mac

Measurement API references: [Python resource usage](https://docs.python.org/3.14/library/resource.html#resource-usage),
[Python thread CPU clock](https://docs.python.org/3.14/library/time.html#time.thread_time_ns),
and [Snakemake benchmark fields](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#benchmark-rules).

The 13 October probes use the actual source Zarr, existing 100-person cohort,
and a nested 300-person cohort with 100 CEU, 100 CHB and 100 YRI individuals.
The extension cohort was sampled with seed 20261013 and ordered by original
Zarr columns. All counts and panels are rebuilt for their own cohort: additional
individuals are not merely matched against the old 100-person panel.

Hardware: Apple M5 Pro, 64 GiB RAM, 18 physical/logical CPU cores. The OS labels
its groups as six `Super` and twelve `Performance` cores. There was roughly
28 GiB available RAM and 611 GiB available disk at the first probe. Do not
assume these are efficiency cores or that 18 workers always beat six.

The local chromosome-20 store contains 1,644,384 source variant rows, 3,202
diploid individuals and a chromosome length of 64,444,167 bp. Its genotype
chunks are `(20000, 3202, 2)`: a decoded physical block is about 122 MiB.
The native reader's cached sample chunk covers **all physical sample columns**
and all selected inference sites, so its minimum cache entry is
`num_selected_sites * 3202 * 2` bytes even for a 100-person cohort. Use an
explicit adequate cache rather than silently accepting the 256 MiB default.
The original store is about 289 MiB compressed; compressed disk size does not
predict transient decode memory. Source row counts for `[1 Mb, 11/21/31/61 Mb)`
are 254,189 / 502,784 / 744,714 / 1,508,151.

The following are exploratory single runs, not the proposed repeated benchmark
matrix. Resident measurements exclude Zarr
extraction, shared index construction, compilation, native assembly and output
writing; they include per-haplotype method preparation and full-HMM gaps.

| Individuals | Requested window | Inference sites | Reference nodes | Reference edges |
| ---: | --- | ---: | ---: | ---: |
| 100 | `[1 Mb, 11 Mb)` | 61,798 | 35,641 | 219,082 |
| 100 | `[1 Mb, 31 Mb)` | 150,643 | 80,962 | 510,664 |
| 300 | `[1 Mb, 11 Mb)` | 86,119 | 54,652 | 376,371 |
| 300 | `[1 Mb, 31 Mb)` | 214,879 | 127,453 | 922,379 |

The 30 Mb windows contain three native inference intervals totalling about
25.58 Mb, because large site-free gaps are excluded. All these tree sequences
still have the chromosome's 64.44 Mb `sequence_length`. Use the requested
window, inference intervals and site counts for normalisation, not that full
contig length.

Resident **wall seconds**, including composite gap matching:

| Individuals × requested Mb | Workers | Full HMM | Reduced HMM | Greedy |
| --- | ---: | ---: | ---: | ---: |
| 100 × 10 | 1 | 6.96 | 7.61 | 4.56 |
| 100 × 10 | 6 | 1.28 | 3.64 | 1.06 |
| 100 × 10 | 18 | 0.49 | 4.22 | 1.29 |
| 100 × 30 | 1 | 18.39 | 24.53 | 15.81 |
| 100 × 30 | 6 | 3.28 | 9.25 | 3.29 |
| 100 × 30 | 18 | 1.25 | 10.34 | 3.24 |
| 300 × 10 | 1 | 43.36 | 34.34 | 24.04 |
| 300 × 10 | 6 | 8.29 | 16.13 | 4.98 |
| 300 × 10 | 18 | 4.53 | 17.49 | 4.54 |
| 300 × 30 | 6 | 24.34 | 42.53 | 18.64 |
| 300 × 30 | 18 | 11.62 | 37.83 | 13.60 |

The main workload's process accounting illustrates why both CPU and wall time
are needed:

| Method | Workers | Wall s | User s | System s | Effective cores |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full HMM | 6 | 24.34 | 143.87 | 0.63 | 5.94 |
| Reduced HMM | 6 | 42.53 | 125.55 | 3.29 | 3.03 |
| Greedy | 6 | 18.64 | 107.14 | 1.39 | 5.82 |
| Full HMM | 18 | 11.62 | 188.69 | 0.98 | 16.33 |
| Reduced HMM | 18 | 37.83 | 141.83 | 14.70 | 4.14 |
| Greedy | 18 | 13.60 | 148.23 | 6.83 | 11.40 |

Greedy can win against full HMM at six workers while full HMM wins at 18.
The reduced method has substantially poorer CPU utilisation and higher system
costs as workers increase. Simplification holding the GIL, Python path work,
regional matcher resets and endpoint edge scans are candidate explanations;
measure their contributions before changing them. A fast focal kernel alone
has not established an overall advantage over a well-parallelised full HMM.

Measured preparation wall seconds, with eight requested native workers:

| Workload | Ancestor inference | Ancestor matching | Focal lookup | Interval generation | Mask |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 × 30 Mb | 30.75 | 40.95 | 2.86 | 2.76 | 0.059 |
| 300 × 10 Mb | 22.33 | 30.78 | 4.26 | 2.58 | 0.046 |
| 300 × 30 Mb | 74.84 | 214.14 | 9.93 | 8.54 | 0.140 |

The new 300-person copied store/count annotation took another 10.83 seconds,
once for both widths. These preparation rows include I/O; the first interval
sweep in each process can include compilation. They are sizing observations,
not isolated scan-kernel benchmarks.

The fresh process building and probing 300 × 30 Mb peaked at **8.09 GiB** RSS;
after resident trials it held roughly 2.7–3.5 GiB. The earlier chained sizing
process reached 12.22 GiB after several workloads. These are lifetime peaks,
not separate per-method peaks. The main resident sample-call matrix is only
123 MiB, whereas one native all-column cached sample chunk is **1.28 GiB**.
The probes used a 4096 MiB native cache, eight Blosc threads, Zarr asynchronous
concurrency 10, and the default Zarr thread-pool setting. Log these settings in
the actual benchmark rather than treating the matching worker count as the
whole thread budget.

Raw data: [pilot measurements](../data/performance/sizing_20261013/pilot_measurements.csv),
[dataset sizes](../data/performance/sizing_20261013/pilot_datasets.csv),
[manifest/configuration provenance](../data/performance/sizing_20261013/manifest.json)
and the [frozen nested 300-person cohort](../data/performance/sizing_20261013/tgp_chr20_n300.csv).
The temporary probe scripts and generated panels are under
`/private/tmp/tsinfer_performance_20261013/`. The formal suite should keep its
inputs under `data/performance/` and rebuild the old 100 × 10 Mb reference under
the same frozen generator version, or verify its generator fingerprints.
No system CPU affinity or thermal controls were enforced in these pilot runs.

## Recommended few-hour TGP benchmark

Use **300 diploid individuals (600 haplotypes), 30 Mb of chr20,
`POS >= 1000000 & POS < 31000000`** as the main workload. This size is directly
verified on this machine. Keep the frozen 100-person cohort nested within the
300-person cohort, with CEU/CHB/YRI counts fixed as above. Use
`variant_ancestral_allele`, AC 10, mismatch budget zero, recombination 0.01,
mismatch probability `1e-20`, path compression disabled, and native inference
intervals unchanged. Build the panel from all 300 individuals.

| Part | Workloads | Methods | Matching workers | Repeats | Trial count |
| --- | --- | --- | --- | ---: | ---: |
| Main resident matrix | 300 × 30 Mb | Full/reduced/greedy | 1, 6, 12, 18 | 3 | 36 |
| Sample/width controls | 100 × 10 Mb, 100 × 30 Mb, 300 × 10 Mb | Full/reduced/greedy | 1, 18 | 2 | 36 |
| Practical I/O comparison | 300 × 30 Mb | Full/reduced/greedy | 1, 18 | 2 | 12 |

Thus the initial suite has 84 trials, sharing four panel/reference preparations.
The controls distinguish widening the sequence from enlarging the cohort;
matching more samples to an unchanged small panel would answer a different
question. Three repeats at one operating point are more useful initially than
a large grid of AC/mismatch choices. Add AC 3/5 only after these costs are clear.

For the practical I/O cells retain the native reader-worker policy
(`max(1, matching_workers // 2)`) and record it. This measures the deployed
policy, including its coupling to matching workers. Resident cells use an
in-memory reader with no read-ahead workers. A later fixed-one-reader diagnostic
can separate I/O concurrency from matching concurrency; label it as a different
reader policy. Default to 4096 MiB cache for this workload and validate its
required entry size before a trial.

**Reserve two to three hours for preparation, runs and verification.** The main
reference preparation was under five minutes. The 6/18-worker resident cells
were 12–43 seconds each. Budget approximately 30–80 minutes for the full
resident matrix and input/compilation setup, another 20–40 minutes for practical
I/O trials, and 15–30 minutes for preparations/checks and run-to-run variation.
These are conservative planning allowances, not measured completion times for
the future suite. The main one-worker and 12-worker cells, output/assembly
costs and repeated operational reader costs remain to be measured. As a rough
resident budget check, twice the observed six-worker CPU totals gives serial
allowances of about 289 / 258 / 217 seconds for full/reduced/greedy respectively;
use those allowances only to schedule the experiment, not as performance data.

Use an initial one-repeat pilot gate before committing all repetitions. Put the
limits in the config: allow up to 15 minutes for a reference preparation and
10 minutes for an individual trial, with a 24 GiB process RSS budget. Retain
failed/timed-out records. Pause the suite when memory pressure or other active
work invalidates timings; close competing applications or reduce read-ahead
before rerunning affected trials. This leaves substantial room on a 64 GiB
machine without benchmarking through sustained swapping.

For a **width stress test**, the next extension is the same frozen 100-person
cohort across the entire chr20 source (`[0, 64444167)`), initially only one
resident repeat at 1 and 18 workers for each method. This was not directly
measured; extrapolating the observed 10→30 Mb growth suggests it is a reasonable
additional pilot, subject to the RSS/time gate. Extend the repeated matrix only
once that pilot passes. The verified first benchmark remains 300 × 30 Mb.
A 1000-person/full-chromosome suite has no measured few-hour budget here; it
needs a separate cohort/panel sizing pilot rather than extrapolation solely
from the number of haplotypes.

## Implementation order

1. Add process elapsed/user/system/RSS measurement records and deterministic
   CSV output; cross-check a serial native full-HMM run with `time -l`.
2. Add the resident three-method driver using the existing native/stitched
   matching calls, compilation warm-up and isolated trials. Run the 100 × 10 Mb
   smoke comparison and check scientific equivalence.
3. Add coarse pipeline preparation/assembly/output stages and per-haplotype
   thread-CPU phase aggregates, with no per-site event stream. Verify accounting
   and instrumentation overhead.
4. Expose cache/decode statistics, add common/heuristic preparation IDs and the
   practical I/O mode, and validate cache capacity and memory sampling.
5. Freeze input/configuration/code fingerprints, run the 300 × 30 Mb pilot gate,
   then the matrix above. Analyse the CSVs in a jupytext notebook by editing its
   `.py` in `~/work/dphil-analysis/notebooks/` when that analysis is requested.
