# Performance testing for sample matching

## 1. Objective and scope

Compare the full HMM, the sample-specific reduced HMM plus full-HMM gaps, and the exact greedy cover plus full-HMM gaps. Answer three questions: which complete method is fastest, where its major costs lie, and how time and memory change with sequence length, cohort size, and worker count.

Keep this an opt-in experiment, not a profiling framework. Follow `AGENTS.md`: implementation belongs in `lib`, settings belong in `config.yaml`, use explicit intermediate variables and dataclasses, and run Python tooling through `uv run`. Do not add a unit-test suite; use the manual checks below. Do not change matching algorithms as part of this plan.

Implement this layer only after all T1–T3 and M1–M3 steps in `~/work/tsinfer/plans/improve_performance.md` are complete and verified. Assume stitched matching already uses persistent native topology, region-sized native path buffers, compact `CopyingRuns`, and bounded haplotype batches with append/finalise assembly. Preserve those execution paths and their whole-haplotype threading model. Ordinary full-HMM matcher setup and its default `reuse_topology=False` policy remain as defined by that plan. [8]

The initial comparison uses `ac_cutoff: [10]`, `max_mismatches: [0]`, and `focal_methods: [hmm, greedy]`. Preserve the current sample selection, allele encoding, HMM probabilities, inference intervals, missing-call handling, and native output semantics. `reduced_hmm` is the benchmark label for the existing `method="hmm"`; it does not mean a globally pruned reference. [1]

## 2. Two timing boundaries

| Comparison | Included in the measured window | Excluded |
| --- | --- | --- |
| `resident_match` | Complete path and mutation computation for all selected haplotypes: worker matcher creation, per-haplotype simplification and reduced indexing, topology resets/advancement, focal matching, full-HMM gaps, stitching, and collection of completed batches. | Input extraction/loading, shared full-reference index construction, Numba warm-up, native tree assembly, and file writing. |
| `from_reference` | Everything needed to produce sample TS/JSONL outputs from an existing ancestor reference and annotated sample store. For either focal method, include candidate lookup, interval generation, masks, loading, matching, assembly, and writing. Include first-use compilation if it occurs. | Interpreter imports, count annotation, ancestor inference, and ancestor matching. |

`resident_match` implements the initial experiment with Zarr extraction outside timing. `from_reference` prevents that experiment being mistaken for an end-to-end speedup. This follows the meeting's distinction between establishing matching behaviour and subsequently addressing storage. [2]

Use one directly measured total for each comparison. Never reconstruct elapsed runtime by adding overlapping worker timings. In `from_reference`, regenerate the focal prerequisites for that trial and charge them once; do not divide their cost by benchmark repeats or pretend cached masks were free. Reuse only the common annotated store and ancestor reference. Keep the initial suite to one cutoff/budget so shared parameter-sweep accounting is unnecessary.

Common reference construction may have one coarse preparation timing in its existing log, but is not a third benchmark mode or part of a sample-method speedup.

## 3. Small measurement layer

Add `lib/performance.py` with a measurement dataclass and a small timer/context manager. Pass an optional collector explicitly; ordinary runs leave it `None`. Keep collected records local to the trial or haplotype task, and merge them in the coordinating thread. Write measurement files only after timing ends.

For each whole measured window, record:

- Wall seconds using `time.perf_counter()` and process user/system seconds using `resource.getrusage(resource.RUSAGE_SELF)` differences.
- Approximate peak RSS in bytes, sampled with `psutil.Process().memory_info().rss` at a configurable interval, including a sample at each boundary.
- The workload and worker counts needed to interpret the result.

Add `psutil` as a direct dependency because this repository will use it directly. Sample the child process from the launcher, outside the matching process and its GIL. Keep timestamped samples in the launcher; the trial returns measured-window start/end timestamps and boundary RSS values so the launcher can select the relevant samples. Record both `window_peak_rss_bytes` and `trial_peak_rss_bytes`: the latter includes preparation. Both are sampled maxima, not exact allocation peaks; short spikes can be missed. Do not use `ru_maxrss`, subtract memory high-water marks, or add platform-specific RSS conversions. [3]

The process CPU counters include matching and reader threads. They must not be sampled independently around concurrent haplotypes and summed. For stitched sub-stages, collect only elapsed service time and `time.thread_time()` differences, aggregated per task, merged when each batch completes, and then accumulated per stage. Release task-local measurement records with the batch in production; retain only run-level aggregates. These are worker diagnostics, not additive components of parallel wall time; worker CPU also excludes helper-thread work. [4, 8]

Use these standard timing and memory APIs unchanged on macOS and Linux, without CPU-vendor branches. The accounting assumes threaded matching within one trial process. If matching later introduces worker processes, include their CPU usage exactly once and sample aggregate live-process RSS; label summed RSS explicitly because shared pages can be counted more than once. [3, 4]

### Only record these major stages

| Stage | Boundary |
| --- | --- |
| `focal_preparation` | In `from_reference`, all of `find_focal_ancestors`, `make_ancestor_intervals`, and mask construction together, including their I/O. |
| `matching` | The complete resident path computation. In stitched `from_reference`, aggregate the non-overlapping batch execution/collection windows, including matching executor startup/shutdown, while excluding intervening assembly/output. For the deployed full-HMM leaf in `from_reference`, use an inclusive `full_pipeline` stage instead of trying to split native internals. |
| `simplify` | The existing `simplify_ancestors_ts` call, including its node mapping. |
| `reduced_setup` | Reduced `MatcherIndexes` and reduced matcher construction together. |
| `focal_match` | Complete reduced `_match_region` calls, including topology advancement, run conversion, original-parent mapping and owned mutation construction; or the complete greedy selection/sweep and compact run creation. |
| `gap_match` | Complete full-reference `_match_region` calls, including topology advancement through skipped focal spans, run conversion and owned mutation construction. |
| `stitch` | Final concatenation/coalescing of ordered copying runs, conversion to native `PathSegment` objects, and collection of mutations in site order. |
| `assembly_output` | Stitched table initialization, per-batch `append_match_results` and JSONL writing, then one `finalise_match_tables` and TS dump per budget, in `from_reference`. Aggregate these non-overlapping windows into one record. |

The five stages from `simplify` through `stitch` are diagnostics inside `matching`, not additional charges. Aggregate repeated regional calls into one record per stage, not a region event stream. Keep the entire run-based `_match_region` wrapper inside its corresponding HMM stage; do not recreate dense parent/copied-allele arrays for measurement. Native advancement, dense probability resets and the node-sized traceback parent copy remain inside the HMM calls, without new sub-stages. Gap cost therefore includes traversal across skipped focal spans, not just work at gap sites. Preserve left-to-right calls, independent full/reduced topology, and resets for every haplotype/budget. [1, 8]

Production batches interleave matching and assembly. Measure their coarse process stages in the coordinating thread while matching tasks are quiescent during assembly; background reader CPU may still fall in either stage. These stage aggregates do not replace the single directly measured `from_reference` total. Reader waits, topology resets and worker startup remain in the total; there is no requirement to force every second into a named bucket. Do not instrument individual sorts, dictionary joins, cache hits, or array writes.

## 4. Reuse the current execution paths

Add `lib/benchmark.py` as the suite driver and trial entry point. Its responsibility is selecting a workload, running an existing method, and writing measurements, not implementing matching.

### Resident trial

1. Load the native configuration/reference and build sample jobs using the same native job construction as `match_stitched_samples`. Preserve the native haplotype identities and order. Initially support the present-day, single sample-group workload; reject other group structures rather than silently flattening them.
2. Use the native `HaplotypeReader` to obtain canonical, contiguous `int8` haplotypes. Store them in a small in-memory reader implementing `read_haplotype(job)`. Retain the allele mapper and allele-count array for subsequent assembly. Shut down and release the Zarr reader and its cached data before timing. Do not replace native allele mapping with a separate genotype decoder.
3. For full HMM, construct native `tsinfer.matching.Matcher` before timing, then time consumption of its complete `match(...)` iterator. Keep its native per-haplotype matcher policy. For stitched methods, prepare shared inputs with `_prepare_stitched_inputs` before timing, but leave all per-haplotype reductions and worker matcher creation inside timing.
4. Extract the post-M3 serial/threaded batch execution into one plainly named helper used by both the production leaf and resident driver. It yields completed, canonically ordered batches, submits at most `haplotype_batch_size` jobs at once, and keeps one executor and thread-local worker state alive across batches. Preserve `_worker_stitched_haplotype`, `match_stitched_haplotype`, reusable worker scratch, and topology lifetimes/resets; do not copy their algorithms. Production consumes, appends and releases each batch before requesting the next. Add the optional stage collector at the boundaries above.
5. Warm only the Numba signatures used inside the measured window, including the run-emitting greedy sweep when applicable, using a tiny valid input with the production dtypes/layouts and output-array arguments. Discard warm-up outputs and start with fresh worker state. Do not warm by matching the full measured cohort.
6. Time matching to completion and collection of all results. The resident consumer retains the completed compact results in canonical batches until timing ends; keep job submission bounded and worker state alive as in production. Stop the timer, then consume those batches through the same production assembly helpers: initialize tables from the original reference, append results and write JSONL in canonical batch order, then finalise/dump once per budget. Match throughout against the original immutable reference, never the growing sample tables. Perform quality checks outside this window. Full HMM retains its native assembly path. [1, 5, 8]

Resident mode deliberately requires the selected sample matrix and all completed compact match results to fit in RAM. Retaining results is necessary to keep assembly/output outside one continuous matching window; it is specific to this diagnostic consumer. Record this retention policy in the manifest and interpret resident RSS accordingly: it does not demonstrate production's bounded result memory. Fail clearly on an unsuitable configured workload; use the already-batched `from_reference` path for scale runs. Final output tables, masks/associations, and existing reader caches still have the cohort/reference-sized limits documented in M3. [8]

### From-reference trial

For full HMM, call `lib.matching.match_samples`. For reduced/greedy, run the existing focal preparation functions into that trial's directory, then call the post-M3 `match_stitched_samples` with the configured `haplotype_batch_size`. Keep its one reader/schedule, one executor, per-batch result release, open JSONL handles, and one final table assembly per budget. The outer timer starts before method-specific preparation and stops after finalisation/dumps, output closure, executor shutdown and reader shutdown. Charge all interleaved matching/assembly and first-use compilation to this total. Retain existing caches and their logs across batches; do not add detailed cache instrumentation. [8]

Expose `read_workers` as an optional leaf argument and pass it through to native `pipeline.match` or `HaplotypeReader`, both of which already support it. Hold it fixed across matching-worker comparisons. Avoid monkey-patching or changes to tsinfer merely for timing attribution. [6]

## 5. Configuration, workflow, and outputs

Add a `performance` section to `config.yaml.example`. Include `experiment_id`, selected dataset names, `modes`, `methods`, `worker_counts`, `repeats`, `seed`, `read_workers`, `rss_sample_interval_s`, `trial_timeout_s`, and codec/helper-thread settings. Reuse dataset selection, HMM, AC/budget, cache settings and the existing top-level `haplotype_batch_size` rather than duplicating them. Keep batch size fixed across stitched method/worker comparisons and large enough to occupy the requested matching workers when the cohort permits. Defaults and experimental thresholds belong in configuration, not Python constants.

Add an explicit `performance` Snakemake target. Leave `rule all` and ordinary output locations unchanged. The suite rule depends on common prepared inputs and reserves the largest requested allocation; it launches exactly one trial subprocess at a time using the active `uv run` environment. Deterministically shuffle method/worker combinations within each repeat. Each trial gets its own output directory and log. A timeout or exception produces a failed row, not a zero runtime.

Do not let the stitched rule's workflow-wide thread allocation override a trial's requested worker count. Pass the trial value explicitly and record the actual count after capping by available jobs, including any batch-size limit on concurrent matching tasks. Refuse a worker setting exceeding the scheduler-granted allocation, and account for reader and codec/helper concurrency within the resource budget. Do not choose worker counts from all CPUs visible on the host. A fresh trial process does not imply a cold filesystem cache. [7, 8]

Write under `data/performance/{experiment_id}/`:

| File | Contents |
| --- | --- |
| `manifest.json` | Resolved configuration; both repository commits and dirty flags; Python/package versions; machine, CPU and OS; allocation; input paths/cohort identity; requested BP window and actual inference intervals; reference sites/nodes/edges; cache/reader/helper-thread settings; `haplotype_batch_size`, topology-reuse policy by method, result-retention policy by mode, and compilation policy. Copy the YAML and generated TOML alongside it. |
| `runs.csv` | One row per trial: unique run ID, dataset, method, mode, repeat/order, requested/actual workers, stitched `haplotype_batch_size` (empty for full HMM), status, wall/user/system seconds, both RSS maxima, output paths and error text. Also store matched haplotypes/domain sites, focal/gap sites and call counts, number of reductions, and summed reduced nodes/edges. |
| `stages.csv` | One row per run/stage/scope: call count, process wall/user/system seconds or worker service/thread-CPU seconds. Inapplicable fields are empty. No per-site or per-haplotype performance table. |

Use lowercase underscore names. Keep raw trial rows; analysis can calculate medians, spread, `cpu_s = user_s + system_s`, `effective_cores = cpu_s / wall_s`, and full-HMM/method wall-time ratios for identical workloads, modes, and allocations. Do not normalise by the reference's contig-wide `sequence_length` when only a smaller window was selected.

## 6. Initial local job outline

Use 300 diploid individuals, balanced across CEU/CHB/YRI, and chr20 `1000000 <= POS < 31000000`; freeze the selected IDs and rebuild counts/reference for that cohort. Use the existing ancestral-allele field, AC 10, budget 0, recombination `0.01`, mismatch `1e-20`, and no path compression. Start with one pilot, then three repeats at 1 worker and one sensible parallel allocation; use a 4096 MiB matching cache only after confirming it fits the selected workload. Run resident comparisons first and a small from-reference comparison second. Extend separately to a longer sequence and a larger cohort, rebuilding the corresponding panel. No large parameter grid is needed.

## 7. Implementation and acceptance order

1. Add the configuration, measurement helper, sequential trial launcher, and three output files. Smoke-check one native full-HMM trial.
2. Add the shared batch execution helper, resident reader, and stitched stage aggregates. Check that enabling measurements does not alter paths or mutations, recreate dense paths, or change topology resets/worker reuse. Confirm that production still releases each batch's results and measurement records.
3. Add from-reference orchestration and reader-worker pass-through. Verify that focal preparation is inside its total and outside resident timing.
4. On a small dataset, compare resident versus deployed outputs within each method, serial versus parallel outputs, one batch versus several batches, and reconstructed called alleles against input. Compare scientific tables and canonical records by haplotype identity; do not compare raw file bytes containing timestamps/provenance. Check all-gap, all-missing and inference-boundary cases, alternating focal/gap regions, and consecutive haplotypes across a batch boundary on a reused worker. Use the post-optimization verified outputs as the integration reference; investigate any changed paths, mutations or seams. [8]
5. Check accounting and timing overhead with instrumentation on/off. Confirm that interleaved assembly/output is inside `from_reference` and outside `resident_match`, with no double charge in stage aggregates. Check that resident result retention and production batch release match their documented memory policies. Worker service sums may exceed total wall time. If very short regional timers materially perturb the run, collect the breakdown in a separate diagnostic trial and keep headline totals from total-only trials.
6. Run the local pilot, then the selected repeats. Report quality separately from timing: path agreement, switches, mismatches, and likelihood under one common full-reference model. Different reference sizes alter native transition probabilities, so scores from each method's own reference are not directly comparable. Do not require approximate methods to choose identical parents. [2, 5]

Stop when the tables explain overall speed, simplification versus matching, gap costs, parallel scaling, and memory. Further instrumentation requires a specific unanswered question.

## Source references

Repository sources and the prerequisite optimization plan were inspected on 13 October 2026. Implement against the local sources after that plan is complete; the linked current implementations identify entry points, while [8] defines the assumed changes. These are implementation instructions, not new benchmark results.

[1] [Evaluation matching implementation](https://github.com/Duncan-JR/tsinfer-match-eval/blob/main/lib/matching.py).

[2] Supplied `2026-10-01_gil_yan_jerome.md`, especially "Refining the unmatched gaps" and "Baseline implementation first"; supplied `AGENTS.md` governs implementation style.

[3] [psutil process CPU and memory APIs](https://psutil.readthedocs.io/stable/).

[4] [Python resource accounting](https://docs.python.org/3/library/resource.html) and [time clocks](https://docs.python.org/3/library/time.html).

[5] [Native matching on fix_reference_ts](https://github.com/Duncan-JR/tsinfer/blob/fix_reference_ts/tsinfer/matching.py).

[6] [Native pipeline](https://github.com/Duncan-JR/tsinfer/blob/fix_reference_ts/tsinfer/pipeline.py) and [VCZ reader](https://github.com/Duncan-JR/tsinfer/blob/fix_reference_ts/tsinfer/vcz.py).

[7] [Current Snakefile](https://github.com/Duncan-JR/tsinfer-match-eval/blob/main/Snakefile).

[8] [Efficient stitched matching MVP](../../tsinfer/plans/improve_performance.md), all tsinfer T1–T3 and tsinfer-match-eval M1–M3 steps.
