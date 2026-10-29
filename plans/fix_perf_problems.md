# Fix greedy matching startup, memory and initial sample loading

## Objective and constraints

Reduce unnecessary process startup cost, repeated reference memory and the
initial sample-loading barrier in greedy matching. Preserve the scientific
behaviour of every exact match, bridge and independent HMM gap. Optimise only
costs large enough to justify their implementation complexity.

Implement stages 1-4 and 6 in `tsinfer-match-eval`; Stage 5 makes a narrow
native change in `tsinfer`. Keep Full HMM's matching, reader, reference
construction and concurrency unchanged as the comparator. Use the existing
spawned processes, work/event queues, one haplotype per task, plain NumPy
arrays and temporary files; do not introduce a cache service, new worker
scheduler, fork-based inheritance, hybrid backend or elaborate monitoring
framework.

### Why the 40-46-second startup may improve, but is not yet solved

The previous 94-worker experiments recorded approximately **39, 45 and 46
seconds in `coordinator_process_startup`**. This is the coordinator's serial
loop constructing `multiprocessing` process objects and calling `start()`.
It is distinct from later reference packing and native matcher readiness.

- **Stage 3 directly targets that window.** Each spawned process currently
  receives six catalog arrays and a newly computed allele-count array in its
  process arguments. Moving these to read-only mapped files replaces repeated
  argument serialisation with small descriptors. A large gain is plausible,
  but not assured until measured.
- **Stage 6 chiefly reduces memory and readiness cost.** Sharing native edge
  indexes avoids building/owning two large edge buffers in every worker, but
  those buffers are constructed after packing, *outside* the measured process
  startup loop. Do not count a readiness gain as a startup gain. One-time
  index preparation also moves work into the coordinator's preparation stage.
- **Spawning 94 Python interpreters still has a cost.** Imports, interpreter
  initialisation, process creation and filesystem contention can remain even
  after array serialisation is fixed. Using fewer workers is a simple option
  for smaller cohorts. Do not switch multiprocessing backend speculatively.
- **Stage 4 addresses a different barrier:** initial sample decoding/loading,
  reported as 1.34, 7.40 and 61.91 seconds across the three cohorts.

Measure the launch loop before and after Stage 3 at the *same* worker count.
The goal is a large reduction in actual end-to-end elapsed time, not just a
smaller number assigned to a different stage. If process startup is still
unacceptable after shared descriptors, inspect measured spawn/import costs
before deciding whether any further process-level optimisation is justified.

The largest cohort also has a calculated **30.26 GiB of avoidable replicated
native edge buffers** at 94 workers. That strongly justifies Stages 5-6 as
memory work, but it does not establish any fixed reduction in total job peak.
The prior smallest-cohort memory trace has a long sampling gap and should not
be used as a reliable peak-memory comparison until reproduced.

## Implementation principles

Preserve canonical identities, allele mapping, exact-extension and tie rules,
focal regions, one-site bridges and mutation corrections, sequence/inference
boundaries, independent likelihood resets in every HMM gap, and existing
`reuse_topology=True`. Keep gap calls ordered inside their haplotype. Never
merge gaps across focal regions, restrict candidate parents, or change AF or
minimum-length parameters simply to gain speed.

Use small named dataclasses, existing queues, read-only mapped arrays and
straightforward ownership. Keep native topology, likelihoods, traceback,
matcher scratch and output state private per worker. Validate array dtype,
shape and lifetime where consumed; avoid repeated inner-loop checks,
compatibility layers and defensive fallback machinery. Preserve bounded
slabs/pending tasks, immediate result append and coordinator-owned cleanup.

Run repository Python commands with `uv run`. Check scientific output equality
using existing smoke/native/benchmark tests; do not build a large new test
framework or put noisy performance thresholds in CI. Each implementation
stage below has a short suggested commit message. Stage 7 is explicitly
untracked and has no commit.

## Stage 1: measure launch costs and make existing memory sampling usable

**Planned commit:** `Measure greedy process startup and memory`

**Files:** `tsinfer-match-eval/lib/matching.py`, `lib/performance.py`,
`lib/benchmark.py`, and concise benchmark documentation.

Extend the existing collector rather than adding another diagnostic system.

1. Keep start/end monotonic timestamps for the current coarse coordinator
   phases: input setup, catalog preparation, launch, packing, worker readiness,
   matching/drain, shutdown and final output. Keep initial-slab latency and
   loader service time separate as nested measurements; do not add them or
   nested worker service totals to coordinator elapsed time.
2. **Measure the serial launch loop explicitly.** For each worker, collect
   elapsed time for construction of `ctx.Process(...)` plus `process.start()`;
   additionally separate time spent inside `process.start()`. Save only a
   compact aggregate (total, median, maximum, and optionally the per-worker
   durations in one benchmark diagnostic file). This distinguishes repeated
   argument construction from slow process start calls without tracing every
   system call. The existing coordinator startup window remains the headline
   measurement. If the cause is still unclear after Stage 3, use a one-off
   child-entry timestamp or import-time profile; do not build persistent
   worker-bootstrap telemetry pre-emptively.
3. Move frequent benchmark memory snapshots to a trial-owned directory on
   node-local scratch; copy completed histories into the experiment directory.
   Store actual timestamps and indicate stale/missing samples and the longest
   observation gap. The old trace gap may reflect observer/filesystem issues,
   but that is not established.
4. Where `memory_full_info()` exposes PSS, record it alongside existing USS.
   Keep mapped-input capacity separate. Scheduler/job-memory reports may
   corroborate physical usage, but do not add general cgroup discovery or
   frequent per-worker memory inspection to this MVP.
5. Record actual sample chunk shape, number of haplotypes per slab, mapping
   capacity and temporary filesystem path using existing experiment metadata.

**Validation and decision:** check timestamp accounting, coverage of the
smallest run and unchanged scientific outputs. On the same 94-worker dataset,
obtain pre-change launch timing and compare it with Stages 3 and 6 without
mistaking phase shifts for reduced total wall time. A small configuration-only
pilot with fewer workers is worthwhile, especially for the 100-haplotype
cohort, but preserve a fixed-94-worker comparison for attribution. Measure
observer overhead once; keep expensive diagnostics out of headline runs.

## Stage 2: retain overall matching time for each haplotype

**Planned commit:** `Record per-haplotype matching times`

**Files:** `tsinfer-match-eval/lib/performance.py`, `lib/matching.py`,
`lib/benchmark.py`, benchmark configuration and concise README documentation.

Use one optional small timing record returned with each existing `HaplotypeResult`.
There is no need for per-gap records, worker-side CSV writes or a second queue.

1. Add a benchmark option `performance.collect_haplotype_timings`, independent
   of `collect_stages`, so cheap task timing does not require all regional
   timers. Default it off and keep ordinary matching unchanged.
2. In `_worker_main`, time exactly the call to
   `match_stitched_haplotype(...)`, after opening the slab and selecting its
   row and before closing the mapping or sending the result. Capture
   `perf_counter()` at both boundaries and a `thread_time()` difference. Reuse
   a PID obtained once in worker setup.
3. Return these scalar values with the result. The coordinator attaches the
   canonical identity from `jobs[result.job_index]` and retains only the small
   record after appending the scientific output. Do not retain result arrays
   or entire per-haplotype collectors. In particular, `Collector.merge()` can
   mutate retained `Measurement` objects: extract independent scalar values
   before merging, rather than saving references to those objects.
4. After the matching window closes, write one `haplotype_timings.csv` in the
   trial directory, ordered by canonical job index. Reference it in the trial
   result and carry it through the existing performance archive. Do not write
   a cumulative CSV after every result or add timing fields to scientific tree
   metadata.

Use the following compact schema:

| Fields | Meaning |
| --- | --- |
| `run_id`, `job_index` | Trial and canonical haplotype identity; use the experiment directory/ID as part of the key when combining suites. |
| `source`, `sample_id`, `ploidy_index` | Stable identity for joining existing haplotype diagnostics. |
| `worker_pid` | Worker grouping within this trial only. |
| `match_start_s`, `match_end_s` | Monotonic offsets from this trial's matching-window start, calculated after the run. |
| `match_wall_s` | Elapsed time inside this haplotype's matching call. |
| `match_thread_cpu_s` | CPU used by its executing thread within the same call. |
| `domain_sites`, `focal_sites`, `gap_sites`, `gap_calls` | Copy the already-computed per-haplotype counters; no new traversal. |

`match_wall_s` includes domain detection, candidate discovery/extension/cover,
bridges, all HMM gaps and construction of owned result arrays. It includes
page-fault or scheduling delays that occur inside that call. It excludes pool
startup, sample decoding, queue wait, mapping open/close, result transfer,
coordinator append and final output. Document this boundary beside the table.

For diploid individuals, report the **sum of their two haplotype service
times** when an individual-level value is wanted. Neither this sum nor their
maximum is automatically individual end-to-end latency. Similarly,
`run_wall / number_of_haplotypes` remains a throughput statistic, not the mean
of the new measurements. Never add summed haplotype durations to run wall time.

This is low overhead in design: a few clock reads and a few scalar values per
haplotype, versus potentially many HMM calls. Verify it once on representative
work rather than assuming a percentage. Collection can then stay enabled in
diagnostic benchmarks with `collect_stages=false`; use timing-disabled
headline trials if its effect is detectable.

**Validation:** exactly one row for every completed haplotype in a successful
trial; correct identities despite out-of-order completion; nonnegative,
consistent durations; identical scientific output with timing on/off. Older
archives have no individual timing data and must remain explicitly unavailable.

Full HMM already logs some per-job timers internally, but exposing a comparable
durable table would require additional Python result plumbing. Leave that out
of this MVP. Do not claim the new table supports paired per-sample Full HMM
comparisons or try to recover them from aggregate totals/debug logs.

## Stage 3: share the existing Python reference arrays

**Planned commit:** `Share greedy reference catalog arrays across workers`

**Files:** `tsinfer-match-eval/lib/haplotypes.py`, `lib/matching.py`.

This is the main code change directly targeting the 40-46-second process-startup
window. Currently all six `ReferenceCatalog` arrays are passed through spawn
arguments to every process; read-only flags do not prevent serialization and
duplication. The launch loop also recomputes `num_alleles_array()` for each
child. Replacing these arguments may cut launch time substantially if
serialisation/argument transfer is expensive, but interpreter bootstrapping and
imports remain.

1. Compute allele counts once. Write the catalog arrays and allele-count input
   once as ordinary `.npy` files in the existing greedy temporary directory.
   Preserve their dtypes; the current allele-count interface uses contiguous
   native `uint64`.
2. Introduce one small descriptor of named paths and required scalars. Pass it
   to workers and use `np.load(..., mmap_mode='r', allow_pickle=False)` to
   reconstruct the existing catalog. Keep packed genotype calls in their
   existing mapped format. No versioned storage format or generic mapping
   registry is needed.
3. Preserve the existing packing/readiness/matching sequence. Packing workers
   open the catalog inputs they need; matching reuses the mapped catalog.
   Native indexes still use the existing owned constructor until stage 6.
4. Charge file preparation to end-to-end greedy timing. Keep files alive until
   every worker has stopped. Extend this same descriptor later for native
   arrays instead of inventing a second prepared-reference abstraction.

**Validation and decision:** compare outputs at one and several workers;
confirm spawn arguments carry paths/scalars instead of large arrays and that
mapped input dtypes do not trigger hidden copies or new Numba signatures.
At 94 workers, compare the whole `coordinator_process_startup` window and
the per-launch breakdown from Stage 1 before/after this change, as well as
private memory. If launch time is still excessive, identify whether the
remaining cost is process creation/imports before adding another mechanism.
Do not attribute worker readiness or native index setup to launch time.

## Stage 4: shorten sample loading if it remains a material barrier

**Planned commit:** `Load greedy sample variant blocks in parallel`

**Files:** `tsinfer-match-eval/lib/haplotypes.py`, `lib/matching.py`,
`lib/utils.py`, `lib/benchmark.py`, `Snakefile`, example/smoke configuration and
README option documentation.

The reported 61.91-second initial load makes a small reader experiment
worthwhile. Preserve the source stores and Full HMM reader. Do not rechunk
shared benchmark inputs or split a wide source tile into repeated small reads.

1. Extract the current read/select/remap/write body for one physical variant
   block into a simple helper. Its writes cover a disjoint set of slab columns;
   preserve every selected sample and ploidy index. Keep decoded scratch local
   to that task and return only a count/completion result.
2. Add one greedy reader-count option, `sample_reader_threads`, defaulting to
   one. Pilot 1, 2 and then 4 readers only while gains warrant it. A small `ThreadPoolExecutor` inside
   the existing loader is appropriate for this read/NumPy work. Bound in-flight
   blocks to the small reader count; simple batches are acceptable.
3. Publish the slab only when all block writes finish. Retain the existing live
   slab bound, pending-haplotype bound, ownership and failure propagation. Do
   not allow tasks to read incomplete rows or retain all decoded tiles in
   futures/results.
4. Keep sample loading **after worker readiness** for this MVP. Currently
   `_pack_reference()` and the readiness loop consume the shared event queue
   without routing arbitrary event types. Starting the loader earlier would
   make `slab` events interfere with those protocols. Do not introduce an event
   router or buffering scheme merely to overlap these phases.
5. Account for reader threads and the actual Zarr/codec helper pools in
   `utils.cpu_allocation()` and both Snakemake/benchmark callers. Zarr's
   `async.concurrency` and `threading.max_workers` are separate limits, and the
   current configuration sets both to one. Extra outer threads may therefore
   have limited benefit. Start with fixed helper settings; any separate helper
   adjustment must use the existing options and stay within the allocation.
   Do not multiply a shared helper pool by reader count in accounting.

**Validation and decision:** compare slab calls and scientific outputs against
serial loading, including missing calls, non-reference ancestral alleles,
multiple physical chunks and a final short block. Preserve actual tile-read
counts and ensure publication waits for all writes. Measure load latency,
end-to-end wall and additional decode scratch at the same matching-worker
count. Keep the smallest useful reader count. Do not require a twofold gain or
retain parallel loading if contention/extra memory outweighs its benefit.

## Stage 5: tsinfer changes — borrow immutable edge indexes

**Planned commit:** `Allow matchers to borrow immutable edge indexes`

**Files:** `tsinfer/lib/ancestor_matcher.c`, `lib/tsinfer.h`, `_tsinfermodule.c`,
and the `MatcherIndexes` wrapper in `tsinfer/matching.py`.

This is a moderate ownership/API change with a clear memory justification.
It primarily affects memory and the later worker-readiness phase, *not* the
coordinator process-startup loop. Its scope is the two large edge arrays. The
existing owned constructor and Full HMM path remain unchanged; mutable topology, likelihoods, traceback and
site/mutation storage remain private.

1. Export the two prepared native edge arrays in their existing insertion and
   removal orders, with a documented contiguous `(num_edges, 4)` `int32`
   layout in left/right/parent/child order. Preserve the native coordinate
   representation exactly. Verify structure size/layout once at the boundary.
   Read-only views with a retained owner are suitable; a one-time copied export
   in the coordinator is also acceptable if it makes the implementation simpler.
   The important no-copy requirement is in every worker, not that one export.
2. Add one explicit prepared-array constructor. Borrow its read-only edge
   arrays without implicit dtype/layout coercion that recreates the large
   buffers. Accept the small numerical inputs/scalars needed to initialise
   current site data; reuse its existing allocation and initialization logic.
3. Preserve the existing reference assumptions: valid node/site IDs, root-zero
   semantics, current allele coding, positions and the terminal sequence-length
   entry, and existing mutation ordering. The native contract permits **at
   most one mutation per site**, including zero; do not accidentally tighten
   the owned API to exactly one or generalise it to arbitrary recurrent ARGs.
4. Make owned versus borrowed edge buffers explicit. Retain the borrowed
   Python arrays on the index object and free only owned native buffers.
   `AncestorMatcher` already retains its index object, so extend that existing
   ownership chain. No raw pointers cross processes, and a caller must not
   close a backing mapping while its index/matcher is alive.

**Validation:** use existing regional cases to compare owned and borrowed
paths/intervals/copied alleles, including missing calls, changing topology,
boundary gaps and consecutive haplotypes whose first gap moves backwards.
Exercise caller-reference deletion, output survival across later calls and
final cleanup. Open the same files in two spawned processes with independent
matchers. Verify the borrowed constructor avoids the two edge allocations;
use touched pages when assessing memory. These ownership checks are justified
by concrete use-after-free/double-free risks, not a general testing expansion.

## Stage 6: integrate shared native indexes into greedy matching

**Planned commit:** `Share prepared native indexes across greedy workers`

**Depends on:** stages 3 and 5. Stage 4 is independent and may be omitted.

**Files:** `tsinfer-match-eval/lib/haplotypes.py`, `lib/matching.py`, and greedy
benchmark workload accounting.

1. Build one owned native index from the coordinator's already-loaded
   reference. Export/write its edge arrays once, then release construction
   objects and export views before spawning workers. Include this work and
   its transient peak in preparation measurements.
2. Extend the stage-3 descriptor with edge-array paths, site/mutation arrays,
   node count, sequence length and the information needed to preserve
   `sequence_intervals`/inference boundaries. Removing the worker reference
   load must not discard those boundaries or site positions.
3. Replace each worker's `tskit.load(reference_path)` plus owned constructor
   with mapped inputs and the new borrowed constructor. Keep its private
   `AncestorMatcher`, exact-match state and output scratch. Preserve the actual
   read-only Numba warm-up and count it inside preparation.
4. Retain one pool for packing and matching. After all results, stop/join
   workers, release matcher/index owners and their views, then remove temporary
   files. Reuse the existing failure cleanup. Remove the old greedy worker
   construction path once replaced; do not keep two selectable backends.
5. Include all new mappings in the mapped-input byte account. That field
   currently counts packed calls and sample slabs and would otherwise become
   incomplete. The coordinator still needs its reference/output tables; do
   not promise their removal.

**Validation and decision:** compare finalized scientific tables and metadata
by canonical identity, ignoring provenance timestamps. Preserve called alleles,
parents, focal/bridge decisions, boundaries and native gap-call counts. Confirm
one owned index construction in the coordinator and no reference tree load or
owned edge construction in matching workers.

At fixed reference and worker count, demonstrate removal of the duplicated
edge buffers and report USS, sampled PSS, shared capacity and any available
job-memory measure, including preparation. Use a smaller and a larger worker
count if needed to establish the reduced replication. Do not require an
arbitrary 25% reduction in total peak: the other allocations and the timing of
the peak determine that percentage. Retain the change when it provides the
intended material memory benefit without an unexplained matching regression.

## Stage 7: create `config.yaml.hpc_perf4` for the follow-up experiment

**No commit.** Create this file locally by copying `config.yaml.hpc_perf3`.
Keep `config.yaml.hpc_perf4` **untracked**; do not modify or commit the original
configuration or scientific inputs.

Use the **same dataset entries and cohort selection** as `hpc_perf3` (the same
50-, 500- and 5,000-diploid-sample cohorts), with unchanged input paths,
ancestral references, AF cutoffs, minimum-region settings, HMM parameters,
helper limits, cache and scratch policy. In the copied configuration, set:

```yaml
performance:
  experiment_id: hpc_perf4
  methods: [full_hmm, greedy]
  modes: [from_reference]
  worker_counts: [94]
  repeats: 1
  collect_stages: true
  collect_haplotype_timings: true
  quality_checks: true
```

These are the values to **update within the existing `performance` block**,
not a replacement for its other required settings: retain the same
`performance.datasets`, seed, limits, memory sampling interval, quality
budgets and timeout. If `performance.diagnostics_run` is present, set it to
`{mode: from_reference, workers: 94, repeat: 0}`. Set the new
`sample_reader_threads` to the simplest useful count retained from Stage 4
(or `1` if parallel loading was dropped).

This gives **one run per cohort/method** at the same 94-worker concurrency,
with both methods measured under the updated observer. It is a focused
post-change check of startup, readiness, initial-slab latency, elapsed wall,
CPU and memory, plus the new greedy per-haplotype distribution. Compare the
matched configuration and scientific quality checks with `hpc_perf3`; a
single replicate is exploratory, not proof of a small performance difference.
Do not add analysis-code changes, a benchmark scheduler or a worker-count
sweep to this stage.

## Work deliberately deferred

The largest prior cohort reported only about 83 worker CPU seconds in
candidate discovery, extension and cover, versus roughly 12,730 seconds in
matching CPU. Do not prioritise focal-run vectorisation, new object pools,
queue compression, changing integer widths across the pipeline, or per-gap
instrumentation. Similarly, do not bit-pack already shared ancestor calls or
invent another sample storage format without a new measured bottleneck.

Do not overlap sample loading with packing/readiness in this MVP. The current
packing and readiness phases consume the shared event queue under phase-specific
assumptions; starting the loader earlier requires event routing and adds
complexity that is not warranted before the simpler changes are measured.

Keep native HMM algorithms and gap semantics unchanged. If startup remains
excessive after Stage 3, use the Stage 1 launch measurements to isolate imports,
process creation or filesystem contention; only then consider a separate,
minimal follow-up. If HMM matching remains the dominant cost after this MVP,
profile a representative slow haplotype before changing native code.
