# Further performance improvements

## Scope

Remove repeated diagnostics preparation, expose enough independent diagnostic tasks,
and reduce Python allocation and output overhead in greedy matching. All changes
below belong in `tsinfer-match-eval`. No changes to `tsinfer` are proposed: the
existing regional matcher already supports the required topology reuse.

Shared-helper changes may also improve full-HMM benchmark output assembly when
that is the simplest implementation. Do not change full-HMM matching, its worker
state, or its timing boundaries.

Preserve sample identities, allele encoding, exact-extension stopping rules,
stable ties, touching-union filtering, independent gap likelihoods, coordinates,
missing-data handling, diagnostic denominators, and archive contents. Replace
implementations directly; do not add compatibility modes or tuning options.

Small synthetic checks support the proposals below. They establish selected
correctness properties, not performance gains. Measure actual sample chunk widths
and stage timings before attributing a large run's duration to any one bottleneck.

## Stages and planned commits

| Stage | Planned commit name | Work removed |
| --- | --- | --- |
| P1 | `Prepare and share diagnostic copying paths once` | Repeated tree loading, validation, sorting and retained path arrays in each diagnostics worker. |
| P2 | `Distribute diagnostics by haplotype and release temporary slabs` | Analysis limited by physical sample chunk count; repeated baseline support construction; forced temporary-map writeback. |
| P3 | `Append matching edge columns in bulk` | One Python/native call and scalar conversion per output edge. |
| P4 | `Simplify greedy assembly and retain reference topology` | Singleton focal arrays and unconditional topology clearing per haplotype. |

Implement P1 before P2. P3 and P4 are independent small changes. Each stage should
remain reviewable on its own, with output equality checked before proceeding.

## P1. Prepare and share diagnostic copying paths once

**Relevant code:** `lib/diagnostics.py`: `read_paths()`, `_initialise_worker()`,
`WorkerData`, and `run()`.

Currently every diagnostics worker loads the reference and calls `read_paths()`
for the baseline and every stitched minimum. The pool is recreated for each AF.
With W initialised workers and M minima, this repeats path preparation
W × (1 + M) times per AF, including in workers initially used for packing.
`read_paths()` already sorts and indexes edges once per tree; the problem is
repeating that work across processes.

### Implementation

1. Call the existing `read_paths()` in the coordinator. Preserve its site-axis,
   reference-node, roster, duplicate-identity, parent and overlap checks. Keep
   its existing edge selection and sort; no second indexing algorithm is needed.
2. Pack its returned paths in canonical roster order into four temporary `.npy`
   arrays: `offsets: int64[H + 1]`, `left: float64[E]`, `right: float64[E]`, and
   `parent: int32[E]`. Compute offsets from path lengths. Allocate the three
   destination arrays with `np.lib.format.open_memmap()` and copy each path into
   its slice, avoiding another full concatenated edge array in coordinator RAM.
   Release the source path dictionary after packing each tree.
3. Add one small dataclass describing the index location, with helpers to open
   its arrays read-only and retrieve an `EdgePath` for a canonical job index.
   Use `np.load(..., mmap_mode="r", allow_pickle=False)`. Empty paths have equal
   adjacent offsets. Resolve identities through the existing dictionary: sample
   node IDs and iteration order need not match canonical job order.
4. Prepare the baseline index once per dataset, alongside the already prepared
   `SiteAxis`, outside the AF loop. Prepare each stitched index once per AF.
   Use dataset and AF temporary-directory scopes so files outlive their pools.
5. Pass the axis and small index descriptors into `_initialise_worker()`.
   Workers open the indexes and allocate private exact scratch; they no longer
   load reference/output trees or build per-haplotype path dictionaries.
   Keep the existing reference-packing barrier.

This retains approximately 20E bytes of shared edge columns per tree, plus
8(H + 1) bytes of offsets, instead of separate edge columns per worker. It does
not eliminate coordinator preparation scratch, mapped pages, or later archive
compression loads. Continue materialising dense parents only for the haplotype
being analysed. Do not introduce a cohort-by-site parent matrix or a persistent
cache.

### Validation

Compare mapped slices with `read_paths()` by canonical identity, including
reordered sample nodes, empty paths, touching edges and physical gaps. Preserve
all existing validation failures. Compare diagnostic rows and masks by canonical
keys. Confirm one baseline preparation per dataset and one preparation per
stitched output, independent of worker count.

## P2. Decode chunks once and analyse individual haplotypes

**Relevant code:** `lib/diagnostics.py`: `_analyse_chunk()`, `comparison_metrics()`,
and `run()`; `lib/haplotypes.py`: `load_sample_chunk()`, `pack_reference_chunk()`.

`pool.map(_analyse_chunk, samples.chunks)` permits at most one analysis worker per
physical sample chunk. A wide chunk can hold many haplotypes while occupying only
one worker. `load_sample_chunk()` already provides the required boundary: a
`SampleSlab` of contiguous complete haplotype rows and their canonical job indices.

### Implementation

1. Split `_analyse_chunk()` into a decoding wrapper and `_analyse_haplotype()`.
   Use a small task dataclass containing the slab's `MappedCalls`, row and job
   index. Do not copy genotypes into task messages or reuse the greedy task type
   merely to carry its unused chunk field.
2. Open the slab read-only in the task and compute the called domain,
   `greedy.candidates()`, baseline parent array and baseline support once.
   Process all minima for that haplotype together. Pass baseline support into
   `comparison_metrics()` instead of rebuilding it for every minimum.
   Common support, valid comparisons and switch counts remain minimum-dependent.
3. Fetch paths from P1 by job index. Return the existing `HaplotypeDiagnostic`
   results, ensuring retained arrays own their data rather than referencing
   reusable scratch. Release the row view and close the slab mapping in a
   `finally` block. The coordinator, not individual tasks, owns slab deletion.
4. In the coordinator, collect complete sample chunks until their selected
   haplotype count reaches W, or input ends. Decode that wave through the existing
   process pool, then submit one analysis task per row through the same pool.
   Keep `ProcessPoolExecutor.map`'s one-task chunksize so scheduler batching does
   not recreate the chunk-level parallelism limit. Fill canonical result slots
   as results arrive from the iterator.
5. After all analysis tasks in a wave finish, unlink its slabs. Then start the
   next wave. On failure, let executor shutdown finish before temporary-directory
   cleanup. Nest the pool inside the directory scope; no custom cancellation
   scheduler is needed.

For nonempty chunks, the wave holds fewer than W + Hmax haplotypes, where Hmax is
the largest selected chunk. Its decoded storage is that count times the number
of inference sites in bytes. A physical chunk spanning the cohort still creates
a cohort-sized slab. This bounds decoded input, not the diagnostic results that
are already retained for output. Decode/analysis barriers are intentional: avoid
nested pools, continuous producer/consumer machinery, and new queue controls.

### Temporary-map cleanup

Remove only `slab.flush()` in `load_sample_chunk()` and `packed.flush()` in
`pack_reference_chunk()`. These are temporary shared mappings on the single-host
Linux/macOS execution path. Consumers need completed writes visible through the
mapping, not durable storage. Retain mapping closure, completed-task handoff and
the reference-packing barrier. Closing a mapping and waiting for its writer are
still required; removing explicit flushes does not eliminate eventual disk I/O.

These shared helpers serve greedy preparation and diagnostics. Keep durable
output writes intact. Do not add platform modes or a flush configuration option.

### Validation

Use both fewer sample chunks than workers and many small chunks, plus a short
final wave. Count physical genotype tiles decoded: each selected chunk must still
be decoded once per AF. Compare every haplotype/minimum row and mask with serial
execution; verify results occur exactly once and earlier masks survive scratch
reuse. Check slab cleanup on success and a worker error. Compare packed reference
and sample calls with and without explicit flushes under multiple workers.

## P3. Append numerical edge columns in bulk

**Relevant code:** `lib/matching.py`: `append_result()` and greedy finalisation;
`lib/benchmark.py`: the resident baseline's call to `matching.append_result()`.

Replace the per-edge `add_row()` loop with:

```python
tables.edges.append_columns(
    left=result.left,
    right=result.right,
    parent=result.parent,
    child=np.full(len(result.left), child, dtype=np.int32),
)
```

Keep node creation, mutation decoding and mutation appending unchanged. Append
completed results immediately; retain the existing final squash and sort. Do not
buffer the whole run or add a second append API just to isolate the benchmark.

This simplifies a shared local helper and also improves resident full-HMM output
assembly. That assembly occurs outside its timed matching window, so this is not
an improvement to the full-HMM kernel or reported resident matching time. The
native full-HMM implementation is untouched.

### Validation

Compare final edge and mutation tables, identities and reconstructed called
alleles. Include empty edges, unordered blocks and adjacent same-parent segments.
Compare scientific contents after normal finalisation, not provenance or bytes.
Bulk appending accepts empty numerical arrays without a special branch.

## P4. Simplify greedy assembly and retain reference topology

**Relevant code:** `lib/matching.py`: `match_stitched_haplotype()`, `_match_gap()`,
and `_assemble_result()`.

### Keep focal runs together

`focal_runs()` already returns numerical arrays. The current loop converts each
run to three singleton arrays and a `CopyingRuns`, then concatenates them again.

1. Start with `blocks = [focal]`, including an empty focal result.
2. Iterate over focal left/right endpoints only. Preserve cursor updates,
   focal-site counts and gap calls in their existing left-to-right order.
   Remove singleton construction; append gap results as before.
3. Preserve inference-boundary splitting and terminal gaps. Assemble once using
   `_assemble_result()`; it copies focal views before exact scratch is reused.
   Gap matching does not overwrite that scratch.

The assembled edges need not be position-sorted: existing edge squash and table
sorting establish final order. Do not introduce a per-haplotype merge or sort.
All-focal and all-gap cases use the same flow, including the empty focal block.

### Remove the explicit topology reset

Delete `worker.matcher.reset_topology()` at the start of
`match_stitched_haplotype()`. The existing matcher in `tsinfer` already prepares
invalid topology, rebuilds for earlier positions and advances for later positions.
Each `find_path()` resets likelihood state independently. Retaining reference
topology therefore does not connect the probabilities of separate gaps or samples.

The explicit reset clears five node-sized arrays, then marks topology invalid,
causing another clear at the next gap. All-focal haplotypes currently pay for an
unneeded clear too. Keep `reuse_topology=True` and ordered gap calls. No dependency
changes or native reset optimisations are needed.

When implementing this stage, update the README's API description that currently
says the runner explicitly calls `reset_topology()`; keep it brief and factual.

### Validation

Compare final parent runs, mutations, gap call order/count and focal/gap site
counts. Include touching focal runs, internal gaps, terminal missingness,
all-missing calls, all-focal/all-gap cases and separate inference intervals.
Exercise consecutive haplotypes with first gaps earlier than, equal to and later
than the preceding final processed position, and an all-focal haplotype between
two haplotypes with gaps. Retained results must survive later worker calls.

## Smoke checks completed during plan review

Temporary scripts used the local environment via `uv run --no-sync python`.
Production functions were modified only in memory for comparison; no application
source, dependency source or test suite was edited.

- Canonical ragged path slices matched `read_paths()` for reordered sample
  identities, an empty path, touching edges and a physical gap.
- Two spawned processes wrote disjoint parts of a temporary shared map without
  explicit flushes; a subsequent process task read the expected complete values.
- Bulk edge append matched row-wise append after squash for empty arrays and
  unordered touching same-parent edges.
- On a synthetic reference with topology changes, 135 regional calls with random
  missing/observed calls produced identical paths and copied alleles with retained
  topology versus an explicit reset before every call.
- The combined focal-block and reset removal matched current greedy output for
  126 haplotype/minimum combinations, using three minima, split inference
  intervals, missing calls and random haplotypes. Edge arrays were compared after
  ordering; mutation sites and codes agreed directly.

These checks validate the key API and ownership assumptions. P1's complete worker
integration and P2's new scheduler have not been implemented or tested. The shared
map check ran on macOS; confirm the same pipeline handoffs on the deployment host.
A representative end-to-end run is still required to quantify speed and memory.

## Implementation validation and stopping point

Use existing performance collection where available and a few diagnostics elapsed
logs for index preparation, reference packing, sample decoding, analysis and output
writing. Log selected haplotypes, physical sample chunks, worker count and analysis
task count. Separate analysis from archive compression/packaging and benchmark
prerequisites; total target duration alone cannot identify the bottleneck.

Use small manual fixtures with `uv run`; do not add a test suite unless requested.
After equality checks, measure one representative run at the intended worker count
before launching the large grid. Preserve resident/end-to-end timing boundaries
and compare private process memory as well as elapsed time.

Leave exact extension, stable sorting, linear cover selection, ragged masks,
interval comparisons and archive packaging intact. Do not add new matching
algorithms, cross-run caches, native scratch/reset changes, a separate mask kernel
or unrelated compression tuning without measured evidence. Stop when the listed
repeated work is removed and its effect has been measured.
