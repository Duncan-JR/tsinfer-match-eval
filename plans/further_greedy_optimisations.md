# Further greedy orchestration improvements

Revised 23 October 2026. This replaces the previous plan in this file. It is a
plan for implementation; no performance improvement is claimed until measured.

## Scope and baseline

Make the existing pipeline practical to benchmark at population scale through
small, result-preserving changes to preparation, resource lifetime and benchmark
orchestration. Keep the greedy matching policy from `6b0d716`, including missing
sample handling and supported one-site-gap copying. Those behaviour changes are
already committed and are not implementation tasks here.

The local tsinfer baseline is `67710ca`, which fixes duplicate genotype loading.
Record the actual installed revision in each experiment. Likelihood tie-breaking
is outside this work. Retain the full ancestral reference for gap matching.

Evidence reviewed: the recent dphil-analysis parameter and scaling notebooks
(pt6–pt10), their gap-parent and root/dating diagnostics, the 8/15/22 October
meeting notes, and the current matching, loading, diagnostics and benchmark code.
The archived HPC suites disabled stage collection. Both already used 128 pending
haplotypes with 32 matching workers, so the default admission limit of W cannot
explain their observed idle capacity.

The new `tsinfer-anc-eval/config.yaml.anc_perf` describes 100 through 1,000,000
haplotypes over 28.1 Mb, with genotype errors and no phasing errors. Earlier
10-Mb simulations had a different error regime. Their coverage and timings are
context, not predictions for the new series. The large generated inputs are not
present in this local checkout.

Follow AGENTS.md throughout: reusable execution belongs in `lib`; exploratory
analysis belongs in dphil-analysis, editing the Jupytext `.py`. Use module imports,
dataclasses for returned records, pathlib, explicit intermediate operations and
Zarr v3. Keep numerical settings in configuration. Validate defined input
contracts at entry points; avoid compatibility layers, speculative fallback
paths and generic scheduling or caching frameworks. Keep README updates concise
and limited to behaviour, configuration and usage.

## Commit sequence

Each message below is the entire commit message: one imperative subject line,
without a prefix or body. Implement each coherent change with its necessary
configuration and documentation updates.

| Order | Commit message |
| --- | --- |
| 1 | Make greedy parameters dataset-specific |
| 2 | Bound memory during benchmark quality checks |
| 3 | Reuse benchmark outputs for diagnostics |
| 4 | Count derived alleles in physical sample chunks |
| 5 | Avoid redundant ancestor scans during reference packing |
| 6 | Release matching workers before finalising output |
| 7 | Separate greedy preparation and gap matching timings |

Commits 2, 4, 5 and 6 do not depend on one another. Commit 3 follows 1 and 2;
commit 7 measures the resulting execution boundaries. These are the proposed
pre-benchmark changes. Further scheduling work is conditional on their results.

## Local smoke-test protocol

Use a small simulation and a small real-data selection, initially the existing
100-individual, approximately 1-Mb fixtures. Use one and two matching workers,
all helper-thread settings at one, and the actual helper-inclusive reservation
from `utils.cpu_allocation`. Do not run the archived 32-worker configurations on
the MacBook. Keep benchmark trials sequential.

During implementation, put smoke configuration in a dedicated local YAML file
with a fresh experiment ID and output directory. Start with a 300-second trial
timeout, a 256-MiB quality-check matrix budget, and prefetch of one sample chunk.
Use the existing resident limits only for resident probes; they do not constrain
ordinary streamed matching. Inspect fixture sizes before running. A timeout or
memory-pressure failure means reduce the fixture, rather than enlarge the laptop
workload automatically. Store fixture sizes, seeds, chunk shapes, budgets and
repeat counts in the smoke configuration.

A second tiny Zarr v3 fixture should contain at least three physical sample
chunks and two variant chunks, noncontiguous selected samples, missing calls,
and both ancestral polarities. Include a partially filled final chunk. Use this
fixture for streaming checks: the existing one-chunk inputs cannot demonstrate
bounded behaviour across a whole cohort. Reuse the fixture across commits.

Suggested entry points, using the reservation resolved from the smoke config:

```sh
uv run snakemake performance --configfile config.yaml.greedy_smoke --cores <allocation> --dry-run
uv run snakemake performance --configfile config.yaml.greedy_smoke --cores <allocation>
uv run snakemake diagnostics --configfile config.yaml.greedy_smoke --cores <allocation> --dry-run
uv run snakemake diagnostics --configfile config.yaml.greedy_smoke --cores <allocation>
```

These are commands to use when implementing the plan; the smoke config is not
created by this planning change. For bounded helper checks use `uv run` as well.
The requested smoke tests may use a small pytest module: organise checks in
classes and use fixtures for setup. Do not create a general benchmark framework
or a broad regression suite. Fixtures and test helpers must not complicate
production interfaces. Run Ruff on edited Python files.

Freeze baseline outputs before changing orchestration. Compare canonical sample
identities, squashed parent runs, and mutation sites/states, allowing differences
in row ordering and provenance. Check reconstructed called alleles separately.
Every implementation change here should preserve matching results at the same
resolved dataset parameters. Smoke timings establish functionality, not HPC
speedups.

## 1. Make greedy parameters dataset-specific

**Commit message:** `Make greedy parameters dataset-specific`

Move `af_cutoff` and `min_region_size_kb` into each dataset entry. Require them
there and remove their top-level definitions. Other existing settings retain
their current scope and meaning, including HMM parameters, focal choice, worker
counts, helper limits, prefetch and pending-work limits. Do not add global
fallbacks or a nested override/merge system.

For example, alongside each dataset's existing paths and metadata:

```yaml
datasets:
  - name: small_simulation
    af_cutoff: [0.03, 0.05]
    min_region_size_kb: 25
    # Existing dataset fields follow.
  - name: large_simulation
    af_cutoff: 0.03
    min_region_size_kb: 100
    # Existing dataset fields follow.
```

Preserve the existing supported value forms: ordinary matching and diagnostics
accept a scalar or list for either parameter; performance accepts scalar/list
AF and requires a scalar minimum for each benchmarked dataset. Supporting a
benchmark grid of minima is separate work. Preserve the current mean-site-density
conversion of kb, computed once at the matching leaf, and record its resolved
site count.

Update validation, Snakefile target enumeration, diagnostic inventories and AF
loops, trial construction, effective configuration snapshots and README/example
configuration together. Enumerate each dataset's own Cartesian product. A union
of values may constrain wildcard syntax, but must not authorise another
dataset's parameter pair; reject explicitly requested unconfigured pairs before
matching. Keep parameterised output paths and metadata consistent.

Full HMM remains one trial per dataset/mode/worker/repeat, independent of that
dataset's AF list. Update maintained runnable configurations to the new schema;
do not rewrite archived experiment snapshots. Reject leftover top-level keys
with a direct migration error rather than silently ignoring them.

**Local smoke checks**

- Use two small datasets with disjoint AF values and different scalar minima.
  Check the dry-run targets, trial specs, output metadata and masks contain only
  their respective settings. With one mode/worker/repeat, two AFs for dataset A
  and one for B produce three greedy and two Full HMM trials.
- Exercise a list of minima through ordinary inventory generation without
  running a larger grid. Check benchmark validation rejects that list only for
  datasets selected for performance.
- Check missing dataset keys, obsolete top-level keys, duplicate values and an
  explicitly requested pair belonging only to the other dataset fail clearly.
- Compare one migrated dataset against its frozen baseline at identical values.

## 2. Bound memory during benchmark quality checks

**Commit message:** `Bound memory during benchmark quality checks`

Replace `quality_check`'s whole-cohort `np.stack` with bounded batches. Keep full
called-allele validation and sample/reference metadata checks. Introduce one
global `performance.quality_max_matrix_mib` budget for decoded validation calls;
this is not a total-process memory limit.

Reuse sample preparation, canonical remapping and mapped-slab loading from
`lib/haplotypes.py`. Group work by physical sample chunk and split its selected
columns into smaller batches when the configured budget requires it. Preserve
whole diplotypes and canonical job identities. Derive batch capacity once from
the reference site count and ploidy; report a clear error if one sample cannot
fit. Avoid using the native reader in a way that retains a full-genome physical
sample chunk beyond this budget.

Load the output tree sequence once. For each batch, decode its corresponding
output sample nodes, compare all called sites, accumulate error counts and
release its calls before loading the next batch. Check metadata once per output.
The remaining memory includes the reference/output trees, one decoded source
tile and decoder scratch; document this separately from the matrix budget.
Splitting a physical chunk may reread tiles, which is acceptable for an untimed,
explicitly bounded validation pass. Do not silently replace exhaustive checking
with sampling.

Run validation in the isolated trial process after its matching window closes,
so the existing launcher timeout covers it and the memory sampler sees it.
Return validation results to the suite for `quality.json`. Preserve matching
measurements when validation fails and mark the suite unsuccessful. Document
that whole-trial memory now includes validation; use the measured matching-window
peak for matching-memory comparisons. Keep validation outside matching wall/CPU
measurements and retain `quality_checks: false` as the existing opt-out.

**Local smoke checks**

- On the tiny multi-chunk fixture, compare the batched report with the previous
  dense calculation and require zero called-allele errors. Force multiple
  batches within a physical chunk and check the configured allocation bound.
- Check missing calls and non-reference ancestral alleles; preserve canonical
  identities with noncontiguous sample selection.
- Alter a called allele in a temporary output copy and require validation to
  fail while retaining the completed matching measurements.
- Verify quality results reach `quality.json`, validation lies outside the
  matching window, and a deliberately short configured timeout terminates a
  controlled validation delay through the existing launcher.

## 3. Reuse benchmark outputs for diagnostics

**Commit message:** `Reuse benchmark outputs for diagnostics`

Implement the intended benchmark-then-diagnostics workflow. The current benchmark
writes `data/performance/{experiment_id}/{run_id}/samples_raw.trees`; diagnostics
requires ordinary trees under `{data_dir}/matches/`. Completing a benchmark alone
therefore leaves dependencies that can schedule duplicate matching.

When performance is enabled, consume completed trial trees directly. Add a small,
explicit global selector `performance.diagnostics_run` with `mode`, `workers`
and `repeat`; for the smoke suite use `from_reference`, 2 and 0. These select one
trial per required method/dataset/AF combination. Validate the selection against
the configured suite. Do not select the fastest trial or infer identities from
filenames. Use `runs.csv` and the saved trial specification to resolve paths.

Require the selected suite to cover the diagnostic datasets and their parameter
settings, including both Full HMM and greedy. Explain incomplete coverage as a
configuration error; do not fill missing combinations by running ordinary
matching. Pure performance runs may still use a subset or a single method;
apply diagnostic coverage requirements when diagnostics is requested.

In the Snakefile, benchmark-backed diagnostics must depend on suite completion
and prepared inputs, without dependencies on ordinary matching outputs. Resolve
selected trial paths after completion, since `runs.csv` does not exist when a
fresh DAG is first parsed. A runtime resolver inside the diagnostics driver is
sufficient; no checkpoint framework or copied canonical outputs are needed.
Verify the selected files exist before analysis. For `performance: null`, retain
the existing ordinary-output dependencies.

Use the resolved paths consistently for path preparation, comparisons and
archiving. Remove the archive builder's separate hard-coded ordinary Full HMM
path. Record selected run IDs and parameters in the diagnostic manifest, and
reuse the existing archive deduplication when a trial tree serves both roles.
Keep ordinary `all` target enumeration independent from the source selection for
diagnostics. Diagnostics reconstructs greedy masks and reads saved copying paths;
it does not invoke matching HMMs.

**Local smoke checks**

- Complete a tiny suite in an isolated output directory with no ordinary match
  trees. Dry-run diagnostics: it must schedule neither `match_samples` nor
  `match_stitched_samples`. Run diagnostics and check masks, tables and archive.
- Check the manifest's selected run IDs and archived trees correspond to the
  configured selector. Compare masks/statistics with diagnostics of the same
  trees through the ordinary-output route in a separate temporary directory.
- Test an unavailable selector, absent AF trial, missing required method and
  deleted selected tree. Each should fail clearly without scheduling replacement
  matches. Check a repeated completed diagnostics invocation does no work.
- With `performance: null`, verify the original ordinary workflow remains valid.

## 4. Count derived alleles in physical sample chunks

**Commit message:** `Count derived alleles in physical sample chunks`

In `utils.annotate_derived_counts`, retain the existing outer variant-block loop
and accumulate counts across physical sample chunks. Select only the requested
columns within each decoded tile. Compute ancestral polarity once per variant
block; accumulate integer derived and called counts, then divide once after all
sample chunks. Preserve missing-call exclusion and zero-called-site behaviour.

Keep the existing copied-store output and annotation fields. This change does
not introduce storage overlays, shared mutable inputs or a second loader pool.
Memory should scale with one physical tile and its selected-call/mask buffers,
rather than a variant block times the entire cohort.

**Local smoke checks**

- Compare derived counts and AFs with a direct dense calculation on the tiny
  fixture, covering both polarities, missing calls, an entirely missing site,
  noncontiguous selection and a partial final chunk.
- Verify source calls are unchanged and unrelated output arrays retain their
  values. Count physical tile accesses to confirm each required tile is read
  once and unselected sample chunks are skipped.
- Prepare and match one small dataset using the resulting annotations; compare
  its eligible catalog and canonical output with the baseline.

## 5. Avoid redundant ancestor scans during reference packing

**Commit message:** `Avoid redundant ancestor scans during reference packing`

In `haplotypes.pack_reference_chunk`, use the sorted `catalog.columns` to find
its physical column-chunk slice with two `searchsorted` calls. Avoid recomputing
`catalog.columns // width == chunk` over the full catalog for every packing job.
Selection then costs binary searches plus the selected slice, instead of a
whole-catalog scan for each job.

For each required variant block, form a numeric overlap mask over that job's
ancestor supports and iterate only overlapping ancestors in Python. Retain
original catalog order, clipped supports, packed offsets, seed checks, disjoint
writes and one decode per required physical tile. Keep the existing pool and
packing barrier. No persistent artifact, interval index or scheduler is needed.

A local all-seeded-catalog audit of the 300-individual, 30-Mb TGP panel counted
587,051 ancestor/block checks and 136,837 actual overlaps. This is evidence of
redundant work, not a timing result or an estimate for the new large inputs.

**Local smoke checks**

- Compare every packed byte and catalog array against the frozen implementation
  on a small panel with multiple column and variant chunks. Include sparse AF
  selection, spans ending at chunk boundaries and a partially filled final tile.
- Exercise empty eligible selection and an invalid derived seed. Preserve their
  existing outcomes and the physical block-read count.
- Run packing with one and two workers and compare output bytes. Match a small
  dataset and require identical canonical paths and mutations.
- Use metadata-only operation counts to confirm selection no longer scans the
  entire catalog per job and Python copying visits only actual overlaps.

## 6. Release matching workers before finalising output

**Commit message:** `Release matching workers before finalising output`

End the worker/loader lifetime after `_coordinate` has appended every result.
Perform the existing orderly shutdown and joins before edge squashing, native
finalisation and output dumping. Reorganise the existing try/finally boundary
so cleanup has one owner; avoid a second shutdown path or duplicate sentinels.
Stop the watcher appropriately so expected exits are not reported as failures.

Release the coordinator's original reference after sample metadata, output
tables, catalog and size counters have been prepared. Workers load their own
reference from its path. Release other clearly dead preparation objects where
ownership permits, without adding manual garbage-collection machinery.

Keep cleanup on matching failures and finalisation failures correct. The full
`from_reference` timing still includes worker shutdown and output finalisation.
Preserve the existing resident matching-window definition. This change removes
unneeded worker indexes/scratch from the finalisation memory peak; it does not
promise faster matching or solve the memory required by final output tables.

**Local smoke checks**

- Match the same small input at one and two workers and compare canonical output
  with the baseline.
- At finaliser entry, assert the recorded matching processes have exited and the
  loader/watcher have joined. Confirm temporary mappings are cleaned up.
- Trigger a worker-side failure with an invalid-seed fixture and a separate
  finalisation failure with a test monkeypatch. Require prompt propagation and
  no surviving workers or temporary slabs; add no production failure-injection
  switches. Use importable helpers for checks involving spawned processes.
- Inspect matching-window and whole-trial memory/timing records. Use lifetime
  assertions as the correctness check; a tiny fixture need not show a reliable
  memory or speed improvement.

## 7. Separate greedy preparation and gap matching timings

**Commit message:** `Separate greedy preparation and gap matching timings`

Extend the existing `performance.Collector` only where measurements are missing.
Retain existing workload counters and stage aggregation. Add coordinator elapsed
windows for process startup, packing completion, worker readiness, matching/drain
and shutdown. Record first-slab and first-result latency relative to matching
admission so startup loading can be distinguished from subsequent work.

Inside `gap_matching_assembly`, separate native `find_path` from path conversion,
mutation extraction and final concatenation. Measure coordinator result waiting
separately from append work. Keep worker timers scoped to their executing thread;
coordinator elapsed windows and summed worker service times have different
meanings. Label inclusive totals and their children and never add both together.
Avoid per-gap event streams, queue-size polling or a new tracing service.

Preserve `collect_stages: false` as a cheap path retaining workload counts. Record
one staged run at every new scale, including the million-haplotype run. Record
validation separately from matching after commit 2. Use the corrected tsinfer
baseline and whole `from_reference` windows for method-level speedup claims.

**Local smoke checks**

- Run the same small fixture with stages enabled and disabled. Require identical
  canonical output and workload counters, populated diagnostic stages when on,
  and no regional timing records when off.
- Check native call counts match actual `gap_calls`; one-site shortcuts must not
  count as native calls. Exercise all-focal and all-gap examples.
- Use the multi-chunk fixture to inspect first-slab/result latency and draining.
  Require finite nonnegative durations and sensible ordering; do not assert
  that overlapping worker services sum to wall time.
- Compare one- and two-worker records to verify service aggregation and elapsed
  windows remain interpretable. Run a few short alternating on/off trials only
  to detect gross instrumentation overhead, not to claim statistical speedups.

## Measurements and stopping point before the large benchmark

After these commits, use the 10,000- and 100,000-haplotype steps on the compute
node to choose feasible workers, prefetch and admission limits. Size the packed
reference, largest live sample slabs, private worker state and output tables;
USS alone does not bound shared mapped working sets. Use node-local temporary
storage with sufficient capacity. Keep the million-haplotype comparison narrow.

Only tune existing settings when the staged run identifies a reason. The archived
HPC admission bound was already 4W. Compare a small number of alternatives on one
representative workload, holding the other settings fixed, and judge whole-call
elapsed time and memory. More pending work is not automatically an improvement.
Configuration changes require no scheduling framework or mandatory extra commit.

If first-slab latency is material, a later bounded change may start the existing
single loader after packing, during worker index construction. It must handle
early slab events in the same bounded coordinator state, preserve the readiness
gate and shutdown invariants, and fit the simultaneous helper allocation. Leave
this conditional until measurements show useful overlap; it is not part of the
seven required commits.

Defer persistent worker sessions, prepared-reference persistence, IPC batching,
process-shared native indexes, native reset/traceback redesign and reduced gap
references. Keep strict physical-span filtering, general mismatch tolerance,
boundary-conditioned HMMs and likelihood tie-breaking outside this plan. No
further matching-policy change is needed to execute this benchmark programme.
