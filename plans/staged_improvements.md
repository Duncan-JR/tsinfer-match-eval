# Staged matching improvements

## Objective and scope

Produce a clear, committed baseline for profiling large cohorts, principally
more than 10,000 individuals. Keep changes that reduce work proportional to
reference size, gap count or cohort size, plus small improvements that make the
implementation simpler. Prefer pipeline changes in tsinfer-match-eval. Retain
native batching, regional resets and direct seeking with Full HMM isolation.

This plan is self-contained. Execute it only when implementation is requested.
Its nine commits must account for every current tracked change in
tsinfer-match-eval and TSLS, plus the two TSInfer packaging overrides listed
below. Dphil-analysis, notebooks, scientific results and other repositories are
outside scope. Preserve untracked personal files and data. Do not push changes.

Use existing correctness checks and tiny known-answer comparisons. Do not run
performance suites, full-cohort matching, upstream inference or regenerate
scientific diagnostics. Larger-scale profiling is a separate task.

## Starting state

All repositories are on BMRC beneath `/well/kelleher/users/uuc395/`.

| Repository | Branch | Starting HEAD |
| --- | --- | --- |
| tsinfer-match-eval | `main` | `fb2ba17625e46f75adb2827522ec68578c42caec` |
| tsls | `conditioning` | `e4cf2333b99eb6b104ff44cbb34625f22a16e880` |
| tsinfer | `tsls` | `63837fd44b661714c774c3c5957904eea25d7210` |

Recheck instructions, branch tips, index and working diffs before executing.
Use TSLS's starting HEAD as the reference for normal scalar/matrix behaviour.
TSInfer already contains the required sample-size support. Its only outstanding
tracked changes are machine-specific paths in `pyproject.toml` and `uv.lock`;
discard those overrides in step 2. No TSInfer source or history edit is needed.

The working diffs contain the changes to organise. Baseline dependency files
for removing unrelated package upgrades are available under
`/well/kelleher/users/uuc395/tmp/greedy_optimization_20261114/before_promotion/dependencies/`.
Preserve unstaged work while constructing the commit sequence; stage by
responsibility and keep intermediate commits buildable.

## Required matching flow

The orchestration must expose these operations in order:

1. Activate eligible ancestors, extend exact matches and compute the greedy
   cover. Apply minimum size to whole touching focal unions.
2. Remove whole focal unions touching zero-rate transitions.
3. Partition copied spans and gaps within the called domain and inference
   boundaries.
4. Resolve supported one-site gaps in a separate pass, recording mismatch
   corrections and leaving zero-rate sites as HMM gaps.
5. Merge compatible copied spans and construct conditioned residual windows.
6. Match those windows in one native batch per haplotype, clip to the true gaps,
   assemble copying paths/mutations and finalise output.

Step 2 and the zero-rate guard in step 4 enter in commit 9. The core greedy
cover handles exact intervals and minimum size only. Map filtering, one-site
policy, conditioning windows and output assembly are separate operations in
the existing `lib/greedy.py` and `lib/matching.py` modules.

Retain compact arrays and compiled linear passes. Use named intermediate results
and small dataclasses at meaningful orchestration boundaries. Keep worker-owned
scratch and immutable reference inputs; do not add a general pass framework or
an object for each copying interval.

## TSLS implementation requirements

### Regional selection and ownership

`find_paths` selects regional optimisation explicitly. Scalar `find_path`,
including partial and conditioned scalar calls, and `viterbi_matrix`,
`forward_matrix` and `backward_matrix` retain the normal strategy. A batch row
covering the whole sequence is still valid. Do not select the strategy by window
coordinates or endpoint pins.

Add one narrow regional C entry alongside `tsls_hmm_find_path`. Both entries
share checked execution and cleanup. Keep Python signatures unchanged apart
from the additive `find_paths` API; no constructor flag or public mode setter.

Put the live-node list, inverse positions, visited-node list, visited marks and
counts in one optional workspace owned by the HMM. Allocate on first nonempty
regional use, reuse across calls and free with the HMM using existing allocator
conventions. Normal-only matching allocates none of these per-node arrays.

Use the workspace pointer for allocation state and the existing
`topology_valid` for validity. A single last-strategy marker is sufficient to
invalidate retained topology on a normal/regional change. Do not introduce
parallel readiness, recovery or per-array validity flags. Reuse the existing
cold path for strategy changes, backwards positioning and errors; an error
invalidates retained topology and propagates its original error code. Keep
normal-to-normal and ordered regional topology reuse. An empty batch naturally
runs no windows, allocates no workspace and leaves retained state alone.

### Shared traversal without new Full HMM hot-loop checks

Keep one maintained sweep body, one actual traceback body and shared numerical
site-processing code. Supply a private constant Boolean policy from small normal
and regional entry wrappers. Use TSLS's existing `ALWAYS_INLINE` convention
only along the driver/topology-helper chain needed to propagate that constant.
Normal calls supply `false`; regional calls supply `true`. This must eliminate
regional tests and work from normal edge/node/site loops.

Choose normal or regional reset/seeding/traceback-preparation helpers once per
window, outside node loops. Keep these housekeeping helpers short. Do not
hand-copy the sweep or traceback, add per-edge callbacks, or load a mutable
regional flag/workspace pointer as the mode test on every operation. Keep
existing numerical functions and their validation style.

### Changes required in the live patch

| Live code | Required implementation |
| --- | --- |
| `lib/tsls.h` tracking fields and `track_active` | Replace with the optional workspace and one coarse strategy marker. Add only the regional C entry declaration and short ownership comments. |
| `tsls_hmm_alloc/free` | Move the four tracking-array allocations to regional use. Keep ordinary allocation and existing cleanup conventions. |
| `tsls_hmm_run` | Replace bounds/pin-based activation with constant entry selection. Keep costs, recurrence, pruning, sparse processing and statistics unchanged. |
| `tsls_hmm_remove_edge/insert_edge` | Replace per-edge `self->track_active` checks with constant-specialised shared helpers; regional updates occur after the same topology mutations. |
| New cache clearing under `!update_likelihoods` in remove-edge | Restrict it to regional topology-only updates, including traceback. It must add no cache work to normal traceback. |
| `tsls_hmm_track_reset/track_node` | Keep visited deduplication and dense live membership with swap removal; these are algorithmic operations. Put the existing remove-edge comment back above remove-edge instead of leaving it attached to track-reset. |
| `tsls_hmm_position_tree` | Preserve normal positioning and cold setup. Restrict extra likelihood/cache/traceback/inverse-position clearing to regional cold setup. Direct regional seek enters in commit 5. |
| `tsls_hmm_init_sweep` | Keep the original ascending normal scan and root seeding. Use a separate regional visited reset/live enumeration, with no per-node strategy ternary and no normal root sort. |
| `tsls_hmm_seed_initial_state` | Keep scalar conditioning independent of workspace allocation. Regional seeding uses listed likelihoods and live children; preserve ascending root/child insertion order using existing scratch and a simple sort. |
| `tsls_hmm_run_traceback` | Keep normal all-node flag preparation; regional preparation touches listed nodes after a complete visited reset. Share the subsequent traceback decisions and specialise topology bookkeeping only. |
| `docs/likelihood_model.md` | Restore the starting version; this implementation does not change the mathematical model. |

### Checks and MVP boundaries

Keep necessary external-input and memory-safety checks; establish internal
invariants through simple control flow and correctness checks.

- In the batch binding, retain object-state, array shape/dtype/length and writable
  contiguous-output checks, safe window bounds, allocation failures and overflow
  protection when summing output capacity. The Python wrapper delegates these
  checks to the binding.
- Remove endpoint-range checks from the new batch preflight loop: the shared
  native path entry already validates endpoints before use. Keep its native
  validation and conditioned-path errors. Keep batch site bounds for safe
  allocation even though the engine also validates them.
- Do not add another full-haplotype scan at batch entry, window-order checks or
  automatic sorting. Existing per-window allele validation and cold positioning
  handle the supported inputs. Overlapping and backwards windows remain valid.
- Keep existing allele, endpoint-in-tree, impossible-conditioning and
  zero-recombination errors, tree-position error propagation and established
  internal assertions. `setup.py` enables ordinary C assertions by undefining
  `NDEBUG`; retain that build policy and existing `TSLS_EXTENDED_CHECKS`.
- Add no per-edge/node assertions or validation of workspace existence, IDs,
  list capacity, inverse-position agreement or uniqueness. Those follow from
  validated topology and the list update rules. No new permanent invariant
  checker or recovery mechanism is required for this MVP.
- Keep membership/deduplication branches and seek-edge coverage predicates:
  they select correct algorithmic updates. Keep root/child ordering needed for
  deterministic path ties. Do not replace these with defensive whole-panel scans.
- Size output once from window lengths. The existing bound of at most one path
  edge per site makes repeated capacity checks unnecessary. Propagate a failed
  row without rolling back earlier output; document partial writes briefly.
- Follow neighbouring C types, names, indentation, `WARN_UNUSED`,
  `ret`/`goto out` and allocator conventions. Use short helpers with one clear
  responsibility and comments only for non-obvious invariants. Keep temporary
  implementation diagnostics out of committed code.

### Minimal TSLS documentation

Keep the `find_paths` docstring to its array/return contract, independent
likelihood resets, explicit regional selection and output/error behaviour.
The README needs only a short API pointer consistent with its existing style.
Explain live versus visited state and seek coverage in short comments beside
the implementation. Leave the likelihood-model document unchanged; add no
optimisation narrative, benchmarking guidance or build/inspection instructions
to TSLS documentation.

## Planned commits

Use these capitalised titles without category prefixes or co-author trailers.
Split shared files by responsibility; validate the code being committed rather
than unrelated unstaged changes.

### 1. tsinfer-match-eval — `Move personal configurations into ignored storage`

- Add `configs/` to `.gitignore`; commit deletion of the tracked personal
  `config.yaml.greedy_smoke`, `config_tsls_conditioned.yaml` and
  `config_tsls_unconditioned.yaml`. Preserve their ignored personal copies and
  the tracked `config.yaml.example`.
- Replace obsolete root-config commands with generic
  `--configfile configs/my_run.yaml` examples.
- Make `tests/conftest.py` use the tracked example plus explicit tiny fixture
  overrides, without depending on ignored configuration.
- Limit pytest discovery to `tests/` in `pyproject.toml`.
- Acceptance: active commands and existing fixtures work without personal files.

### 2. tsinfer-match-eval — `Configure sample mismatch calibration and haploid panel size`

- Keep reusable `sample_hmm_settings` in `lib/utils.py`: calibrate epsilon over
  the explicit span and selected source sites before singleton exclusion, and
  fix the selected haploid cohort size.
- Keep ancestor matching on stock mismatch/recombination settings with that
  cohort size; do not pass sample epsilon, map or numerical overrides to it.
- Retain exclusive mismatch/epsilon selection, positive finite calibration span,
  nonempty calibration sites and `0 < resolved_mismatch < 1`. A bounded check
  such as `0 < epsilon < 1` already excludes NaN/infinity; remove its redundant
  separate finiteness check. Validate raw settings once, then check only the
  derived probability after calibration.
- Keep resolved mismatch/map/sample-size manifest checks in `lib/benchmark.py`.
  Keep `lib/profile_regions.py` scoring with the configured haploid size and
  neutral missing-observation emissions. Update existing tiny fixture inputs.
- Pin TSInfer `63837fd44b661714c774c3c5957904eea25d7210`. Restore unrelated
  dependency versions using the baseline files above, changing only required
  constraints/source identities. Keep an intermediate compatible TSLS pin;
  commit 7 pins the final native revision.
- Discard TSInfer's two packaging-path overrides. The pipeline supplies its
  explicit native dependency override; no TSInfer feature commit is needed.
- Document only generic parameter contracts in the pipeline. Calibration sweeps
  and other disposable analysis code are outside this repository and plan.
- Acceptance: existing configuration/metadata checks use the resolved settings;
  dependency resolution contains committed native sources and no unrelated
  package refresh.

### 3. tsls — `Batch independent regional matches`

- Stage the batch API against the starting engine. Preserve engine work for
  commits 4/5 while constructing this clean intermediate version. Restore the
  uncommitted `docs/likelihood_model.md` addition to its starting contents.
- Commit `Hmm_find_paths` in `_tslsmodule.c`, the thin wrapper/docstring in
  `tsls/matching.py` and the short README API pointer.
- Release the GIL once around a loop over `tsls_hmm_find_path`. Use shared
  output arrays and per-window offsets; each row resets likelihoods independently.
  Apply the boundary-check simplifications above.
- Support zero rows with the ordinary empty arrays/initial offset. Preserve
  output outside matched windows; specify overlapping windows and partial
  writes on error without adding a rollback layer.
- Leave the scalar binding/method, engine and header unchanged at this commit.
- Acceptance: existing affected native checks and tiny batch-versus-scalar
  comparisons pass for pins, overlaps, empty input and output preservation.
  Capture any additional tiny scalar reference outputs needed for steps 4/5.

### 4. tsls — `Reset regional matches using live and visited nodes`

- Implement the workspace, regional C entry, constant entry selection and shared
  traversal specified above in `lib/hmm.c`/`lib/tsls.h`. Switch only the batch
  binding to the regional entry.
- Use the existing incremental positioning algorithm at this stage. Track
  topology changes during the sweep, traceback and between-window positioning.
- Before each regional restart, clear the visited union, including nodes made
  inactive by traceback; seed the live tree with fresh likelihoods. Preserve
  deterministic root/child insertion order. Restrict new cache and cold-state
  writes to regional operation.
- Preserve normal allocation, reset, scalar conditioning and traceback
  preparation. Reuse existing error/cold handling with the single strategy
  marker; do not add a second state-repair system.
- Update only the short regional-selection sentence in the batch docstring if
  needed; keep invariants in local code comments.
- Acceptance: tiny reference comparisons cover paths/alleles, pins, normal calls
  before/after batches, failed calls, and forward/backward methods. Run affected
  existing checks. Compiled-code inspection is reserved for the final native
  validation after commit 5.

### 5. tsls — `Seek directly between independent regional windows`

- Add indexed forward positioning to the regional helper only. Normal scalar
  and matrix calls retain their starting positioning algorithm.
- Reuse `tsk_tree_position_seek_forward`, breakpoint lookup and existing edge
  indexing/coverage helpers. Remove outgoing edges present in the old tree and
  insert incoming edges covering the target, preserving edge order and regional
  live/visited accounting. Skipped spans have no observations/likelihood updates.
- Keep same-tree reuse and existing cold/backwards handling straightforward.
  Propagate seek errors; no silent recovery traversal or extra per-edge checks.
- Use shared cold setup where clear and one short regional warm-seek helper.
  Limit documentation to the necessary nearby coverage comment.
- Acceptance: tiny comparisons cover same-tree and multi-tree jumps, overlaps,
  backwards windows and errors, preserving paths/ties/alleles. Run the relevant
  existing native suite and the one-time inspection below; record the final
  native commit ID for step 7.

### 6. tsinfer-match-eval — `Compile reference packing and reuse open inputs`

- Keep `_pack_tile` as a short GIL-free kernel. Share opened genotypes and the
  writable mapped destination; the coordinator owns opening/closing. Make
  `packed` an explicit required argument to the packing helper, removing its
  optional-opening branch and `owns_mapping` cleanup. Update serial diagnostic,
  profiler and existing test callers to open once and close at their boundary.
- Keep disjoint packing writes, bounded decoder/I/O controls and existing CPU
  accounting. Publish the mapping read-only before matching.
- Keep direct vectorised seed lookup, ancestral-label reuse in `SampleInput`
  and the contiguous slab slice. Use selected `columns` and their `seeds`
  directly; remove the full-panel `-1` seed array and second `flatnonzero`
  used only to select those same columns again.
- Retain external catalog axis/identity/support checks and the two-stage seed
  bounds-then-equality check. Validate derived packed seeds at packing completion,
  not per copied byte. Workers consume the prepared read-only catalog without
  repeating its preparation/validation.
- Remove sample/reference preparation overlap from `lib/matching.py`: delete
  early loader startup, `deferred_slabs` and `_preparation_event`. Complete
  reference/catalog preparation, packing and worker readiness before starting
  the existing bounded sample loader; keep loading while matching.
- Preserve shutdown/failure cleanup and resident timing boundaries. Update
  matching/diagnostic/profiling callers and existing packing fixtures together.
  Remove `TrackedOffsets` instrumentation; retain packed-data/tile-read checks.
- Acceptance: existing packing, sample reconstruction and lifecycle checks pass;
  preparation and matching are distinct, bounded phases.

### 7. tsinfer-match-eval — `Prepare copying paths in explicit passes`

- Replace `_prepare_intervals` with the required explicit partition, one-site,
  coalescing and window-construction passes. Keep compiled numeric assembly,
  compact arrays and a short exact-cover scan with whole-union minimum filtering.
- Use the partition contract instead of rechecking neighbour adjacency when
  building windows. The support-prefix search already supplies the fallback
  ancestor's start bound; omit that duplicate check but keep its stop and known
  reference-allele checks. Do not repeat sortedness, ID, capacity or coverage
  validation between trusted passes.
- Resolve one-site gaps after minimum filtering, retaining its effect on focal
  unions. Choose the original supported left neighbour, original supported
  right neighbour, then activated ancestor with furthest support end in stable
  support-start order. Do not let earlier bridge decisions change later
  neighbours. Respect inference boundaries and known reference calls. If the
  prefix-selected fallback is unsupported, leave the gap to the HMM rather than
  searching other ancestors. Record a correction for a called disagreement,
  none for a missing observation.
- Merge compatible copied spans before constructing residual windows.
  `condition_gaps: false` removes pins/padding, not the one-site policy.
- Submit one native batch for nonempty windows. Keep logical-gap and
  native-batch counters distinct; retain useful timers without temporary
  instrumentation. Clip native output and assemble paths/mutations once.
- Make diagnostics and profiling call these same passes. Update their batch
  recorder and existing adapters together. Keep named array results without
  a ten-value positional bundle or per-interval objects.
- Pin the final TSLS commit from step 5 in `pyproject.toml`/`uv.lock` using the
  pipeline's local-Git source convention and consistent transitive override.
  Replace `path = "../tsls"`; preserve unrelated package versions.
- Acceptance: existing conditional/unconditional, bridge-mismatch, boundary and
  assembly checks pass. The call sequence is readable. Zero-rate filtering and
  its one-site guard enter in commit 9.

### 8. tsinfer-match-eval — `Document the matching workflow`

- Keep the pipeline README to concise installation, generic configuration,
  matching, diagnostics, profiling and recovery guidance. Describe maintained
  configuration options and timer/resource scopes where useful.
- Remove experiment reports, named cohort/run recipes, numerical findings and
  absolute experiment paths throughout the README. Use neutral examples such
  as `configs/my_run.yaml`; preserve saved experiment files in place.
- Keep the small `lib/utils.py` Git dirty-state provenance improvement.
- Include this plan and its execution record. TSLS documentation remains limited
  to the API material specified above.
- Acceptance: examples use maintained commands/options; no experiment report is
  presented as usage documentation.

### 9. tsinfer-match-eval — `Exclude focal regions touching zero recombination blocks`

This is the final feature commit.

- Compute a read-only incident-site prefix once from the actual HMM recombination
  probabilities. Exclude the artificial first entry; mark both sites adjacent to
  every real zero-rate transition, including zeros produced by map conversion.
- After exact cover and minimum filtering, explicitly compact away each whole
  touching focal union containing a marked site, including a union extending
  into a zero-rate block. Use the unmodified probabilities and exact zero.
- Remove this policy from `greedy.cover` and remove the map prefix from
  `ExactState`/`make_state`. Keep it with the prepared matching model/worker
  and pass it explicitly to the filter and one-site pass.
- Check marked-site exclusion only for actual one-site bridge candidates. Leave
  marked sites as HMM gaps: inference boundaries/called domains can split a
  zero-rate block into one-site pieces, so whole-union filtering alone is
  insufficient.
- Construct residual windows afterwards, with conditioning anchors outside
  zero-rate blocks. Apply the same filter for conditioned and unconditioned
  greedy matching. Matching, diagnostics and profiling use the same prefix and
  passes.
- Update the pipeline's generic matching explanation and existing callers.
- Acceptance: tiny known answers cover the initial sentinel, a union extending
  into a plateau, an entirely zero map, terminal blocks and inference-boundary
  one-site pieces. Confirm matching/diagnostic mask agreement and run the
  existing pipeline suite after final integration.

## Complete tracked-file disposition

Numbers refer to the commits above. Shared files are split by responsibility;
all current tracked changes must be retained as specified or discarded.

| Repository | File(s) | Disposition |
| --- | --- | --- |
| tsinfer-match-eval | `.gitignore`; deleted `config.yaml.greedy_smoke`, `config_tsls_conditioned.yaml`, `config_tsls_unconditioned.yaml` | 1: ignore personal configs and commit deletions. |
| tsinfer-match-eval | `README.md` | Generic usage updates in 1/2/6/7/9; complete experiment cleanup in 8. |
| tsinfer-match-eval | `lib/benchmark.py` | 2: resolved scientific settings/manifest checks. |
| tsinfer-match-eval | `lib/diagnostics.py` | 6: packing ownership; 7: common preparation/batching; 9: zero-rate policy. |
| tsinfer-match-eval | `lib/greedy.py` | 7: simple cover and explicit passes; 9: explicit map filter/bridge guard, no map state in the exact-cover state. |
| tsinfer-match-eval | `lib/haplotypes.py` | 6: compiled packing, seed/label/slice simplifications and explicit ownership. |
| tsinfer-match-eval | `lib/matching.py` | 6: packing ownership and removal of preparation overlap; 7: passes/batching/counters; 9: map-policy orchestration. |
| tsinfer-match-eval | `lib/profile_regions.py` | 2: generic scores; 6: packing ownership; 7/9: common preparation/batch contracts. |
| tsinfer-match-eval | `lib/utils.py` | 2: scientific settings/concise validation; 8: dirty-state provenance. |
| tsinfer-match-eval | `pyproject.toml`, `uv.lock` | 1: pytest discovery; 2: TSInfer pin/dependency cleanup; 7: final TSLS pin. |
| tsinfer-match-eval | `tests/conftest.py` | 1: tracked config fixture; 2/6: necessary tiny input links/fixtures. |
| tsinfer-match-eval | `tests/smoke_helpers.py`, `tests/test_greedy_smoke.py` | 6/7/9: adapt existing packing, batch, stage and policy checks; remove obsolete implementation instrumentation. |
| tsinfer-match-eval | New `plans/staged_improvements.md` | 8: include plan; 9: final execution record. |
| tsls | `_tslsmodule.c` | 3: thin batch API; 4: batch selects regional C entry. Preserve scalar binding. |
| tsls | `tsls/matching.py` | 3/4: thin batch wrapper and concise regional API contract. Preserve scalar method/constructor. |
| tsls | `README.md` | 3: short API pointer only. |
| tsls | `lib/tsls.h` | 4: optional workspace, single strategy marker and regional entry declaration. |
| tsls | `lib/hmm.c` | 4: isolated reset/shared traversal; 5: regional seek. Preserve normal algorithms and existing numerical behaviour. |
| tsls | `docs/likelihood_model.md` | Discard the uncommitted addition; final file equals starting HEAD. |
| tsinfer | `pyproject.toml`, `uv.lock` | 2: discard machine-specific path overrides; no TSInfer feature commit. |

Keep `Snakefile`, repository instructions and the generic example unchanged
unless a retained general option requires a small example correction. Preserve
untracked checkpoints, editor swap files, sample lists, logs and scratch files.
Add no exploratory modules, experiment reports or new test suite.

## Execution validation

Builds, Python and checks run on cloudcomp113 after `loadpy`, using `uv run`;
use direct execution, not Slurm. Preserve scientific outputs and notebook
environments. Adapt existing tests to changed APIs; use bounded ad hoc known
answers if needed rather than adding a permanent harness. Run affected checks
as implementation changes, and the existing native/pipeline suites after their
final integration; do not repeat broad checks without a new change or failure.

After commit 5, inspect the final optimised object/disassembly once with the
normal project build flags on cloudcomp113. Confirm normal edge/node/site loops
contain no new regional flag/pointer tests, tracking calls or regional cache
stores. Refine the small specialisation boundaries if needed. Repeat only if
subsequent native changes during this execution invalidate the result, then
record the result here. This inspection validates this plan's implementation
only: do not add a build hook, CI check, permanent inspection script or future
per-build requirement.

Also confirm:

- Normal scalar/matrix numerical behaviour, errors and tiny copying paths/ties
  agree with the starting engine, and regional batch rows agree with independent
  scalar calls. Check normal-only allocation and mixed normal/regional API use.
- Normal reset, seeding, flag preparation and positioning retain their algorithms.
  Regional state is absent from normal-only per-node allocation and hot loops.
  Fixed call-boundary and compiler/layout effects can remain; do not claim
  identical elapsed time from code inspection.
- Every active preparation/packing consumer is updated. Matching, diagnostics and
  profiling share policy; memory, I/O and failure cleanup remain bounded.
- The TSLS patch matches existing style, with no duplicated numerical engine,
  extra runtime invariant machinery or broad unrelated refactor. Its documentation
  changes stay within the API scope above.
- Relevant format/lint/whitespace checks pass. Both READMEs remain generic.
- Every tracked working change is resolved, TSInfer is clean at `63837fd4`,
  retained dependencies are pinned and personal files/results are preserved.

Record commit hashes, validation results and genuine blockers below as execution
proceeds. Include the final record update in commit 9 so no tracked plan change
remains after completion.

## Execution record

- Plan review: 17 November 2026. Commit numbering, tracked-file coverage,
  documentation scope and whitespace checked. Source contents, indexes, HEADs
  and tracked status match the pre-review fingerprints across all three repos;
  the dphil-analysis tracked diff is unchanged. Only this plan was edited.

- Execution: 17 November 2026, entirely on BMRC. Builds, Python and checks ran
  on cloudcomp113 after loadpy, through uv. No scientific runs, performance
  suites, notebooks or upstream inference were executed; nothing was pushed.
  Initial tracked diffs and source files are preserved in
  /well/kelleher/users/uuc395/tmp/staged_improvements_20261117/.
- Commit 1 (tsinfer-match-eval): 3ad31b234ffd084c8af36e08dae07cead1616801 — Move personal configurations into ignored storage.
- Commit 2 (tsinfer-match-eval): cc145e94d8ad8e294d8f062329879d0aaa5c58ae — Configure sample mismatch calibration and haploid panel size.
- Commit 3 (tsls): a1fa3d4b813986a2cb80347bf989a0eee9cbe0e2 — Batch independent regional matches.
- Commit 4 (tsls): 20b8da42709ef189238b7cae9840b8a3a272ece2 — Reset regional matches using live and visited nodes.
- Commit 5 (tsls): 925be379cb9e94d8e7de6e2f0cc5926da292f170 — Seek directly between independent regional windows.
- Commit 6 (tsinfer-match-eval): f0a13f79b5da3af6dea6f86004c32eefefe2a7b0 — Compile reference packing and reuse open inputs.
- Commit 7 (tsinfer-match-eval): 784ce5010b0565201442a73321240fa1b0622fc4 — Prepare copying paths in explicit passes.
- Validation for 1: exact index snapshot, six existing configuration checks
  passed without personal configs. Ignored personal copies remain in place.
- Validation for 2: exact index snapshot, nine existing settings/manifest checks
  passed. Tiny fixture paths and explicit HMM sample size were included.
  Epsilon known answer confirmed the explicit span, pre-singleton source-site
  denominator and selected haploid size. Unrelated dependency versions equal
  the saved baseline. TSInfer packaging overrides were discarded as requested.
- Validation for 3: 45 existing interface/conditioning/reuse checks passed.
  Saved 231 tiny scalar path/error results and matrix references for seven
  panels; batch comparisons cover pins, overlaps, backwards/whole-sequence
  windows, zero rows and output preservation.
- Validation for 4: 113 affected existing native checks passed, as did the
  saved scalar/matrix references and mixed normal/regional API comparisons.
- Validation for 5: full existing native suite passed (688 tests).
  Final normal-project build used -O3, -UNDEBUG and -std=c11. One-time optimized
  object/disassembly inspection confirmed normal scalar/matrix edge, node and
  site loops contain no regional workspace accesses, tracking work or added
  regional cache stores. The strategy marker is used at call boundaries only.
  Normal reset/root order, scalar seeding, traceback preparation and incremental
  positioning retain the starting algorithms. Shared numerical site functions
  remain in use. No elapsed-time equivalence is inferred from inspection.
  A bounded allocation check confirmed scalar/conditioned/matrix calls and empty
  batches allocate no regional workspace; nonempty batches allocate and reuse it.
  Inspection artifacts remain only in the execution scratch directory.
- Validation for 6: 13 existing packing, reconstruction and failure/lifecycle
  checks passed. Active matching, diagnostic, profiling and test packing callers
  supply opened inputs and explicit mappings; preparation precedes sample loading.
- Validation for 7: 21 existing conditional/unconditional, bridge, boundary,
  assembly, timer and diagnostic-selection checks passed using installed pinned
  Git dependencies. 240 tiny comparisons agree with the saved fused preparation.
  TSLS resolves as 0.1.dev125 at the final native commit. An initial provenance
  check using a source-tree import lacked installed distribution metadata;
  it passed after installing the final Git pin. No production fallback was added.
- Commit 8 includes this record and concise generic usage/configuration docs.
  Commit 9 and final pipeline integration validation remain pending.
