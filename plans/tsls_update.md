# tsls integration MVP

## Scope

Adopt upstream tsinfer's tsls matcher and retain the independent Python changes
needed by this pipeline. Add three capabilities to tsls: fixed endpoint
conditioning, topology reuse across ordered regional calls, and regional output
without chromosome-sized allocation or copying per call.

Keep the greedy candidate selection, exact cover, one-site bridges, and
inference-interval splitting. Use one immutable `tsls.TreeSequence` per reference
and one reusable `tsls.Hmm`, `ExactState`, and matched-output buffer per worker.
The evaluator's matching workers share the reference within one process.

All new C matching changes belong in tsls. Retain its compressed likelihoods,
deferred costs, sparse updates, and delta traceback. The MVP does not include
arbitrary boundary-weight arrays, conditioned forward/backward output, new
pruning methods, donor-restricted panels, multi-gap batch APIs, or a new index
sharing/serialization format.

## Matching contracts

### Gap boundaries

For a residual gap `[a, b)` in inference-site indexes:

| Available copied anchor | HMM window | Fixed copying state |
|---|---|---|
| Neither | `[a, b)` | None |
| Left | `[a - 1, b)` | Left donor at site `a - 1` |
| Right | `[a, b + 1)` | Right donor at site `b` |
| Both | `[a - 1, b + 1)` | Both anchors |

Endpoint constraints apply at the first and last **included sites**, rather than
the first and last gap sites. Padding includes the actual transitions into and
out of the gap. Clip the returned path and mutation extraction to `[a, b)`;
adjacent copied intervals retain ownership of anchor mutations and bridge fixes.

Use only anchors supplied by `greedy.stitched_intervals`, which already suppresses
conditioning across inference boundaries. Preserve `condition_gaps=False` as
unpadded, unconditioned matching. Process a haplotype's gaps in increasing order
of their padded starts. Every gap starts with fresh likelihoods, including when
its padded window overlaps the previous window.

### Reference, alleles, and coordinates

- State IDs are reference node IDs from `catalog.parents`. A copying state must
  have a parent or at least one child in the local tree; isolated nodes are not
  states. An internal-node pin identifies that node alone, not its descendants.
- Keep the complete reference and its node IDs. Use `tsinfer.matching.hmm_parameters`
  once per model/reference, including the original denominator `n`; do not
  recompute rates or restrict the state space per gap.
- Construct the tsls panel with the same explicit per-site allele codebook used
  for sample calls. Preserve missing calls, recurrent mutations, reversions, and
  alleles absent from the panel. Remove the old manual 0/1 mutation encoding.
- Apply upstream's ultimate-root validation and normalization at the integration
  boundary. The generic tsls API must not assume node 0 is equivalent to node 1.
- Inputs use site indexes. Returned segments use physical coordinates and are
  ordered right to left. Preserve the existing endpoint convention: zero at the
  first site-index boundary, sequence length at the last, and site positions at
  internal boundaries.
- Results passed out of a worker must own their data; they cannot retain views of
  scratch that a later gap or haplotype will overwrite.

## Conditioning design

Implement ordinary HMM boundary conditions. For a requested window `[s, e)`, the
Viterbi objective is defined by:

```text
v_s(u) = initial(u) * emission_s(u)
v_j(u) = emission_j(u) * max_v(v_{j-1}(v) * transition_j(v, u))
last_state = argmax_u(v_{e-1}(u) * terminal(u))
```

With no pins, use the usual uniform initial weights and terminal weights of one.
A fixed initial state supplies a point mass; a fixed final state supplies a
terminal indicator. This gives the most likely path conditional on the endpoint
states when run without approximation; configured flooring/pruning retains its
usual approximation semantics. Endpoint identities remain hard constraints in
either case. Matching needs one forward sweep and the existing traceback,
without a backward probability pass or changes to the transition model.

The public MVP API needs only optional `initial_state` and `final_state` keywords
on `Hmm.find_path`. Keep initialization and terminal selection separate from the
recurrence, so these are ordinary HMM operations independent of the greedy
application. User-supplied boundary distributions can be added later without
changing these semantics. In particular, an initial state must not change the
distribution used to choose donors on later recombinations.

### Initialization and termination

Initialize the first included site's costs from the boundary weights and its
emission. There is no transition from an unobserved predecessor into that site.
Reuse the existing emission and delta-recording machinery, with an explicit
first-site initialization step; do not patch the input recombination rates or
overwrite `switch_cost[start]` to impose a pin. Preserve the existing
unconditioned path and score conventions, and make first-site normalization and
traceback bookkeeping consistent with the boundary step.

Represent the initial point mass using the existing compressed costs: local
roots have infinite cost, the selected node has zero cost, and its immediate
children have explicit infinite costs so descendants cannot inherit its value.
Handle a selected root without duplicate entries. Keep `value`, `current`,
last-change site, and listed/cache state consistent, preserving the separate
not-in-tree sentinel. Apply the hard initial-state mask after any first-site
flooring/pruning so those numerical operations cannot admit other starting
states. The seed is the baseline before the first emission; record the first
site's changes normally and update the traceback baseline assertions.

After processing the final site, select the requested terminal node instead of
the globally best node. Resolve its effective cost through its nearest listed
ancestor and `tsls_hmm_deferred_cost`; neither its raw `value` nor its cached
`current` is necessarily the terminal cost. Start the usual traceback at that
actual node. Do not replace the final site's best-node record, since traceback
still needs the ordinary prefix maxima.

### Numerical settings and errors

Accept the HMM's configured `likelihood_threshold`, `prune_threshold`, and
`relative_prune_threshold` unchanged. Preserve existing defaults when they are
unspecified. Neither tsls nor the evaluator should require zero thresholds,
override them when conditioning is enabled, or reject a configuration merely
because it combines nonzero thresholds with endpoint constraints.

Flooring/pruning retains the existing approximation semantics; endpoint IDs
remain hard constraints. Keep compressed likelihoods, deferred updates, and exact
sparse/dead-state shortcuts. The last site's dead ceiling remains infinite.

Follow the current division of checks: the binding handles array shapes, the C
entry point checks the site range, and the sweep checks haplotype alleles and
impossible updates. Add endpoint ID bounds checks before indexing and check local
membership when the cursor reaches each endpoint. Map invalid endpoint arguments
to `ValueError` through the existing error mechanism.

Use `MatchImpossible` if the terminal cost is infinite, one-site pins conflict,
or traceback cannot recover the requested initial state. Verify that initial
state once when traceback reaches the first site. An error caused by lost
history should describe failure under the supplied parameters, without claiming
the underlying model has no solution. If a traced switch has zero recombination
probability, return the same matching error at the existing switch branch. Keep
internal invariants as assertions and use the existing `extended_checks` path
for detailed state checking. Propagate failure without changing parameters or
dropping constraints.

## Planned commits

### 1. tsinfer — integrate upstream

**Planned commit message:** `Merge upstream tsls matching and retain pipeline APIs`

Create an integration branch from `fix_reference_ts` and merge upstream main.
Resolve the matcher replacement in favor of upstream's sole tsls engine, explicit
allele mapping, shared reference, root conventions, and contract tests. Remove
dependencies on the deleted legacy C matcher and index bindings.

Retain the small independent Python changes:

- In-memory reference configuration support.
- `append_match_results` / `finalise_match_tables`, reconciled with current
  `extend_ts`, so streamed results need only one final sort/index/mutation-parent
  calculation.
- `ScheduledCache` in-flight deduplication and error propagation.
- Post-processing and tsdate support.

Accept upstream dependency and ancestor-builder fixes. Update affected usage
documentation concisely. Preserve the existing regression tests for the retained
Python changes. Keep ordinary upstream full-matcher scheduling; a separate HMM
cache for that path is outside the MVP.

### 2. tsls — add endpoint conditioning

**Planned commit message:** `Add endpoint conditioning to Viterbi matching`

Implement the conditioning contract above in `tsls/matching.py`, `_tslsmodule.c`,
`lib/hmm.c`, and the associated declarations/errors. Expose optional keyword-only
`initial_state=None` and `final_state=None`; existing calls remain unconditioned.
Keep tsinfer's anchor selection and padding out of tsls.

Add the conditioning tests in the essential test outline below, using the
existing dense reference and small tree builders.

### 3. tsls — retain topology across regional calls

**Planned commit message:** `Reuse tree topology across ordered regional matches`

Make topology reuse an internal optimization of `find_path`; no ordered-call
option or new session API is needed. Keep one tree-position cursor and the
existing parent links and child counts in the HMM. Separate topology positioning
from likelihood initialization.

1. On the first call, build the start tree as now. On later calls whose start tree
   is at or after the retained cursor, apply only intervening edge removals and
   insertions. Skip observations, rate updates, and traceback recording while
   fast-forwarding between windows.
2. Reset likelihoods, endpoint pins, deferred-cost ceilings, traceback state,
   statistics, allele/walk caches, and sparse indexes for each call. Retain only
   topology and its cursor. The existing node-wide likelihood reset is acceptable
   for this MVP; it must not clear parents or child counts or rebuild edges.
3. The forward sweep leaves the cursor at the final site's tree. Start traceback
   from that tree directly, removing its current second seek from null and
   topology rebuild. Initialize traceback list flags from the forward state as
   needed.
4. During traceback, maintain child counts as well as parents while moving the
   same cursor backwards. Retain the ordinary null-parent and `ROOT_ALLELE`
   conventions, using the existing root-aware allele lookup/unset machinery
   instead of rewriting every root's parent to a virtual root. Preserve delta
   undo ordering and the existing traceback representation.
5. After a successful call the cursor and topology describe the first site's
   tree. Fast-forward from there to the next start, including the prior window's
   edge changes where necessary. This intentionally avoids a second topology
   copy; fresh likelihoods mean no probabilistic state crosses the skipped span.
6. For an earlier start tree, rebuild using the existing cold path. Invalidate
   the retained cursor after errors or other HMM operations that overwrite its
   topology. The next call rebuilds safely. A new haplotype requires no special
   state propagation or public reset method.

Test cursor reuse against fresh HMM calls as outlined below. Inspect traversal
costs in the final profile.

This removes repeated topology construction, not all per-node work. Keep the
remaining likelihood/traceback preparation visible in the final profile.

### 4. tsls — bound regional output work

**Planned commit message:** `Write regional matches into caller-owned output buffers`

Add `preserve_outside=False` to `Hmm.find_path`:

```python
left, right, parent = hmm.find_path(
    h, start, end, match_out,
    initial_state=left_parent,
    final_state=right_parent,
    preserve_outside=True,
)
```

Check the range before allocating. Size each path array to `end - start`, the
maximum number of output segments. Allow the native binding to write directly
into the caller's writable contiguous reference-length `int8` matched array,
retaining its Python owner while native code runs. Keep absolute site indexing.
Check this new output-array contract once in the binding, without coercing the
array to a detached copy.

The opt-in path leaves values outside the window untouched and performs no
chromosome-sized matched-array allocation, fill, or copy. The default retains
missing-outside-window behavior. Return owned path arrays as before; no reusable
native path views or gap-relative pointer tricks are needed.

Add the regional-output tests below and inspect allocation sizes in the profile.

### 5. tsinfer — consume the completed tsls API

**Planned commit message:** `Pin tsls with conditioned regional matching support`

After commits 2–4, update the tsls source pin and lockfile. Sibling editable
checkouts are useful during development; record the tested commits for the
reproducible integration. Extend the existing tsls contract tests with one
regional integration case as outlined below. Do not restore a legacy matcher
facade.

### 6. tsinfer-match-eval — port matching and workers together

**Planned commit message:** `Port full and greedy matching to shared tsls workers`

Update `lib/matching.py`, `lib/haplotypes.py`, `lib/benchmark.py`, and dependency
configuration together. Add tsls as a direct dependency and pin the tested
tsinfer/tsls revisions. Share the panel/model setup between the main runner and
resident full-HMM benchmark to keep allele handling consistent.

- Build one `tsls.TreeSequence(reference,
  alleles=samples.allele_mapper.forward_map())` in the coordinator. Construct rates
  once per model and keep the immutable reference and donor inputs alive until
  matching finishes. Remove legacy native-index export/import and descriptors.
- Replace the evaluator's spawn pool with persistent worker loops in a bounded
  `ThreadPoolExecutor`, using its existing task/result queues and dataclasses.
  tsls already supplies the shared immutable reference; no separate reference
  sharing phase or ABI is needed. Each worker exclusively owns one reusable HMM,
  `ExactState`, and `np.empty(num_sites, dtype=np.int8)` output scratch. Native
  matching and the main greedy Numba kernels release the GIL.
- Submit whole haplotypes, processing each one's gaps in order on that worker.
  Preserve bounded decoded sample slabs, pending work, and append-on-arrival.
  Reuse existing packed/mapped inputs where useful. Configure helper threads and
  warm the needed Numba signatures before timed work. Retain worker readiness and
  propagate loader/worker failures without hanging queues; remove spawn-only
  process watchers and startup machinery.
- Adapt `_match_region` to the endpoint and output APIs, preserving padding,
  clipping, root normalization, and the existing coordinate conversion. Compare
  calls against matched output only inside the residual gap. Keep greedy
  candidates, cover, stitched intervals, bridge corrections, focal-region
  filtering, and missing-data handling unchanged.
- Only the coordinator mutates output tskit tables. Pre-create sample nodes in
  job order, append owned results as they arrive, finalize once, and release
  worker scratch/mappings before final sorting/indexing. The coordinator owns
  shared mapping lifetimes; individual workers must not close them.

Port `_resident_full_hmm` through the same panel and codebook; its current legacy
construction omits explicit mutation states. Reuse an HMM per resident benchmark
worker. Include setup in end-to-end timing and exclude it from the resident
window only after workers report readiness.

Update benchmark accounting as part of this port: record tsls version/revision
and effective floor/pruning settings; remove borrowed-index counters; sample
memory once per process rather than once per thread. Remove worker memory
decorators that would race on the same PID-named files. Identify measurements by
worker index, use thread CPU time appropriately, and label `Hmm.total_memory` as
traceback allocator bytes rather than worker RAM.

Run the existing greedy and resident/end-to-end smoke workflows with one worker
and a small parallel run, retaining their genotype-reconstruction and streaming
checks. Detailed HMM tests belong in tsls.

### 7. tsinfer-match-eval — profile and fix obvious overhead

**Planned commit message:** `Add tsls regional profiling and benchmark configuration`

Use one representative prepared reference/cohort and its actual recombination
model, including a genetic map if that is the target workload. Keep the reference,
codebook, rates, and greedy settings fixed. Use existing `full_hmm` and `greedy`
methods and config controls; compare conditioning on/off in separate runs. Start
with one worker and one representative parallel allocation, one replicate each.
Repeat only when variability prevents interpretation. Use the same effective
floor/pruning settings for full and greedy comparisons, retaining the configured
defaults for performance runs rather than forcing an exact-mode policy. Report
any conditioning failure with its window, pins, and effective parameters.

Include a focused profiling test replaying real ordered, padded gap windows:

- Compare a reused HMM with independent cold calls under identical numerical
  settings, checking constrained scores, pins, and output consistency with each
  returned path. Allow tied paths to differ. Include short gaps, long skips, and
  overlapping anchors.
- Record call count, gap/window sizes, and native time. Separate topology
  positioning, likelihood reset, forward processing, traceback preparation/sweep,
  and output handling sufficiently to identify dominant overhead. Use profiler
  traces or temporary instrumentation, without per-node production logging.
- Confirm reusable calls avoid null seeks and complete topology rebuilds, skipped
  spans process only edge changes, and output does no chromosome-sized clear/copy.
- Compare resident and end-to-end wall/CPU time, process memory, and worker
  scaling. Retain candidate/cover, conversion/mutation extraction, assembly, and
  finalization timings so Python/GIL work and I/O remain visible.

Fix obvious regressions before completing the MVP: repeated reference/HMM
construction, unnecessary topology rebuilds, full-length output work, duplicate
memory sampling, and avoidable serialization. If node-wide likelihood resets
dominate, make the smallest measured improvement that preserves independent gap
likelihoods; do not introduce a general cache framework pre-emptively. Use a
follow-up commit message describing the concrete fix.

Keep quality checks enabled. Tied optimal paths may differ; compare scores and
reconstructed calls rather than requiring identical old/new edge arrays. A
greedy constrained solution need not attain the unconstrained full-HMM optimum.
Store measurements with experiment outputs and keep README/config changes to
concise usage and option descriptions.

## Essential test outline

Use the repositories' existing pytest and CUnit suites. Put new Python tests in
small classes with pytest fixtures for setup, direct assertions, NumPy testing
helpers, and `pytest.raises` for expected errors. Give each test one purpose and
explicit inputs. Parameterize short lists of endpoint cases, not products of
panels, seeds, rates, and donor choices.

### tsls

Keep Python algorithm tests in `tests/test_matching.py` and reuse
`single_tree_ts`, `two_tree_ts`, `gap_ts`, `make_matcher`, and `path_by_site` where
needed. Pick a specific builder per test instead of expanding the parameterized
`example` fixture. Extend the existing Viterbi and path-score helpers in
`tests/dense.py` for the endpoint semantics, including first-site initialization;
keep their remaining recurrence unchanged.

| Test group | Essential coverage |
|---|---|
| Conditioned optimum | On one small existing tree, compare the returned path's score with the constrained dense optimum for no pin, left pin, right pin, and both pins. Choose an internal initial node and a forced non-best terminal node among these cases; assert their actual IDs. Use zero-floor, unpruned settings only for this oracle comparison. Allow tied paths to differ. |
| Sparse conditioning | One conditioned case on the same tree compares `sparse=True` with `sparse=False`. Set `index_min_sites=1` so the tiny example exercises sparse processing, and confirm at least one sparse site ran. Compare scores and endpoint IDs. |
| Parameters and errors | In `TestInterface`, check one conditioned call with defaults and one explicit finite-floor/nonzero-pruning setting. Check an out-of-range pin, a locally absent pin, and conflicting one-site pins. One zero-recombination example covers a reachable same-donor path, failure for incompatible donors, and successful reuse after that failure. |
| Regional reuse | On `gap_ts`, use one explicit short sequence of ordered and overlapping windows, an earlier window, and a second query. Compare each call with a fresh HMM under the same pins and parameters. This fixture already includes topology changes between observed sites. Run with `extended_checks=True`. |
| Regional output | Extend `TestInterface` with a subwindow call using `preserve_outside=True`: outside sentinels survive, inside values match the returned path, and coordinates agree with the default call. Include a one-site window. Use a short list of invalid output arrays to cover length/dtype and writable/contiguous requirements. Retain the existing missing-outside-window test for the default mode. |

Keep `lib/tests/tests.c` lean: extend `test_forest_matching` with a conditioned
regional call and a subsequent call on the same HMM; add the new endpoint error
return codes to `test_bad_inputs`. Reuse `build_forest_ts`. Detailed optimal-path
comparisons stay in Python. Update the existing extended-check assertions for
the new initialization and retained topology where needed.

### tsinfer

Use the upstream fixtures and test classes after the merge. HMM correctness is
tested in tsls; tsinfer tests cover its integration and retained Python behavior.

| Location | Essential coverage |
|---|---|
| `tests/test_tsls_contract.py::TestTslsContract` | Add one conditioned regional call using the existing `panel_ts` and `allele_mapper` fixtures, tsinfer's HMM parameters, and reusable output mode. Check the requested donor IDs, matched allele codes, and regional coordinates. Retain the existing tests for root normalization, missing data, shared-reference threading, and ordinary Matcher results. |
| `tests/test_matching.py::TestExtendTs` | Retain/adapt `test_append_batches_matches_extend`: two streamed appends and one finalization give the same tables, mutations, and metadata as `extend_ts`. |
| Existing config/pipeline/cache tests | Retain the in-memory-reference tests in `test_config.py` and `TestReferenceTs`/`TestReferenceTsEquivalence`, the concurrent-load and failure tests in `TestScheduledCache`, and affected post-processing tests. Adapt calls to the merged APIs rather than adding another test matrix. |

Rebuild changed C extensions before Python tests. Run focused classes during
development, then both full Python suites with coverage (`uv run pytest
--cov=tsls` and `uv run pytest --cov=tsinfer` in their respective repositories),
the tsls CUnit suite and existing C memory checks, and the repositories' existing
lint/format checks before the PR. Inspect coverage for the new branches. Keep
timing and allocation measurements in commit 7's profiling workflow.

The MVP is complete when conditioned regional results are correct, ordered calls
reuse topology with fresh likelihoods, streaming remains bounded, and the profile
shows no obvious integration overhead left unfixed.
