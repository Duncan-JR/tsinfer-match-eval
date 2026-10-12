# Fast stitched matching

This supersedes the whole-genome reduced pass in `stitched_matching.md`. Each
maximal focal component receives an independent reduced-reference HMM match;
gaps receive independent full-reference matches.

The recent dphil-analysis work supports this change: independent focal matching
substantially reduces focal switches without introducing focal mismatches in
the examined k=0 cases. Making gap observations missing would still leave
components coupled and is insufficient.

## Changes in tsinfer

### Scope and constraints

Keep the existing `MatcherIndexes`, `AncestorMatcher`, and `find_path`
interfaces. Reuse a reference's indexes and each worker's matcher across region
calls.

Limit implementation changes to:

- `lib/ancestor_matcher.c`: replace endpoint-tree reconstruction loops and
  bound site processing.
- `tsinfer/matching.py`: copy only the requested output slice.
- The two existing local fixes: `site = start` and `MatcherIndexes` cleanup.
- Focused verification in the existing tsinfer test suites.

Do not add custom native indexes, persistent cursors, allocation schemes,
reference-reduction APIs, or changes to the HMM recurrence.

This approach removes out-of-region tree traversal and site processing. It
uses filtered scans of the existing sorted edge arrays to identify endpoint
topology, and still initialises node-sized state arrays. These are explicit first-pass limitations; this is not a claim of
runtime proportional only to chunk length.

Follow `../../tsinfer/CLAUDE.md` for implementation style, rebuilding, C
coverage, and memory checks.

### 1. Construct only the forward starting tree

Use the seek-from-null principle from
[tskit PR #2874](https://github.com/tskit-dev/tskit/pull/2874): skip irrelevant
edge events and install only edges active at the requested position. Apply it
directly to the matcher's existing sorted `edge_t` arrays.

**Why not call the tree-position API here?** The current API requires a valid
`tsk_treeseq_t`. The matcher owns copied edges and mutation indexes, not a tree
sequence. The previous proposal added a complete owned reference copy, tskit
initialisation (including tree/site/node data), and constructor/destructor
changes to every indexes object, including those used for normal whole-ARG
matching. That is an unnecessary memory and setup-time regression. Replacing
the existing edge representation or changing Python/C ownership would broaden
this work further. Do not add that object, fabricate a partial tree sequence,
or introduce lazy construction and synchronisation. No header, constructor,
cleanup, or tskit dependency change is needed for sections 1 and 2.

In `ancestor_matcher_run_forwards_match`, replace only the "Load the tree for
start" block with two scans, using its existing `in` (insertion order), `out`
(removal order), and cursor variables:

1. Start both cursors at zero. While `in_index < M` and
   `in[in_index].left <= start_pos`, insert that edge only if
   `in[in_index].right > start_pos`, then advance `in_index`.
   Use the existing `insert_edge` and topology arrays.
2. Advance `out_index` past every edge with `right <= start_pos`, without
   applying removals: those edges were never installed.
3. Set `left = start_pos`. Set `right` to the minimum of `sequence_length`,
   the next insertion's left coordinate (if any), and the next removal's right
   coordinate (if any). Both cursors now point to events strictly after
   `start_pos`. Keep `remove_start = out_index` and the existing likelihood
   initialisation and subsequent edge-update loops.

Here `left` deliberately truncates the first tree interval at the first
requested site; the HMM does not need that tree's actual left breakpoint.
This avoids a breakpoint array, tree lookup, and cursor conversion. Insertion
order is retained among the active edges, and expired edges cause no topology
updates. Use the existing signed cursor types and bounds checks for empty or
exhausted arrays.

Keep `site = start`. Bound site processing with `site < end` and exit before
"Move on to the next tree" once the last requested site has been processed.
The topology need not advance to `end_pos`: traceback resets its parent array,
and the next call already resets the matcher.

### 2. Construct only the traceback starting tree

Apply the same principle in reverse using the existing sorted arrays. No tree
position object, extra arrays, allocation, or persistent state is needed.

In `ancestor_matcher_run_traceback`:

1. Retain the existing parent and recombination-state resets. Set
   `site = end - 1` and use `sites_position[site]` as the topology lookup
   coordinate, `last_pos`. Keep `end_pos = sites_position[end]` for output.
2. Starting at `M - 1`, scan `in` (removal order) backwards while
   `in_index >= 0` and `in[in_index].right > last_pos`. For each edge with
   `in[in_index].left <= last_pos`, assign `parent[child] = parent_node`.
   Decrement the cursor for every candidate. Only the parent array is needed.
3. Scan `out` (insertion order) backwards past edges with `left > last_pos`,
   without applying removals. Both cursors now point to events at or before
   `last_pos`. Preserve signed cursors and their `-1` exhaustion sentinel.
4. Set `pos = end_pos` and enter the existing reverse loop. Its first boundary
   update loops do nothing because their next events are at or before
   `last_pos < end_pos`. Its calculation of `left` from the next events finds
   the endpoint tree's actual left boundary. The first site's parent map is
   already installed; subsequent iterations cross earlier boundaries normally.
5. Bound the site loop with `site >= start`, and stop the outer loop once
   `site < start`, before any further topology updates.

The initial `right = end_pos` is a processing bound, not necessarily the
endpoint tree's true right breakpoint. This is safe because there are no
requested sites between `last_pos` and `end_pos`: the first iteration processes
only requested sites belonging to the installed tree. It skips all topology
events after the last observation, including events in that trailing gap.
Events exactly at `last_pos` stay pending for the next reverse iteration,
after that site's traceback, preserving half-open interval semantics.

Keep output path endpoints, recombination decisions, allele reconstruction,
and traceback cleanup unchanged. Do not share likelihood state or topology
between independent calls.

**Whole-ARG cost and scope:** these changes add no persistent state or native
allocations and leave index construction and per-worker memory unchanged.
For a whole-sequence call whose first/last sites lie in the first/last trees,
the scans touch the same endpoint edge groups as normal initialisation, with
a cheap activity predicate per candidate. Interior HMM traversal is unchanged
apart from explicit site bounds and the stopping condition. If there are
unobserved leading/trailing trees, their topology updates are skipped. Do not
add a separate whole-genome implementation or special-case dispatch without
measuring a meaningful regression. This is a cost assessment, not a measured
speed guarantee.

A regional call still scans an insertion-order prefix and removal-order prefix
forwards, and corresponding suffixes backwards: worst-case O(number of edges)
per endpoint. This is also a limitation of the proposed tskit seeks from null;
calling them would not provide an indexed active-edge lookup. Removing these
remaining scans or node-sized resets is deferred until timing shows a material
need.

### 3. Limit copying in the Python wrapper

In `AncestorMatcher.find_path`, replace:

```python
match_out[:] = matched
```

with:

```python
match_out[start:end] = matched[start:end]
```

Document that the call writes only the requested slice of `match_out`. Values
outside it remain unchanged.

Retain the current low-level allocation sizes and return format for this pass.
Those arrays are allocated uninitialised; removing the unconditional
genome-wide copy is the straightforward improvement. Region-sized native
allocations can be considered later if allocation costs remain material.

### 4. Retain existing fixes; withdraw the native simplification proposal

Preserve the existing local changes that:

- Begin forward observation processing at `start`.
- Call `matcher_indexes_free` during `MatcherIndexes` destruction.

Do not introduce a native wrapper around tskit simplification. Continue using
the existing Python/tskit reduction operation.

This withdraws the previous promise of thread-parallel reduced-reference
preparation. The repository's eventual executor choice must account for
simplification holding the GIL; a small preliminary timing probe does not
justify adding another native API.

### 5. Implementation order and verification

Implement in three reviewable steps:

1. Python slice copying and its output contract.
2. Replace forward endpoint reconstruction with filtered edge scans and add
   the stopping condition.
3. Reverse endpoint construction and site bounds.

After each C change, rebuild the extension before running Python checks, as
required by `CLAUDE.md`. Cover the changed branches in the existing C suite;
use classes and fixtures for Python cases.

Verify:

- Whole-sequence matches retain their existing results.
- Regions beginning or ending inside a tree process only their requested sites.
- Regions at tree breakpoints respect half-open edge intervals.
- Forward and reverse endpoint scans reproduce the endpoint tree's parent map,
  filtering expired/not-yet-active candidates and respecting descending stop
  sentinels. Verify cursor handoff at both boundaries without duplicate events,
  including multiple tree boundaries between the last site and `end_pos`,
  first sites after coordinate zero, and exhausted edge arrays.
- Single-site regions, first/last sites, missing observations, and relevant
  match-impossible cases behave correctly.
- Changing observations outside `[start, end)` does not change the regional
  result.
- Reusing a matcher across regions in arbitrary order agrees with fresh
  matchers.
- The wrapper leaves sentinel values outside the requested output slice
  unchanged.
- Repeated reduced-index construction and destruction does not accumulate
  native memory. Sections 1 and 2 introduce no new ownership or cleanup paths.

Finally, time representative short and long focal/gap calls at several
chromosome positions using already-constructed indexes and matchers. Separately
include preparation in end-to-end timings. Compare ordinary whole-sequence
matching against the current implementation on the same full references,
including index construction time, repeated match time, and process peak RSS;
require no material regression. Use process memory measurements because the
matcher's existing memory counter does not account for all native allocations.

The acceptance criterion is removal of repeated out-of-region topology updates
and whole-array output copying without changing regional matching semantics.
If the remaining flat edge scans dominate realistic timings, report that result
before expanding the C implementation.

## Changes in tsinfer-match-eval

### 1. Match every mask run independently

In `match_stitched_haplotype`, prepare one reduced reference per haplotype and
AC cutoff, reused across budgets. Retain roots 0 and 1 plus eligible focal nodes,
preserve sites, and invert the node map for stitching.

For each budget:

1. Clip mask runs to the existing called-site domain.
2. Split both focal and gap runs at inference-interval boundaries.
3. Match focal runs against the reduced reference and gaps against the full
   reference.
4. Fill the stitched parent and copied-allele arrays, then use the existing
   path merging and native assembly.

Remove the whole-genome reduced match. A larger component's path cannot be
clipped and reused for a smaller component because its endpoint optimisation
differs.

Preserve missing-call handling, original parent IDs, output names, and native
`MatchResult`/`extend_ts` assembly. Update documentation describing the previous
whole-genome approach.

### 2. Follow native matching orchestration

Keep the existing coarse Snakemake job per dataset/AC/method, producing all
budgets.

Use tsinfer's `ThreadPoolExecutor`/`as_completed` pattern with:

- One shared reference, full-reference indexes, focal archives, masks, and
  scheduled `HaplotypeReader`.
- One complete task per haplotype, processing all budgets.
- A reusable full-reference matcher owned by each worker thread.
- One transient reduced matcher per active haplotype.
- One haplotype read inside its task and deterministic final ordering by native
  haplotype index.

This provides cohort parallelism while avoiding process startup, replicated
references, and haplotype/result pickling. Keep region calls sequential within
each haplotype so matcher allocation and scheduling costs are amortised.
Respect Snakemake's granted CPU allocation.

Reuse native jobs, reader caching, result structures, and assembly; a new
worker-management framework is unnecessary.

The tsinfer revision above withdraws the native simplification helper on which
fully parallel reference preparation depended. Before implementing this
thread-based runner, account for the existing simplifier holding the GIL when
choosing the executor. Do not assume reference preparation scales with matching
threads or add a native helper to make that assumption true.

### 3. Remove obvious preprocessing overheads

**Focal ancestor lookup:** select focal seeds first, then load and polarise
calls only at unique selected seed sites. Currently `find_focal_ancestors` loads
the entire inference-site call matrix although candidate membership uses only
seed observations. Preserve the current identities, ordering, and derived-AC
semantics.

**Ancestor intervals:** keep physical ancestor-column block scheduling and the
directional event sweep. Run blocks in a shared-memory thread pool and compile
`_sweep_block` with `nogil=True`. Each block owns its sweep state and outputs;
sample calls and association inputs are read-only. This avoids copying the
cohort call matrix into every spawned process and repeated worker-local JIT
compilation.

**Focal masks:** retain the interval sort, union, and run representation. These
already avoid dense per-site coverage work. Generate all requested masks in one
dataset job so the compressed interval arrays are loaded once, preserving
separate mask outputs.

### 4. Verification before timing experiments

Use existing checks and small ad hoc runs with `uv run`; add no test suite in
this repo.

Verify that:

- Changing observations outside a region cannot change its match.
- Boundary, single-site, missing-call, and repeated-call cases remain correct.
- Independent focal results reproduce the diagnostic control, allowing
  equivalent tied paths.
- Serial and parallel stitching agree and preserve called sample alleles.
- Shared preprocessing reproduces existing candidate, interval, and mask
  outputs.
- Repeated reduced-reference preparation does not accumulate native memory.

Measure preparation, focal matching, gap matching, and assembly separately on
representative haplotypes, then include them in end-to-end timings. Chunk calls
should no longer replay chromosome prefixes or suffixes or copy genome-sized
outputs; the remaining edge scans and allocation sizes are the explicit
limitations described in the tsinfer section.

Leave sparse node resets, boundary conditioning, greedy matching, cross-sample
reference caching, and finer region scheduling for evidence from those timings.
