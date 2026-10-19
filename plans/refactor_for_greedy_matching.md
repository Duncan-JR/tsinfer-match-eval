# Refactor for greedy stitched matching

## Decision and scope

Read and decode by physical Zarr chunk; match one complete haplotype per job.
Use a packed, read-only ancestor reference, direct exact extension from each
eligible seed, a farthest-reaching interval cover, and the existing HMM for
uncovered gaps. Storage chunks must not become biological boundaries.

Keep the notebook's second greedy rule, not a literal translation of its
Python containers. It solves a precise problem: cover each retained connected
union of the supplied exact intervals with the fewest copying runs. It does
not optimise the full-reference HMM likelihood, choose the biologically best
parent, or minimise switches across independently matched gaps.

Use one persistent multiprocessing pool, one sample-loader thread and one
coordinator that owns output. Bound outstanding jobs and live sample slabs;
do not introduce a cache framework or independently schedule ancestors or gaps
during haplotype matching. The proposed speedup remains to be measured.

## Fixed semantics and products

Use scalar `ac_cutoff` and `min_focal_region_size`, one `focal_choice` per run
(default `left`), and zero mismatches. Preserve selected-cohort derived AC,
canonical allele encoding, sample identities and original reference node IDs.
An eligible ancestor has a chosen seed with `derived_ac <= ac_cutoff`; it is a
candidate for a haplotype only when that haplotype carries the derived seed.
Different ancestors sharing a seed remain separate candidates.

Each exact interval contains its seed and stops at a mismatch, missing call in
either sequence, ancestor support or the containing inference interval.
Bounds are half-open inference-site indices. Touching intervals join, and the
minimum size counts inference sites in the complete connected union, not base
pairs, individual ancestors or individual copying runs. Preserve the current
site-axis connectedness even where distinct inference intervals touch on that
axis; do not silently introduce an additional split for minimum-size filtering.

Retain annotation, ancestor inference/matching and the independent full-HMM
baseline. Remove saved focal-candidate/interval/mask NPZs, reduced references,
focal-method selection, mismatch budgets and biological hyperparameter loops.
The required matching output is the raw tree sequence. Native-shaped path
JSONL is optional and must not affect matching or require retained results.

## Architecture and validation boundaries

- `lib/haplotypes.py`: reference catalog, physical chunk loading, allele mapping
  and small mapped-input descriptors.
- `lib/greedy.py`: array-only candidate/extension and component-cover kernels.
- `lib/matching.py`: a directly callable single-haplotype matcher, process
  scheduling, gap integration and output assembly, in separate plain functions.

Use small dataclasses at Python boundaries and ordinary arrays inside kernels.
No generic job framework, storage backend hierarchy or compatibility layer.
Validate configuration once at the entry point and cross-store/site/identity
alignment once when preparing inputs. Establish valid clipped supports and a
called, derived ancestor seed once during reference preparation. Kernels can
then trust these contracts; retain only checks needed to perform matching.
An empty eligible catalog is valid: use empty arrays rather than a zero-byte
memory map, and let every haplotype follow the normal gap path.

Use the actual editable `../tsinfer` checkout required by this repository and
record its revision. Confirm its regional matcher, topology-reset and output
contracts before integration; do not add runtime fallbacks for other versions.

## Reference preparation

Build a catalog in original ancestor-column order containing parent node ID,
panel column, chosen focal-site index, clipped support bounds and a 64-bit
offset into a flat `int8` genotype buffer. Map panel IDs to reference nodes once.
Eligibility uses annotated metadata, not a sample-by-ancestor association pass.

Allocate the packed buffer in a temporary mapped file, with one contiguous
slice per eligible ancestor. Its length is the sum of clipped support lengths.
Queue one preparation job per required physical ancestor-column chunk. That
job reads each variant block intersecting its ancestors' supports once, copies
the relevant intersections into its disjoint destination slices, and releases
the decoded block. Preserve missing calls within support. Use Zarr v3 library
block access, not direct chunk-file decoding.

Wait for preparation to finish before matching; this is the single data
preparation barrier. Reuse the processes for matching, constructing each
process's HMM state only once for that phase. Share the large genotype buffer;
the small catalog can be sent once per worker without a second sharing system.
There are no ancestral Zarr reads during sample matching.

## Sample loading and lifetime

Prepare reference-site alignment, selected source-column mapping and numeric
allele remapping once, following the native reader's conventions. Group selected
individuals by physical source-column chunk, keeping both ploidies together.
For each group, the loader:

1. Reads each physical variant/sample block containing reference sites once.
2. Selects the required sites and individuals before remapping.
3. Writes canonical `int8` calls into a mapped slab of shape
   `(selected_haplotypes_in_chunk, num_reference_sites)`.

Rows are contiguous complete haplotypes. Do not retain encoded calls for
unselected individuals. Decoding still incurs the full raw physical block.
Use a narrow local reader adapter; do not instantiate a native reader/cache in
every subprocess or require an upstream reader redesign.

`sample_prefetch_chunks` (initially 2) bounds all live slabs, including one being
loaded, ready slabs and slabs still used by workers. Publish only complete
slabs. The loader publishes a descriptor and starts the next permitted slab;
it does not submit all haplotype jobs itself. The coordinator feeds individual
rows to workers and releases a slab after its final result has been appended.
Workers close the sample mapping before reporting completion.

Use one temporary directory for mapped inputs. No persistent packed format,
resume scheme, automatic rechunking or mapping cache is needed. Mapping shares
large input pages under `spawn`; it neither guarantees RAM residency nor removes
temporary-file I/O from the cost.

## Single-haplotype algorithm

### 1. Discover candidates and extend exact intervals

One Numba routine scans the eligible catalog. For a derived sample seed, extend
left and right within the packed support, stopping at the first mismatch or
missing call. Start around the already validated seed and include it exactly
once. These are local outward comparisons, not whole-genome directional passes.

Write `left`, `right` and candidate identity into reusable worker scratch of
capacity equal to the eligible ancestor count. Return the valid count. Keep no
length arrays, mismatch counters, active-site sets, carrier lists or global
association/endpoints archive.

For A eligible ancestors, K candidates and E inspected extension calls, this
stage costs O(A + E). The catalog scan is the initial implementation; an inverted
seed index or alternative candidate search requires evidence of a bottleneck.

### 2. Cover components greedily

Stable-sort the valid candidate indices by `left` using NumPy outside the JIT
kernel. Equal left bounds retain catalog order; equal reaches keep the first
candidate encountered in this sorted order.

The numeric sweep has three natural levels: components, copying runs and
candidate consumption. At a fixed cursor, consume all not-yet-consumed
intervals starting at or before it. Choose the one whose right endpoint reaches
furthest beyond the cursor. Append that copying run and advance to its right
endpoint. If none extends the cursor, the component is complete.

Save the output count on entering a component. Keep its runs only when its
complete width meets the minimum; otherwise restore the saved count. The last
component and empty candidate set need no separate finalisation algorithm.
Numeric output arrays need capacity at most K.

Why consuming alternatives is safe: all candidates inspected at a cursor end
no later than the chosen endpoint. None can help after reaching that endpoint.
Choosing a shorter reach cannot reduce the number of subsequent runs needed.
The sweep therefore needs neither a heap nor a persistent active set. Each
candidate is consumed once despite the nested loops: O(K) traversal after an
O(K log K) sort, with O(K) useful endpoint/run state.

Do not select by total interval length or process only in seed order. For
example, at cursor 8, `[6, 12)` is longer than `[8, 13)`, but the latter reaches
further. Do not discard short intervals before finding components: they can
bridge a union that passes the threshold.

### 3. Match gaps and assemble

Walk the retained focal runs to derive their complement in the sample's
matching domain. Include leading/trailing gaps; touching focal runs leave no
gap. Rejected components naturally merge into surrounding gaps. No mask or
separate component product is required.

Keep the native domain: first through last called site, or the full reference
site axis for an all-missing haplotype. Each worker owns one `MatcherIndexes`,
one reusable `AncestorMatcher(reuse_topology=True)` and matched-allele scratch.
Reset topology for each new haplotype. Visit gaps in order, splitting HMM calls
at existing inference-interval boundaries. Gap likelihoods remain independent;
topology reuse is not boundary conditioning.

Collect each gap's mutations before reusing HMM scratch. Exact focal copying
adds none. Merge touching final runs with equal parents and preserve existing
site-to-base-pair conversion, site-zero and sequence-end conventions. Keep the
native HMM wrapper narrow; do not redesign coordinate semantics in this refactor.

Return a small dataclass of compact numeric path/mutation arrays and job
identity. Results must own their data before enqueueing: no views into reusable
scratch or a sample mapping. Copy only where ownership must change, not at
every helper boundary. Construct Python JSON records only in the output owner.

## Numba and process scheduling

Use serial `@numba.njit(cache=True, nogil=True)` numeric kernels. No `parallel=True`,
`prange`, fast-math or Numba object graphs. NumPy handles the stable sort; Python
handles loading, orchestration and native HMM calls. Pass settings as arguments.
Keep the kernels together so cached compilation does not hide cross-module
kernel changes. Warm the actual layouts/read-only signatures in every worker
before resident measurements; warming only the parent is insufficient under
`spawn`. Include real startup/cache-loading/compilation in end-to-end timing.

Use one explicit `spawn` context and one coordinator-owned admission count:

`pending = submitted haplotype jobs - results appended`

Require `pending <= max_pending_haplotypes`. This bounds queued, running and
completed-but-unappended jobs together. Use it to control submission rather
than separately tuning work-queue and result-queue limits. A work queue sized
to this cap and a result channel bounded by admission are sufficient. The
coordinator drains events and refills available slots continuously, without
batch barriers, busy polling or `Queue.qsize()`/`empty()` decisions. Apply the
same submit/collect pattern to preparation jobs.

The loader's slab-slot bound is separate because it controls large decoded
inputs, not jobs. Ready-slab notifications, completions and errors can use one
coordinator event channel. Never wait for a slab slot while neglecting results
that would release one. Retain only counters and descriptors needed by these
two lifetimes, not multiple scheduling abstractions.

Pre-create sample nodes in canonical job order, preserving individual,
population, flags, time and identity metadata. Append each completed result to
its assigned child node immediately; finalise/sort tables once. Use a narrow
local append helper because the current ordered native append contract is not
an unordered-results interface. Optional JSONL carries job identity and can
also be written in completion order.

Use the allocated CPUs while accounting for the loader/helpers and per-process
HMM memory. Start with single-threaded numerical/codecs and low Zarr helper
concurrency, configured before process startup. Surface loader/worker failures,
stop and join children, then clean up mappings and files. Benchmark timeouts
must stop the worker process tree, not only its coordinator. No retries,
recovery service or daemon supervision framework.

## Memory and implementation sequence

For S sites, A eligible ancestors, P packed calls and W workers, large shared
input costs P bytes plus the sizes of live sample slabs. Each slab costs its
selected haplotype count times S bytes. Add one decoded tile per active
preparation worker/loader, remapping temporaries, worker-local O(A + S) catalog
and scratch, and private native indexes/traceback. Final tables scale with
output; this is not a bounded-memory tree-sequence writer. Do not count shared
mapped pages W times when reporting memory.

Implement in this order:

1. Scalar interface and pure array kernels; establish greedy semantics without
   masks. Check independently of storage and scheduling.
2. Packed reference and selected-column slab loader, with validation at input
   boundaries and explicit ownership.
3. Direct single-haplotype integration with existing gap/output semantics.
4. Persistent process scheduling and completion-order output using the same
   leaf function, including failure cleanup.
5. Remove obsolete pipeline branches/products and simplify benchmark entry
   points; document the remaining config options concisely.
6. Measure the deployed runner before adding further optimisation.

Keep biological settings, worker/helper allocation, `sample_prefetch_chunks`,
`max_pending_haplotypes`, temporary directory and optional path output. Remove
`haplotype_batch_size` as synchronisation. Performance worker-count/repeat loops
remain useful; they are not biological hyperparameter machinery.

## Verification and performance decision

Use focused comparison scripts and notebook cases, not a new test framework.
Compare greedy component bounds and parent runs for empty inputs, ties,
containment, touching intervals, gaps and thresholds below/equal/above union
size, including a rejected component before a retained one. An independent
small site-wise minimum-run calculation avoids relying only on notebook
agreement. Check unsorted catalog order as well as sorted examples.

Compare direct endpoints with current zero-mismatch intervals, then complete
paths, mutations, called-site reconstruction and sample/node metadata. Include
missing calls, polarisation, shared seeds, no eligible ancestors and supports
crossing physical chunks. Verify one/multiple workers agree, each required
physical genotype block is requested once, and optional output changes nothing.
Exercise ordinary failure cleanup and result ownership across successive jobs.

Make end-to-end time from existing ancestor trees to finished outputs the
primary comparison with current greedy and native full HMM. Match inputs,
biological/HMM settings, CPU allocation and output requirements. Use the 30 Mb
case, then longer sequences and larger cohorts; state filesystem/JIT cache
conditions and record the actual dependency revisions.

Define new resident timing explicitly: decoded sample/reference inputs and
worker HMM/JIT setup are outside timing; candidate discovery, exact extension,
greedy covering and gap matching are inside. The old resident greedy timing
excluded precomputed candidate/interval preparation, so it is not an equal-work
speedup baseline for this new window. Keep historical numbers labelled rather
than adding a second production algorithm just to reproduce that boundary.

Record aggregate total wall/process-tree CPU and working memory, setup, reference
packing, sample loading/remapping, per-haplotype numeric matching, gap/assembly
work and physical block counts. Use diagnostic runs for detailed attribution;
do not put a timer around every short HMM call in headline runs. Native indexes
are private per process, and mapped inputs can fault or thrash.

The supplied plan reports roughly 298-328 MiB of packed ancestor spans versus
9.15-13.52 GiB dense, and a 30 Mb greedy advantage only in its old resident
window, not end-to-end. These motivate packing and pipeline cleanup, not a
claimed speedup for this implementation. The underlying local datasets,
measurement CSVs and meeting notes were not independently rerun/read here.

Stop at a correct, readable baseline with measured end-to-end behaviour.
Defer PBWT/tree candidate retrieval, positive mismatch budgets, adaptive AC,
coarse fallback references, boundary-conditioned HMMs, per-gap parallelism,
persistent caches, automatic input rechunking and specialised sort schemes.

## Review basis

This revision uses the supplied plan and the publicly accessible
`tsinfer-match-eval` matching/loading/benchmark code, `Snakefile`, `AGENTS.md`,
and `dphil-analysis/notebooks/ch5_stitching__scratch.py` (attempt 2). Public
native-reader code and Python/Numba documentation informed the implementation
clarifications. The exact local editable tsinfer revision remains an integration
prerequisite, not a verified part of this review.
