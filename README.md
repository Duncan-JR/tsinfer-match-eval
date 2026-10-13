# tsinfer matching evaluation

A Snakemake pipeline that builds inferred ancestor panels, matches ancestors and
samples separately against the full reference, and generates focal candidates
and direct focal-ancestor intervals. It constructs focal paths using either a
reduced-reference HMM or an exact greedy interval cover, then stitches
full-reference gap matches into those paths.
It uses an editable `../tsinfer` checkout
and calls functions in `lib` directly. True-panel extraction and matching remain
available through explicit targets, outside the default workflow.

## Run

From this repository, with the sibling tsinfer checkout and configured inputs:

```sh
cp config.yaml.example config.yaml
uv sync
uv run snakemake --cores all
```

After this focal refactor, regenerate candidates and intervals:

```sh
uv run snakemake --cores all --forcerun find_focal_ancestors make_ancestor_intervals
```

Existing generated products are not migrated or automatically deleted. The new
candidate archive and selected-seed semantics require regeneration. Statistics
CSVs and population enrichment have been removed; readers of those products or
previous focal archives must be updated separately.

Library files are intentionally not Snakemake inputs. After changing focal
lookup, force `find_focal_ancestors`; its consumers regenerate too. After changing
interval comparisons, force `make_ancestor_intervals`. Rebuild native
configurations and downstream ancestors and matches after changing HMM settings.
After changing mask unions, force `make_focal_mask`; after changing stitched
matching, force `match_stitched_samples`. To refresh the dormant true CSV,
explicitly target it and force
`extract_true_ancestors`.

Edit `config.yaml` to choose datasets and output folders. Paths are relative to
this working directory; `~` is expanded for input paths. Each dataset supplies
`name`, `zarr_path`, and `ancestral_state`. `ts_path` is used only for explicit
true-panel extraction and may be null. Ancestral state can be
`{field: variant_ancestral_state}` or `{is_reference: true}`. Inputs are phased,
single-contig VCZs with optional native site and sample selection.

`focal_choice` is a top-level option: `left` selects the smallest non-padding
focal position and `right` selects the largest. Omission means `left`; only
these exact strings are accepted, including rejection of null. The Snakefile
validates it before constructing rules or launching jobs. Direct library callers
supply a valid choice themselves.

`ac_cutoff` is a nonempty, strictly increasing list of positive integers.
`max_mismatches` is a nonempty, strictly increasing list of nonnegative integers.
Both exclude booleans. `focal_methods` accepts `hmm`, `greedy`, or both, without
duplicates. The greedy method requires `max_mismatches: [0]`.
For example:

```yaml
ac_cutoff: [3, 5, 10]
max_mismatches: [0, 1]
focal_methods: [hmm]
```

To compare the exact greedy cover with the reduced HMM, use
`max_mismatches: [0]` and `focal_methods: [hmm, greedy]`. Regenerate interval
archives with `--forcerun make_ancestor_intervals` to save ancestor identities
before using greedy matching. Existing archives are not migrated.

Only the numeric maxima affect interval generation: this example uses inclusive
AC 10 and generates budget columns 0 and 1. Smaller list members select masks
and stitched outputs without repeating interval comparisons. The mismatch
budget applies independently in each direction from the seed and controls
coverage; it is not an HMM mismatch limit. Changes to either maximum regenerate
intervals and their consumers.
Changing `focal_choice` regenerates candidates and intervals while leaving
inferred ancestors and full matches current. `inference_threads` and
`matching_threads` request workers for those stages; interval generation uses
its allocated cores, capped by physical ancestor-column blocks. Stitched jobs
reserve the workflow's available cores and cap workers by haplotype count.

The `hmm.recombination` and `hmm.mismatch` values are scalar per-site
probabilities written explicitly for both matching stages; both must be strictly
between zero and one. Candidates, intervals, and masks do not depend on
matching outputs. Stitched matching needs the ancestor reference but has no
dependency on baseline sample matches.

For every dataset `{name}`, the default workflow requires:

| Product | Path under `{data_dir}` |
| --- | --- |
| Inferred ancestor panel | `ancestors/{name}_inferred_ancestors.zarr/` |
| Inferred ancestor reference | `ancestors/{name}_inferred_ancestors.trees` |
| Focal candidates | `focal_ancestors/{name}_inferred_focal_ancestors.npz` |
| Full-reference raw sample trees | `matches/{name}_inferred_samples_raw.trees` |
| Full-reference sample match records | `matches/{name}_inferred_samples_matches.jsonl` |
| Direct focal intervals | `haplotype_intervals/{name}_inferred_focal_ancestor_intervals.npz` |
| Focal coverage mask, each AC/k | `focal_masks/{name}_inferred_ac{ac}_k{k}_focal_mask.npz` |
| Raw stitched samples, each AC/k/method | `matches/{name}_inferred_ac{ac}_k{k}_{method}_stitched_samples_raw.trees` |
| Stitched match records, each AC/k/method | `matches/{name}_inferred_ac{ac}_k{k}_{method}_stitched_samples_matches.jsonl` |

The example produces six masks and six stitched TS/JSONL pairs per dataset.
One mask Snakemake job per dataset loads the interval archive once and writes
every requested AC/budget mask.
One stitched Snakemake job per dataset/AC/method produces every requested budget.
There are no per-haplotype or per-region jobs or intermediate matching files.

Intermediate products include `samples/{name}_samples_masked.zarr/` and
`configs/{name}_ancestor_inference.toml`. Logs go under `progress_dir`. The
historical sample-store name is retained: it is a regular input copy with the
original variant and sample axes and no singleton mask. It adds exact
`variant_match_eval_derived_ac` and `variant_match_eval_derived_af`, computed
from the selected cohort in genotype chunks. Missing calls enter neither the
derived count nor the called-genome denominator. Derived AF remains available
for dormant true-panel extraction.

Generated native TOML excludes selected-cohort singletons using
`INFO/match_eval_derived_ac == 1`. Dataset `include` expressions are combined
with the non-singleton condition using `&`; `exclude` expressions are combined
with the singleton condition using `|`. Without either dataset filter, only
singleton exclusion is added. Native validation treats `include` and `exclude`
as alternatives. For example, `include: 'POS >= 1000000 & POS < 2000000'` and
`exclude: 'POS < 1000000 | POS >= 2000000'` select the same interval, excluding
singletons. Original sample calls remain available in the copied store.
Coordinates remain absolute. `sample_list` is a headerless CSV of sample IDs,
for example `data/sample_lists/tgp_chr20_n100.csv`. The pipeline joins these IDs
into the native `samples` string; omission or null selects all samples. Count annotation,
native stages, and focal readers use the same cohort. Generated ancestors do
not receive the participant string.

Optional per-dataset `matching_cache_size` passes the native cache size in MiB
to both matching stages; omission or null uses its default. The 10 Mbp real-data
example uses 1024 MiB because its native source chunk needs 501.2 MiB. Changing
site filters rebuilds configurations and downstream products; changing samples
also refreshes counts and focal products. The real chr20 input retains a
multi-contig header: native tsinfer resolves chr20 and its 64,444,167-base length.

## Focal candidates

`lib.haplotypes.find_focal_ancestors` opens samples and the inferred panel and
uses a shared loader to map panel positions into the sample store. Called sample
allele strings equal to the panel's ancestral allele (`variant_allele[:, 0]`)
become 0, other called alleles become 1, and missing calls remain -1. This also
handles ancestry reversed relative to sample REF/ALT order.
Seeds are selected first; lookup loads and polarises calls only at unique
selected seed sites rather than loading the full inference-site call matrix.

An ancestor is a candidate for a haplotype exactly when that haplotype's call at
its selected endpoint seed is derived. Calls at all other focal sites do not
affect membership. Ancestors with no non-padding focal position have no seed and
contribute no candidates or intervals. Both chromosome copies are retained,
including haplotypes with empty candidate sets. These associations describe
allele sharing, including at recurrent sites. Baseline matching uses the full
panel independently of focal lookup; stitched matching uses these candidates.

Load either focal product with `numpy.load(path, allow_pickle=False)`. For H
haplotypes, A panel ancestors, and P candidate associations, the candidate NPZ
contains exactly:

| Field | Shape | Meaning |
| --- | --- | --- |
| `sample_id` | `(H,)` | Unicode IDs in sample-major haplotype order |
| `ploidy_index` | `(H,)` | int64 chromosome indices increasing within each sample |
| `ancestor_id` | `(A,)` | Unicode panel IDs in panel-column order |
| `offsets` | `(H + 1,)` | int64 ragged boundaries |
| `ancestor_index` | `(P,)` | int64 sorted candidate columns per haplotype |
| `derived_ac` | `(A,)` | int64 exact selected-seed AC, zero for unanchored ancestors |
| `focal_site_index` | `(A,)` | int64 selected panel-site seeds, -1 for unanchored ancestors |
| `focal_choice` | scalar | Unicode endpoint choice |

Row i uses `ancestor_index[offsets[i]:offsets[i + 1]]`. Equal offsets preserve
empty sets. Candidate indices refer to panel columns, never TS node IDs. Lookup
does not filter by AC. Saved seeds define both membership and interval extension;
`focal_choice` records provenance only and is not checked against configuration
when reading an archive.

## Direct focal-ancestor intervals

`lib.haplotypes.make_ancestor_intervals` compares selected sample
haplotypes directly with inferred ancestors eligible at the maximum AC. To run
only this branch, explicitly target an interval archive:

```sh
uv run snakemake --cores all data/match_eval/haplotype_intervals/out_of_africa_n300_1mbp_inferred_focal_ancestor_intervals.npz
```

The rule needs annotated samples, inferred ancestors, and focal candidates. It
requires no reference trees or sample matches. Preparation checks archive/store
identity and order once, filters candidates by `derived_ac <= max_ac_cutoff`,
and reuses their saved seeds. No seed is selected again. Neither endpoint is
assumed to produce the same tract or candidate set as the other.

For H haplotypes, A panel ancestors, P eligible associations, and
B = max_mismatches + 1, the
interval NPZ contains:

| Field | Shape | Meaning |
| --- | --- | --- |
| `sample_id` | `(H,)` | Unicode IDs in candidate haplotype order |
| `ploidy_index` | `(H,)` | int64 chromosome index within each sample |
| `offsets` | `(H + 1,)` | int64 association boundaries |
| `ancestor_id` | `(A,)` | Unicode panel IDs, stored once per ancestor |
| `ancestor_index` | `(P,)` | int64 panel column for each association |
| `focal_ac` | `(P,)` | int64 exact selected-seed AC per association |
| `left_site_index` | `(P, B)` | int64 inclusive left bounds, column k for budget k |
| `right_site_index` | `(P, B)` | int64 exclusive right bounds, column k for budget k |
| `num_sites` | scalar | int64 panel-axis length |
| `max_ac_cutoff` | scalar | int64 inclusive generation maximum AC |
| `max_mismatches` | scalar | int64 largest directional budget |
| `focal_choice` | scalar | Unicode endpoint choice |

Associations follow haplotype order then increasing ancestor column. Distinct
ancestors with identical bounds remain separate. Empty slices and both ploidy
copies remain in the roster; an entirely empty result has bounds `(0, B)`.
Association i has identity `ancestor_id[ancestor_index[i]]`; these indices are
panel columns, never TS node IDs. This dictionary avoids repeating strings for
every association. Seeds remain in the candidate archive.
Lower cutoffs select `focal_ac <= cutoff`
without repeating comparisons.

Budget k permits k mismatches independently on each side of the matching seed;
the next mismatch is excluded. Missing sample or ancestor calls stop extension
regardless of budget. Bounds are half-open on the panel site axis, clipped to
ancestor support intersected with the inference interval containing the seed.
For support [0, 10), seed 4, left mismatches 3 and 1, and right mismatches 6 and 8,
budgets 0, 1, 2 retain [4, 6), [2, 8), [0, 10). Larger budgets preserve earlier
columns and produce nested intervals.

The Numba directional sweep reuses physical ancestor-column blocks across
sample associations. A full-height block serves both directions; multi-height
stores stream site blocks while retaining sweep state. A thread pool shares
the read-only panel, sample calls and numeric associations. Each block owns its
sweep state and outputs; Numba releases the GIL during the sweep. Results are
assembled in canonical order. No entire ancestor genotype matrix is loaded.
Coverage summaries are outside this pipeline.

## Focal masks and stitched matching

`lib.haplotypes.make_focal_masks` loads the interval archive once for all requested
cutoffs and budgets. Each mask selects associations with `focal_ac <= ac_cutoff`
and interval column k. It sorts and unions overlapping or touching bounds,
then emits covered runs and their complement. Ordered half-open runs partition
the inference-site axis; `is_gap=True` denotes zero coverage. A haplotype with
no eligible associations has one all-gap run.
`make_focal_mask` also supports generating a single output directly.

The compressed mask NPZ uses numeric/Unicode arrays and contains:

| Field | Meaning |
| --- | --- |
| `sample_id`, `ploidy_index` | Original haplotype roster |
| `offsets` | Haplotype boundaries in flattened run arrays |
| `left_site_index`, `right_site_index` | Half-open site bounds |
| `is_gap` | Boolean gap indicator |
| `num_sites` | Inference-site axis length |
| `ac_cutoff`, `max_mismatches`, `focal_choice` | Selected parameters and seed provenance |

`lib.matching.match_stitched_samples` builds native sample jobs and reads each
haplotype once inside its task with tsinfer's scheduled VCZ reader and allele
mapper. Threads share the reference, full-reference indexes, focal archives,
masks and reader. Each thread owns a reusable full-reference matcher, and each
task processes all k values. A single-thread allocation uses the same function
directly. The dataset's `matching_cache_size` also applies here. Simplification
holds the GIL, so reduced-reference preparation does not scale with matching
threads; native HMM calls release the GIL and can run concurrently.

`haplotype_batch_size` is a required positive integer (configured as 64). Set it
at least as high as the allocated stitched worker count to keep workers occupied
when enough haplotypes are available. It bounds the contiguous canonical jobs
submitted and their Python results. One executor, thread-local full matchers,
reader schedule, cache, indexes and input archives survive across batches.
The parameter does not change candidate generation or reference construction.

For a haplotype with focal coverage, panel candidate IDs are joined to reference
nodes through `(source, sample_id, ploidy_index)` metadata for source `ancestors`
and ploidy 0. Eligible candidates plus roots 0 and 1 are retained by one
site-preserving simplification. Internal ancestors remain copying states.
One transient reduced matcher is reused across every budget, and copied parent
IDs are mapped back to the original reference. Haplotypes without focal coverage
in the called-site domain skip reduction.

The full and reduced matchers retain two independent reference traversals.
Ordered regional calls advance the selected matcher's topology through skipped
spans without running an HMM there, then initialise fresh likelihoods and
traceback for the requested region. Traceback preserves the retained endpoint
tree. Both traversals reset at every haplotype/budget boundary without
reallocating the matchers. Each thread also reuses one full-site matched-allele
scratch array; only the requested slice is read after a call. Greedy focal spans
need no native call: the full matcher advances through them before its next gap.

For `method="hmm"`, each maximal focal run receives an independent reduced-reference
HMM match and each gap an independent full-reference match. Both kinds of run
are clipped to the called-site domain and split at inference-interval boundaries
from reference metadata. Paths are recomputed for each budget's components:
clipping a larger component's match would retain its different endpoint
optimisation. Missing calls are neutral observations. There is no padding or
boundary conditioning. The
output domain runs from the first called site through the last called site,
or over the entire axis for an all-missing haplotype, as in baseline matching.

Regional HMM paths are collected as numeric `CopyingRuns` on the inference-site
axis. Called mismatches become owned mutations immediately, before the thread's
matched-allele scratch is reused. Greedy focal runs are computed once per
haplotype. The ordered blocks are concatenated once and coalesced in a linear
pass, then converted to native BP path segments without a per-site parent or
copied-allele array.

Adjacent equal parents merge into native path segments, including across
seams. Internal boundaries use site positions; outer bounds are BP 0 and the
sequence length when the domain reaches the axis endpoints. Mutations are
derived from called observations differing from the final copied alleles.
They include focal and gap mismatches and exclude missing observations.

Each budget starts with one table collection dumped from the original ancestor
reference. After matching, each batch is sorted by native `haplotype_index`,
appended with native `append_match_results`, and written to its already-open
JSONL file. The batch's futures and results are released before admitting the
next batch. Each budget's tables are finalised once with `finalise_match_tables`
and dumped, then released before finalising the next budget. Every match still
uses the original immutable reference. Output remains raw and unsimplified,
preserves original reference node IDs, and adds one sample node per haplotype
with native individual/population
assignments. JSONL uses the baseline fields below plus `ac_cutoff`,
`max_mismatches`, and `method`, ordered by native `haplotype_index`.
For `method="greedy"`, ancestor identities and bounds come directly from the
interval archive; the candidate archive and reduced reference are not loaded.
For each focal component, start at its first uncovered site and choose an
eligible interval containing that site with the furthest exclusive right bound.
Copy until that bound and repeat, accepting touching intervals. Equal reaches
keep the first interval in stable left-bound order. The sweep visits each
eligible interval once after sorting, costs O(M log M + R) per haplotype, and
uses O(M + R) working memory for M intervals and R selected runs. Numeric output
capacity is bounded by eligible intervals; only compact owned runs survive. Numba
releases the GIL. The first call also incurs compilation overhead.

Working Python match results scale with batch size. Final tables necessarily
scale with cohort output and budget count. Input masks/associations, the canonical
job list and schedule, and retained reader chunks remain cohort/reference-sized;
batching does not bound total preparation memory.

This is an optimal minimum-switch cover of the **saved exact intervals**.
With uniform initial state probabilities and constant recombination
probabilities, all exact paths have the same emission likelihood and each
additional switch multiplies the likelihood by
`(rho / n) / (1 - rho + rho / n)`, so minimising switches maximises likelihood
within that restricted state space. The reduced HMM can also copy retained
internal nodes and matching tracts outside the saved focal intervals; its
minimum can therefore be smaller. Positive HMM mismatch probability can favour
a mismatch over sufficiently many switches even when an exact cover exists.
Different parents or switch positions can also be equally optimal.

Greedy matches stop at each focal component's end; full-reference HMM calls
still resolve gaps, and native assembly merges equal parents across seams.
Inference boundaries are respected because saved intervals cannot cross them.
There is no endpoint conditioning, so minimising focal switches does not
minimise total switches across gap seams. Mismatch-tolerant intervals require
emission costs and are rejected by this first greedy implementation.
Boundary conditioning, sparse node resets, cross-sample reference caching, and finer
region scheduling remain later work. Regional matching relies on the tsinfer
changes in the efficient stitched plan in the local tsinfer checkout. Later
ordered calls avoid replaying edge prefixes and rebuilding traceback endpoints
from reference suffixes. Edge updates through skipped spans, dense probability
resets, and a node-sized traceback parent copy remain; simplification still
holds the GIL. Native path buffers are sized to the requested region.

## Full-reference matching and dormant true panels

Ancestor matching keeps both configured sources to retain the full sample
contig length and stops before the first sample group. Sample matching uses the
unmodified reference and retains the sample source's name, filters, order, and
individual metadata. Root nodes, individual rows, and node metadata survive the
handoff. Matching disables path compression and emits raw trees without
post-processing, simplification, augmentation, or checkpoint directories.

Sample match JSONL contains one record per haplotype with `group`,
`haplotype_index`, `source`, `sample_id`, `ploidy_index`, `time`, `path`
(`left`, `right`, `parent`), and `mutations` (`position`, `derived_state`).
Coordinates are absolute and half-open. Mutation states are canonical allele
indices; TS states are allele strings. Join by `(source, sample_id,
ploidy_index)` because threaded completion order may vary. Parent IDs are raw
TS node IDs; node metadata joins them to panel `sample_id` strings. Each
invocation writes JSONL from scratch.

Explicit true targets include `ancestors/{name}_true_ancestors.zarr/`,
`dataframes/{name}_true_ancestors.csv`, `configs/{name}_true_ancestors.toml`, and
the shared ancestor/sample matching outputs with `true` in place of `inferred`.
True focal lookup is no longer available.

True extraction uses the inferred site axis, selecting one derived mutation per
site. At recurrent sites it chooses the event whose node covers the most
present-day derived carriers, breaking ties by mutation order. Sites selecting
the same positive-time node share one panel column and multiple focal positions.
The CSV retains every inference site; zero-time origins have
`true_mutation_is_singleton=True` and a blank ancestor ID. Final truth node
haplotypes are decoded in chunks, preserving missing calls and original times.
The CSV's `true_node_id` identifies the original truth TS; string ancestor IDs
identify their panel columns. Sample matching introduces terminal mutations in
the final raw sample trees. Ancestor groups must precede sample groups, without
retiming.

Historical extraction and matching details are in
[plans/initial_mvp.md](plans/initial_mvp.md) and
[plans/matching_setup.md](plans/matching_setup.md). The selected-seed contracts
and verification requirements are in
[plans/refactor_focal_ancestors.md](plans/refactor_focal_ancestors.md), which
supersedes earlier focal-selection and statistics instructions. Stitched
matching follows [plans/fast_stitched_matching.md](plans/fast_stitched_matching.md),
which supersedes the whole-genome reduced pass in
[plans/stitched_matching.md](plans/stitched_matching.md).

Code checks are `uv run ruff check lib` and `uv run ruff format --check lib`.
Manual verification uses small temporary products: direct interval unions,
all-gap equivalence to native HMM matching, reduced-reference reuse across budgets,
independent focal controls, serial/threaded agreement, regional observation
independence, TS/JSONL path and called-allele agreement, and the coarse Snakemake
DAG. No test suite is included; full-cohort runs and
analysis are not part of automatic verification.

Retained-traversal verification compared saved independent matches exactly for
both methods with one and two workers: HMM at k 0/1 and greedy at k 0. It used
changing-tree synthetic regions, the 12-site synthetic VCZ workload, and four
haplotypes from each existing 10 Mbp simulation and TGP panel. Scientific tables
and canonical JSONL agreed. Regional call counts matched clipped mask runs
after inference-boundary splitting, including leading/trailing gaps, alternating
coverage, empty focal coverage, trimmed calls, and all-missing haplotypes.
Consecutive haplotypes on one worker and both HMM budgets exercised traversal
resets; the worker and full matcher remained allocated across tasks.

Run and batch verification repeated these saved comparisons with one batch and
batches of two, using one and two workers. Synthetic batches of one and three
also checked a partial final batch. Scientific tables and JSONL remained exact;
called-allele reconstruction, original node IDs, metadata and individual/population
assignments were checked on all three workloads. Manual seam checks covered
touching equal parents, changed parents, outer BP endpoints and single-site
regions, plus greedy equal-reach ties and regional scratch ownership. Weak
references confirmed that four batches of four on a 16-haplotype workload
released every batch's results and futures before the next batch, while reusing
worker, matcher and scratch identities. Invalid batch sizes were rejected and
worker failures propagated with reader shutdown. Forced two-worker Snakemake
stitched jobs with batch size two reproduced the saved HMM k 0/1 and greedy k 0
outputs from prepared synthetic inputs. Diagnostics and saved products stayed
outside production source; no test framework was added.

A complete-run diagnostic on 13 October 2026 used a fixed 12,000-site, 16-haplotype
synthetic reference with 1,200 short mask runs per haplotype and changing trees.
The saved implementation and current runners used identical inputs, AC cutoff 3,
inference intervals, reduced-node selection, HMM parameters, worker allocations,
and a 64 MiB reader cache, against tsinfer revision `59a078b`. The current batch
size was four. Each cell below is the median of three isolated process runs;
RSS is the median process high-water mark measured with macOS `getrusage`, not
the matcher's incomplete `total_memory` estimate. Elapsed time includes reference
and archive loading, reader/job setup, simplification, matching, final table
assembly, JSONL and TS writing. Interpreter imports and greedy sweep compilation
were excluded from elapsed time; process RSS includes them. Native job-grouping
compilation remains included in setup for both implementations.

| Method / budgets | Workers | Saved seconds | Current seconds | Saved peak RSS MiB | Current peak RSS MiB |
| --- | --- | --- | --- | --- | --- |
| Reduced HMM / 0 | 1 | 4.747 | 4.370 | 249.1 | 254.7 |
| Reduced HMM / 0 | 2 | 4.407 | 4.545 | 258.3 | 263.2 |
| Greedy / 0 | 1 | 1.860 | 1.651 | 253.5 | 251.3 |
| Greedy / 0 | 2 | 1.692 | 1.759 | 253.7 | 251.9 |
| Reduced HMM / 0, 1 | 1 | 5.461 | 4.467 | 249.2 | 255.6 |
| Reduced HMM / 0, 1 | 2 | 4.860 | 4.896 | 258.3 | 263.4 |

The unchanged full-HMM runner, with reuse disabled on the same reference and
cache budget, took 1.515/1.512 s and 243.9/244.0 MiB with one/two workers.
Scientific tables and canonical records were checked exactly across repeats;
full-HMM records were sorted for comparison because their existing threaded
writer emits completion order.

Separate one-worker profiles of HMM k 0/1 recorded 38,400 regional calls in both
implementations. Regional wrapper time fell from 1.519 to 0.637 s, including
native `find_path` time falling from 1.102 to 0.051 s. Simplification still took
2.356/2.503 s in the saved/current profiles; table finalisation took 0.015 s in
each. Greedy's 9,600 regional calls fell from 0.404 to 0.155 s, while output
coalescing rose from 0.020 to 0.074 s and current batch assembly/JSONL took
0.099 s. These instrumented timings are diagnostic, separate from the medians.
Native control-flow inspection and the existing cursor assertions establish
forward-only event consumption after each cold start and scratch-parent
traceback without prefix replay or suffix reconstruction. The Python profile
measures the resulting call cost, not individual C edge events.

This small workload shows serial improvements, slight two-worker regressions,
and no general reduction in total RSS. Simplification, dense node resets/copies,
Python run collection/coalescing, shared input preparation and reference/output
tables remain costs. Bounded result lifetimes do not imply bounded total RSS or
good thread scaling, and these synthetic measurements do not establish
whole-cohort performance on biological data.

The refactor verification checked both choices on the existing
`out_of_africa_n100_1mbp`, `out_of_africa_n300_1mbp`, and
`tgp_chr20_n100_1mbp` panels. Each choice retained 114,035, 325,711, and 205,293
associations respectively, with every K=0/1/2 bound checked by independent
outward scans. One-worker and three-worker archives agreed. Temporary synthetic
stores covered different endpoint memberships, missing calls, reversed REF/ALT
ancestry, unanchored ancestors, empty rosters and slices, distinct ancestors with
identical bounds, and four physical site-block heights. Temporary workflow
execution and dry-runs checked the six default outputs, interval independence
from matching, configuration rejection before DAG construction, omitted choice
equivalence to `left`, and choice/AC/K parameter invalidation.

Stitched verification used temporary inputs with 12 inference sites, two
diploid samples, and three ancestors, including a retained internal ancestor.
Six batched masks reproduced the previous arrays with one interval load.
Independent fresh focal matches agreed with stitched results; each covered
haplotype prepared one reduced reference across both budgets. All-gap paths and
mutations agreed with native full-HMM matching for trimmed and all-missing
domains. One- and three-thread TS/JSONL outputs agreed and reproduced called
alleles. Both focal and gap calls split at inference-interval boundaries,
including single-site pieces. Candidate and interval arrays also reproduced the
previous implementation on three existing 1 Mbp datasets, both seed choices,
empty seeds/cohorts, reversed ancestry and three physical site-block heights.
Regional calls preserved output sentinels outside their slice and agreed after
changing outside observations and reusing matchers in arbitrary order, including
breakpoint and trailing-gap cases. Actual Snakemake execution produced six masks
in one job and six TS/JSONL pairs in three stitched jobs; a dry run confirmed no
baseline sample-match dependency. Four haplotypes from each existing 10 Mbp
simulation and TGP dataset exactly reproduced the saved independent-focal
diagnostic paths at AC 10/k 0, with zero focal mismatches.

A small timing probe on the local tsinfer extension used the 44,225-site
`out_of_africa_n100_10mbp` reference. Four haplotypes across k 0/1 took 0.045 s
for reduced simplification/index/matcher preparation, 0.038 s for 319 focal
calls, 0.200 s for 324 gap calls, and 0.128 s for path/native assembly. Shared
reference/reader/index preparation took 3.52 s. Separate eight-haplotype runs
including setup and output took 3.97 s with one thread and 3.80 s with four;
outputs agreed, with peak RSS of 341 and 376 MiB respectively. RSS showed no
accumulation across 200 repeated reduced-reference/index/matcher preparations.
These are small-workload measurements, not a full-cohort scaling benchmark.
