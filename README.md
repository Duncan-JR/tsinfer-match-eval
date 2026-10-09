# tsinfer matching evaluation

A Snakemake pipeline that builds inferred ancestor panels, matches ancestors and
samples separately against the full reference, and generates focal candidates
and direct focal-ancestor intervals. It also matches samples against reduced
focal references and stitches full-reference gap matches into those paths.
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
Both exclude booleans. `focal_methods` currently accepts only `[hmm]`.
For example:

```yaml
ac_cutoff: [3, 5, 10]
max_mismatches: [0, 1]
focal_methods: [hmm]
```

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

Optional dataset `include` and `exclude` strings pass unchanged into native
TOML. Omitted or null values are omitted; native validation treats them as
alternatives. For example, `include: 'POS >= 1000000 & POS < 2000000'` and
`exclude: 'POS < 1000000 | POS >= 2000000'` select the same interval. Coordinates
remain absolute. `sample_list` is a headerless CSV of sample IDs, for example
`data/sample_lists/tgp_chr20_n100.csv`. The pipeline joins the IDs into tsinfer's
native `samples` string; omission or null selects all samples. Count annotation,
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

For H haplotypes, P eligible associations, and B = max_mismatches + 1, the
interval NPZ contains:

| Field | Shape | Meaning |
| --- | --- | --- |
| `sample_id` | `(H,)` | Unicode IDs in candidate haplotype order |
| `ploidy_index` | `(H,)` | int64 chromosome index within each sample |
| `offsets` | `(H + 1,)` | int64 association boundaries |
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
Ancestor IDs and seeds are available in the candidate archive rather than
repeated in the interval archive. Lower cutoffs select `focal_ac <= cutoff`
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
stores stream site blocks while retaining sweep state. Spawned workers reuse
prepared numeric associations and open their own read-only panel. A queued
multiprocessing Pool assembles results in canonical order. No entire ancestor
genotype matrix is loaded. Coverage summaries are outside this pipeline.

## Focal masks and stitched matching

`lib.haplotypes.make_focal_mask` selects associations with `focal_ac <= ac_cutoff`
and interval column k. It sorts and unions overlapping or touching bounds,
then emits covered runs and their complement. Ordered half-open runs partition
the inference-site axis; `is_gap=True` denotes zero coverage. A haplotype with
no eligible associations has one all-gap run.

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
haplotype once with tsinfer's scheduled VCZ reader and allele mapper. A queued
multiprocessing Pool owns one full-reference matcher per worker. Each task
processes all k values; a single-worker allocation uses the same function
directly. The dataset's `matching_cache_size` also applies here.

For a haplotype with focal coverage, panel candidate IDs are joined to reference
nodes through `(source, sample_id, ploidy_index)` metadata for source `ancestors`
and ploidy 0. Eligible candidates plus roots 0 and 1 are retained by one
site-preserving simplification. Internal ancestors remain copying states.
One reduced-reference HMM uses the entire observed inference-site axis,
including gaps, and its parent IDs are mapped back to the original reference.
This match is reused across every budget. Haplotypes without focal coverage
skip the reduced pass.

For each k, the saved mask alone decides which reduced parent/copied-allele
sites survive. Gap slices are overwritten by independent full-reference HMMs,
split at inference-interval boundaries from reference metadata. Missing calls
are neutral observations. There is no padding or boundary conditioning. The
output domain runs from the first called site through the last called site,
or over the entire axis for an all-missing haplotype, as in baseline matching.

Adjacent equal parents merge into native path segments, including across
seams. Internal boundaries use site positions; outer bounds are BP 0 and the
sequence length when the domain reaches the axis endpoints. Mutations are
derived from called observations differing from the final copied alleles.
They include retained focal mismatches and exclude discarded reduced gap
mismatches and missing observations.

Each output uses one native `extend_ts` call against the original ancestor
reference. It remains raw and unsimplified, preserves original reference node
IDs, and adds one sample node per haplotype with native individual/population
assignments. JSONL uses the baseline fields below plus `ac_cutoff`,
`max_mismatches`, and `method`, ordered by native `haplotype_index`.
Only focal `method="hmm"` is implemented. Greedy matching, boundary
conditioning, engine optimisation, and analysis remain later work.

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
matching follows [plans/stitched_matching.md](plans/stitched_matching.md).

Code checks are `uv run ruff check lib` and `uv run ruff format --check lib`.
Manual verification uses small temporary products: direct interval unions,
all-gap equivalence to native HMM matching, reduced-pass reuse across budgets,
overlapping focal parent agreement, TS/JSONL path and called-allele agreement,
and the coarse Snakemake DAG. No test suite is included; full-cohort runs and
analysis are not part of automatic verification.

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
Six masks agreed with direct unions, including touching/overlapping bounds and
all-gap rows; interval maxima were AC 10 and k 1. All-gap paths and mutations
agreed with native full-HMM matching for trimmed and all-missing domains.
Covered haplotypes made one reduced call across both budgets, and parents
agreed on overlapping focal coverage. Saved TS/JSONL agreed on paths and
mutations, reproduced called alleles, and included real seam switches and
retained focal mutations. One- and two-worker outputs agreed; gap calls split
at metadata interval boundaries. Actual Snakemake execution produced six masks
and six TS/JSONL pairs in three stitched jobs. Dry runs confirmed the coarse
DAG and absence of a baseline sample-match dependency; configuration validation
also passed. No full cohort or performance benchmark was run.
