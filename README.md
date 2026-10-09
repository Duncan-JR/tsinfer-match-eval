# tsinfer matching evaluation

A Snakemake pipeline that builds inferred ancestor panels, matches ancestors and
samples separately against the full reference, and generates focal candidates
and direct focal-ancestor intervals. It uses an editable `../tsinfer` checkout
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
uv run snakemake --cores all --forcerun find_focal_ancestors construct_focal_ancestor_intervals
```

Existing generated products are not migrated or automatically deleted. The new
candidate archive and selected-seed semantics require regeneration. Statistics
CSVs and population enrichment have been removed; readers of those products or
previous focal archives must be updated separately.

Library files are intentionally not Snakemake inputs. After changing focal
lookup, force `find_focal_ancestors`; its consumers regenerate too. After changing
interval comparisons, force `construct_focal_ancestor_intervals`. Rebuild native
configurations and downstream ancestors and matches after changing HMM settings.
To refresh the dormant true CSV, explicitly target it and force
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

`ac_cutoff` is a nonempty, strictly increasing list of positive integers. Only
its maximum affects interval generation, inclusively. Changing smaller members
with the maximum fixed affects neither focal product. Changing the maximum or
`haplotype_compare.max_mismatches` regenerates intervals only. The latter is a
nonnegative integer and generates budget columns 0 through that maximum.
Changing `focal_choice` regenerates candidates and intervals while leaving
inferred ancestors and full matches current. `inference_threads` and
`matching_threads` request workers for those stages; interval generation uses
its allocated cores, capped by physical ancestor-column blocks.

The `hmm.recombination` and `hmm.mismatch` values are scalar per-site
probabilities written explicitly for both matching stages; both must be strictly
between zero and one. Focal products do not depend on matching outputs.

For every dataset `{name}`, the default workflow requires exactly:

| Product | Path under `{data_dir}` |
| --- | --- |
| Inferred ancestor panel | `ancestors/{name}_inferred_ancestors.zarr/` |
| Inferred ancestor reference | `ancestors/{name}_inferred_ancestors.trees` |
| Focal candidates | `focal_ancestors/{name}_inferred_focal_ancestors.npz` |
| Full-reference raw sample trees | `matches/{name}_inferred_samples_raw.trees` |
| Full-reference sample match records | `matches/{name}_inferred_samples_matches.jsonl` |
| Direct focal intervals | `haplotype_intervals/{name}_inferred_focal_ancestor_intervals.npz` |

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
allele sharing, including at recurrent sites. Matching uses the full panel
independently of focal lookup.

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

`lib.haplotypes.construct_focal_ancestor_intervals` compares selected sample
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
genotype matrix is loaded. Coverage summaries and population labels are outside
this pipeline.

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
supersedes earlier focal-selection and statistics instructions.

Code checks are `uv run ruff check lib` and `uv run ruff format --check lib`.
Manual verification uses temporary products and independent membership and
interval scans, including both endpoint choices, reversed allele order, missing
barriers, support boundaries, empty slices, worker agreement, and configuration
dependencies. No test suite is included. Existing analyses are not changed by
this refactor.

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
