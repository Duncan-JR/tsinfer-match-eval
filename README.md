# tsinfer matching evaluation

A small Snakemake pipeline that builds inferred ancestor panels, matches
ancestors and samples separately, records focal candidates per sample haplotype,
evaluates copying from allele-count eligible focal ancestors, and constructs
direct focal-ancestor intervals without matching. It reads
simulations from `../tsinfer-anc-eval` and uses an editable `../tsinfer`
checkout. The workflow calls functions in `lib` directly. True-panel rules
remain available through explicit output targets, but the default workflow is
inferred-only and never depends on `ts_path`.

## Run

From this repository, with both sibling repositories available:

```sh
cp config.yaml.example config.yaml
uv sync
uv run snakemake --cores all
```

For the one-time count/config/focal format migration, run:

```sh
uv run snakemake --cores all --forcerun annotate_derived_counts write_inference_config find_focal_ancestors compute_focal_ancestor_stats
```

Library files are intentionally not Snakemake inputs. After changing evaluation
code, force `compute_focal_ancestor_stats`. After changing focal lookup, force
`find_focal_ancestors` and the statistics rule. Rebuild configurations,
ancestor/reference TSs, raw sample TSs, match files, and statistics together
after changing HMM settings. Changing smaller cutoffs reruns only the statistics
rule; changing the maximum also regenerates direct intervals. To refresh the
optional true CSV schema, explicitly target its output and force
`extract_true_ancestors`.

Edit `config.yaml` to choose datasets and output folders. Paths in the YAML are
relative to this working directory; `~` is expanded for input paths. Each
`datasets` entry gives a `name`, `zarr_path`, `ts_path`, and `ancestral_state`.
Use `ts_path: null` to evaluate only inferred panels. Ancestral state may be
`{field: variant_ancestral_state}` or `{is_reference: true}`. The input is one
phased, single-contig VCZ per dataset, with optional native sample selection.
`inference_threads` and `matching_threads` set each job's requested worker count;
Snakemake allocates those workers and schedules independent panels in parallel.
`ac_cutoff` is a nonempty, strictly increasing list of positive integers with
inclusive allele-count thresholds. The `hmm.recombination` and `hmm.mismatch`
values are scalar per-site probabilities written explicitly for both ancestor
and sample matching; both must be strictly between zero and one.

The `data_dir` setting contains category folders. For dataset `{name}`:

```text
{data_dir}/samples/{name}_samples_masked.zarr/
{data_dir}/configs/{name}_ancestor_inference.toml
{data_dir}/configs/{name}_true_ancestors.toml   # truth TS only
{data_dir}/ancestors/{name}_inferred_ancestors.zarr/
{data_dir}/ancestors/{name}_true_ancestors.zarr/  # truth TS only
{data_dir}/dataframes/{name}_true_ancestors.csv   # truth TS only
{data_dir}/ancestors/{name}_{kind}_ancestors.trees
{data_dir}/focal_ancestors/{name}_{kind}_focal_ancestors.npz
{data_dir}/matches/{name}_{kind}_samples_raw.trees
{data_dir}/matches/{name}_{kind}_samples_matches.jsonl
{data_dir}/dataframes/{name}_inferred_focal_ancestor_stats.csv
{data_dir}/haplotype_intervals/{name}_inferred_focal_ancestor_intervals.npz
```

The default targets use `{kind}=inferred` for every dataset. Explicit true
targets use `{kind}=true` where truth is supplied. The optional true rules are
`extract_true_ancestors` and `write_true_ancestors_config`; the shared matching
rules remain callable for either kind.

Rule logs go under the separately configured `progress_dir`. The sample store
is a regular copy of the input, retaining its original variant and sample axes.
The historical `_samples_masked.zarr` output name is retained. It contains
`variant_match_eval_derived_ac` and `variant_match_eval_derived_af`, computed
from the selected cohort in genotype chunks. Missing calls enter neither the
derived count nor its called-genome denominator. There is no singleton mask;
tsinfer determines ancestral eligibility, duplicate handling, and inference sites.

Optional dataset `include` and `exclude` strings pass unchanged into native TOML.
Omitted or null values are omitted. Native validation treats include
and exclude as alternatives. For example, `include: 'POS >= 1000000 & POS < 2000000'`
and exclude-only `exclude: 'POS < 1000000 | POS >= 2000000'` select the same interval.
Coordinates remain absolute. `sample_list` specifies a headerless CSV containing
only sample IDs, for example `sample_list: sample_lists/tgp_chr20_n100.csv` for the
33 CEU, 33 CHB, and 34 YRI cohort. The pipeline joins these IDs into tsinfer's
native `samples` string; omitted or null `sample_list` selects all samples.
All three native stages and local readers use
that same selection; generated ancestors do not receive the participant string.
Optional per-dataset `matching_cache_size` passes the native cache size in MiB
to both matching stages; omitted or null uses the native default. The 10 Mbp
real-data example uses 1024 MiB because its native source chunk needs 501.2 MiB.

Changing site filters rebuilds native configurations and downstream products;
changing samples also refreshes counts, focal candidates, and intervals.

Population enrichment is disabled when `metadata_source` is omitted or null.
`metadata_source: ts` requires the original `ts_path` and joins selected IDs to
`map_to_vcf_model()` names and haploid nodes. It adds `truth_node_id`,
`truth_individual_id`, `population_id`, and `population` to the inferred
statistics CSV.
`metadata_source: csv` requires `csv_path`, `zarr_id_field`, `csv_id_field`, and
`pop_field`, and adds only `population`. CSV identifiers are strings and unique;
selected IDs and labels must exist. Canonical output sample IDs always come from
Zarr `sample_id`. The example uses the pedigree's `superpopulation` labels:
66 EUR, 66 EAS, and 68 AFR genomes. Metadata-only edits rebuild the statistics CSV
without inference or matching; row keys, order, and numerical values are retained.

The real chr20 input retains a multi-contig header. The corrected native tsinfer
checkout resolves chr20 and its 64,444,167-base length in both inference and
matching; this pipeline does not normalize contigs or modify the source store.

The inferred panel's positions define the true panel's site axis. The true
CSV records one selected derived mutation per inference site. A
single-mutation site uses its sole mutation; at a recurrent site, select the
derived-state mutation whose node covers the most present-day derived carriers,
breaking ties by mutation order. Absence of derived carriers is an error.
All sites stay on the shared axis. Rows sort by decreasing original node time,
then inference-site ID. The true panel contains one column per unique selected
positive-time truth node, ordered by its first CSV occurrence. Sites selecting
the same node share `true_ancestor_id` and contribute multiple focal positions
to that column. Final node genotypes are decoded from a
private flags-only TS copy in chunks of `ancestor_chunk_size`, preserving
missing calls and recurrent states.

The site-level CSV columns are:

```text
inference_site_id, true_site_id, focal_position, inferred_ancestor_id,
true_ancestor_id, true_mutation_id, true_mutation_is_singleton, true_node_id, true_node_time,
inferred_node_time, derived_af, derived_ac, num_mutations
```

`derived_ac` is the observed site count on the sample store's inference-site
axis. It is available for both inferred and true IDs in each row and is not a
truth-descendant count.

`num_mutations` retains the original truth site's count. `true_mutation_id`
identifies only the selected event; there are no primary-event or focal-allele
columns. Inferred ancestors use inferred sample times and true ancestors use
original truth node times. Samples remain at time zero. Ancestor matching
requires every ancestor group to precede sample groups; no retiming is applied.

`true_mutation_is_singleton` is `True` when the selected mutation's node has
time zero. These terminal origins have no older panel ancestor, so their
`true_ancestor_id` is blank and focal lookup skips them. Their CSV rows, truth
IDs, original times, and site positions remain present. The flag describes the
selected origin: a recurrent site can have two derived carriers from two
independent singleton mutations. `num_mutations` distinguishes those cases
from ordinary truth singletons promoted by genotype errors. Sample matching
places these mutations in the final raw sample TS, where recurrence is allowed;
the ancestor reference must have at most one mutation per site.

In the CSV, `inferred_ancestor_id` and `true_ancestor_id` are **strings** from
the corresponding Zarr store's `sample_id` array when an association exists.
For instance, `"a17"` may be
at column 17, whose haplotype is `call_genotype[:, 17, 0]`. These IDs are scoped
to their store. Matching writes `source`, `sample_id`, and `ploidy_index`
to node metadata, which supplies the node-to-Zarr join. A matched tree-sequence
node ID is unrelated to its Zarr column index. `true_node_id` refers to the
original simulation TS. The matcher creates its own root nodes; the ancestor
Zarr stores have no synthetic root columns.

Ancestor matching retains both configured sources to obtain the sample VCZ's
full contig length, and stops before the first sample group. Sample matching
uses that unmodified TS as its reference and retains the sample source's name,
order, filters, and individual metadata. Root nodes, pre-created individual rows,
and node metadata survive the handoff. Both stages use the explicit scalar HMM
parameters from `config.yaml`, disable path compression, and produce raw TSs
without post-processing,
simplification, or site augmentation. No checkpoint work directory is used.

Focal lookup polarises sample calls by allele strings and configured ancestral
state on the panel axis. Each haplotype gets the distinct panel columns whose
focal alleles it carries. Multi-focal inferred ancestors can contribute several
associations; true panels use eligible selected CSV events, with several sites
potentially pointing to the same panel column. Singleton-origin rows contribute
no focal association. These
associations describe allele sharing, including at recurrent sites, rather than
actual mutation origin or copying quality. Missing and excluded calls contribute
no candidates. Matching always uses the full panel, independently of focal lookup.

Each compressed NPZ loads with `numpy.load(path, allow_pickle=False)` and contains:

| Array | Meaning |
| --- | --- |
| `sample_id` | Unicode sample ID for each haplotype row |
| `ploidy_index` | Chromosome index, in increasing order within each sample |
| `ancestor_id` | Unicode panel IDs in Zarr column order |
| `offsets` | int64 boundaries, one more entry than haplotype rows |
| `ancestor_index` | int64 concatenation of sorted distinct candidate columns |
| `derived_ac` | int64 observed focal allele count for each inferred panel column |

Row `i` uses `ancestor_index[offsets[i]:offsets[i + 1]]`. Equal offsets preserve
empty candidate sets. Indices refer to panel columns, never TS node IDs.
The panel-level `derived_ac` array is present only in inferred focal files. All
focal sites for one inferred ancestor have the same count.

The native sample match JSONL contains one record per haplotype with `group`,
`haplotype_index`, `source`, `sample_id`, `ploidy_index`, `time`, `path`
(`left`, `right`, `parent`), and `mutations` (`position`, `derived_state`).
Coordinates are absolute and intervals are half-open. Mutation states are
canonical allele indices; TS mutation states are allele strings. No path
likelihood is currently emitted by tsinfer. Join records by
`(source, sample_id, ploidy_index)` because threaded completion order may vary.
Group and haplotype numbering belong to the sample-only invocation. Path parents
are preserved raw TS node IDs; use their metadata to join to panel haplotypes.
Each invocation writes its JSONL from scratch.

The inferred statistics CSV has one row per haplotype and configured cutoff,
in focal-NPZ row order with increasing cutoffs. Its columns are:

```text
dataset, panel_kind, source, sample_id, ploidy_index, ac_cutoff,
evaluated_bp, covered_bp, fraction_covered_bp,
evaluated_sites, covered_sites, fraction_covered_sites,
num_switches, num_mismatches, path_log_likelihood, num_focal_ancestors
```

Evaluation clips the native copying path to the reference TS's
`sequence_intervals`. Every parent, including synthetic roots, contributes to
the base and site denominators. A segment contributes to a numerator when its
parent is in that haplotype's focal candidate set and the parent's
`derived_ac <= ac_cutoff`. Candidate counts are cumulative by the same rule.
Sites are counted directly from reference positions with half-open segment
boundaries.

`num_switches` is the native traceback segment count minus one, and
`num_mismatches` is the number of native mutation records. With mismatch
probability `mu`, recombination probability `rho`, reference node count `n`,
mismatches `m`, and switches `k`, the reported normalized score is:

```text
m * log(mu / (1 - mu))
  + k * log((rho / n) / (1 - rho + rho / n))
```

Here `n` includes the reference's synthetic roots. The score removes the common
match-emission and no-switch baseline; it is not an absolute sequence
likelihood or posterior.

## Checked examples

`uv run ruff check lib` and `uv run ruff format --check lib` are the project code
checks. Manual integration uses zero-error, genotype-error, and `ts_path: null`
n300 examples in separate output folders. No unit-test suite is included.

The matching setup validation used tsinfer commit `a704308` and genotype error
`1.0` (the previous MVP error input). All inferred branches matched successfully:

| Dataset | Inference sites | Inferred ancestors | Raw ancestor nodes | Raw sample nodes | Match records |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero error | 6,120 | 4,639 | 4,641 | 5,241 | 600 |
| Genotype error | 6,063 | 5,443 | 5,445 | 6,045 | 600 |
| No truth | 6,120 | 4,639 | 4,641 | 5,241 | 600 |

All these raw TSs preserve the sample contig length of 64,444,167 and the panel
site axis. Their ancestor references have at most one mutation per site. Match
paths and mutations agree with raw TS nodes, and individual rows and ancestor
metadata survive the split. The zero-error split result equals uninterrupted
matching tables, ignoring provenance.

Both true CSVs retain 6,120 and 6,063 rows respectively and 37 original recurrent
sites each. Their panels contain 3,535 and 3,494 unique older truth nodes.
Selected mutations,
decoded haplotypes, spans, and original node times were checked. All five focal
files have 600 rows and correct distinct candidate sets. Manual probes also
checked missing calls, empty sets, excluded sites, reversed allele order, and
both ancestral-state annotation modes. The no-truth branch requests no true outputs.

The zero-error CSV flags 3 selected singleton origins (positions 185,220,
405,042, and 719,876). Each of those recurrent sites has two independent
terminal mutations. The genotype-error CSV flags the same 3 plus 51 ordinary
truth singleton mutations promoted by erroneous sample calls, for 54 total.
Their ancestor IDs are blank, and no zero-time nodes enter either true panel.
Repeated node selections share one ancestor column: for example, the
selections at 82,649, 84,982, and 85,574 share truth node 27,932 and one
multi-focal panel ancestor, avoiding duplicate mutation introductions during
ancestor matching.

Both true branches now complete ancestor and sample matching:

| Dataset | True panel columns | Flagged CSV rows | Raw ancestor nodes | Raw sample nodes | Sample mutations added |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero error | 3,535 | 3 | 3,537 | 4,137 | 6 |
| Genotype error | 3,494 | 54 | 3,496 | 4,096 | 110 |

Each true reference has at most one mutation per site and no derived mutation
at the flagged positions. Sample matching adds those mutations, producing 3 and
54 recurrent sites respectively in the final raw TSs. Every branch has 600
sample haplotypes and match records, with decoded sample genotypes equal to the
observed input calls. Both zero-error branches' split results equal uninterrupted
matching tables, ignoring provenance. The example validation also checks native
decoding across multiple ancestor chunks. The default configured dataset was
rebuilt with the same policy.

See [plans/initial_mvp.md](plans/initial_mvp.md) for extraction and ID contracts
and [plans/matching_setup.md](plans/matching_setup.md) for matching contracts.
The source VCZs and truth TS are read without modification.

## Direct focal-ancestor intervals

`construct_focal_ancestor_intervals` compares sample haplotypes directly with
eligible inferred ancestors and writes one compressed numeric archive per
dataset. It is included in the default targets. To generate only this product,
select the dataset in your manifest and run:

```sh
uv run snakemake --cores all data/match_eval/haplotype_intervals/out_of_africa_n300_1mbp_inferred_focal_ancestor_intervals.npz
```

Each ancestor uses its leftmost non-padding focal position, chosen before
expanding sample associations or sweeping. Identical carrier patterns at
multiple focal sites do not imply identical intervals. Both sample and ancestor
must have matching derived calls at that anchor; inconsistent inputs fail.

Generation uses the inclusive maximum of `ac_cutoff`, retaining exact AC for
each association. Lower cutoffs select `focal_ac <= cutoff` without repeating
comparisons. Changing smaller configured cutoffs leaves the archive current
when the maximum stays fixed. Changing the maximum AC or
`haplotype_compare.max_mismatches` regenerates intervals. After editing
`lib/haplotypes.py`, force `construct_focal_ancestor_intervals`.

Load with `numpy.load(path, allow_pickle=False)`. For H haplotypes, P eligible
associations, and B = max_mismatches + 1, the archive contains:

| Array | Shape | Meaning |
| --- | --- | --- |
| `sample_id` | `(H,)` | Unicode IDs in focal-NPZ haplotype order |
| `ploidy_index` | `(H,)` | Haplotype index within each sample |
| `offsets` | `(H + 1,)` | Ragged association slices per haplotype |
| `focal_ac` | `(P,)` | Exact focal allele count per association |
| `left_site_index` | `(P, B)` | Inclusive left bounds, column k for budget k |
| `right_site_index` | `(P, B)` | Exclusive right bounds, column k for budget k |
| `num_sites` | scalar | Length of the inferred-panel site axis |
| `max_ac_cutoff` | scalar | Inclusive generation maximum AC |
| `max_mismatches` | scalar | Largest generated directional budget |

All numeric arrays and scalars are int64. Associations follow haplotype order,
then increasing ancestor column. Distinct ancestors with identical bounds
remain separate associations. The roster includes haplotypes with empty slices;
`offsets[0] == 0`, `offsets[-1] == P`, and offsets never decrease. An entirely
empty selection has bounds of shape `(0, B)`. Ancestor IDs, focal indices, BP
bounds, frequencies, and population labels are not repeated in the archive.

Budget k allows **k mismatches per side**, potentially 2k overall. The focal
call matches and consumes neither budget; the next mismatch is excluded.
Missing sample or ancestor calls terminate extension. Bounds are half-open on
the inferred-panel site axis, confined to ancestor support intersected with
the inference interval containing the anchor. For support [0, 10), focal 4,
left mismatches 3 and 1, and right mismatches 6 and 8, budgets 0, 1, 2 retain
[4, 6), [2, 8), [0, 10). Increasing budgets produces nested intervals.

Each physical ancestor-column block is reused across sample associations. A
full-height block serves both directions; multiple physical site blocks retain
numeric sweep state. A spawned multiprocessing Pool uses allocated cores,
capped by physical column blocks, and assembles results in canonical order.
Failures propagate before the archive is written. Completion logs record
association counts and allocated workers.

The interval rule reads only annotated samples, the inferred ancestor store,
and focal candidates. It has no matching, truth, HMM, or tree-sequence inputs.
The existing inference TOML tracks inference and HMM settings, so an HMM edit
can still invalidate that TOML and regenerate upstream ancestors.

`dphil-analysis/src/ch5_analysis.py` reads the archive, checks its roster and
site count, selects requested budget columns, and computes coverage with its
numeric endpoint sweep. Requested AC limits or budgets cannot exceed those
generated. Population metadata continues to come from statistics or the original
truth TS. Coverage summaries and plots remain analysis responsibilities.
Chapter 5 preserves its existing AC >= 2 eligibility: AC=1 associations stay
in the archive and loaded arrays but do not enter its coverage tasks. The
existing 1 Mb simulation and 1KGP stores contain 199 and 180 such associations,
respectively; including them would change the previous coverage summaries.

Measured K=2 archive generation on the existing stores (2026-10-07):

| Dataset | Maximum AC | Associations | NPZ bytes | Generation seconds | Workers |
| --- | ---: | ---: | ---: | ---: | ---: |
| out_of_africa_n100_10mbp | 200 | 925,918 | 10,246,398 | 3.90 | 18 |
| out_of_africa_n100_1mbp | 200 | 114,035 | 945,806 | 1.23 | 18 |
| out_of_africa_n300_1mbp | 600 | 325,711 | 2,775,343 | 1.63 | 18 |
| tgp_chr20_n100_1mbp | 200 | 205,293 | 1,605,287 | 1.80 | 18 |
| tgp_chr20_n100 | 200 | 2,110,059 | 23,873,256 | 12.95 | 18 |

These are library-call wall times including preparation, spawning, comparison,
and compressed writing, measured once without a benchmarking subsystem. They
are NPZ measurements, not workflow timings or multiprocessing speedup claims.
Every generated association and budget was checked against independent direct
mismatch ranks at the leftmost focal. One-worker and multiple-worker n300 arrays
matched exactly; smaller maximum AC selected the same associations and bounds,
and increasing K preserved existing columns. Manual probes checked empty
selections, repeated bounds from distinct ancestors, missing-call barriers,
support ends, zero budgets, and eight physical site-block layouts. All captured
chapter 5 summaries agree exactly at budgets 0 and 1 and the notebook cutoffs
for all five datasets, with chapter 5's existing AC >= 2 eligibility retained.
The existing chapter 5 tests and lint/format checks pass. Explicit and default
workflow dry-runs target NPZ intervals; the explicit branch requires no matching.
A workflow execution succeeded. Smaller cutoff changes left intervals current;
maximum-AC or maximum-budget changes selected only the interval rule.
The coverage notebook produces all five existing plots for both the 1 Mb and
10 Mb simulation/1KGP pairs.

## Checked real-data runs

The frozen 100-individual chr20 cohort completed both 1 Mbp and 10 Mbp runs on
2026-10-01 after the native contig fix. The final configuration retains
`include: 'POS >= 1000000 & POS < 11000000'`, the original sample string, and
CSV superpopulation enrichment.

| Metric | 1 Mbp | 10 Mbp |
| --- | ---: | ---: |
| Inference sites | 7,478 | 82,072 |
| Ancestors | 3,618 | 35,829 |
| Sample haplotypes / match records | 200 / 200 | 200 / 200 |
| Statistics rows | 1,000 | 1,000 |
| Mean bp coverage at AC ≤ 200 | 99.495367% | 99.953116% |
| Total sample switches | 53 | 179 |
| Total sample mismatches | 0 | 0 |

Native evaluation intervals span 999,908 and 9,999,700 bp, respectively. Both
raw tree sequences retain chr20's full sequence length. All selected decoded
calls agree with source genotypes. Population labels are complete: 66 EUR,
66 EAS, and 68 AFR genomes, giving 330 EUR, 330 EAS, and 340 AFR statistics rows
at five cutoffs. Independent checks confirm copying coverage, scores, raw
edges/JSONL joins, and direct interval bounds. Final lint/format checks pass
and the full workflow's second dry run has no pending work.

See [the completed plan](plans/real_data.md#completed-real-data-evaluation-2026-10-01)
for all cutoff summaries, native intervals, validation evidence, and the
1024 MiB native matching-cache setting required by this input's large chunks.
