# Real-data support with native tsinfer filters

## Scope

Support optional dataset `include`, `exclude`, and `samples` settings by
passing them unchanged into native tsinfer TOML. Ancestor inference, ancestor
matching, and sample matching must use the same configured sample source.
Leave duplicate positions, ancestral-state eligibility, and inference-site
selection to tsinfer. Keep the population metadata feature described below.

This revision replaces the earlier mask-based and compact-store design.
No production code or `config.yaml` is changed by this planning task. The
cohort file below has already been created; implementation, validation, and
the final 10 Mbp run are future work. Follow [../AGENTS.md](../AGENTS.md).
Do not add unit tests or a compatibility layer around tsinfer.

## Completed cohort preparation

The downloaded pedigree is
`~/work/dphil-analysis/data/misc/tgp_3202_pedigree.csv`, converted from the
official [1000 Genomes pedigree](https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/20130606_g1k_3202_samples_ped_population.txt).
Its columns are:

```text
family_id,sample_id,father_id,mother_id,sex,population,superpopulation
```

It has 3,202 unique participant IDs, all present in the chromosome-20 Zarr.
Before rewriting this plan, randomly sampled without replacement from its
`population` column and saved the fixed cohort in
[data/tgp_chr20_n100_samples.yaml](../data/tgp_chr20_n100_samples.yaml).

| Population | Available individuals | Selected individuals | Haploid genomes |
| --- | ---: | ---: | ---: |
| CEU | 179 | 33 | 66 |
| CHB | 103 | 33 | 66 |
| YRI | 178 | 34 | 68 |
| Total selected | | 100 | 200 |

The file contains `name: tgp_chr20_n100` and a native comma-separated
`samples` string. Its comment records random seed
`74866665015432590244730031205280436023`. Selected IDs are ordered by their
original Zarr columns. Native `tsinfer.vcz.resolve_samples_selection` was
checked against that string and selects exactly those 100 IDs in that order.
Freeze this cohort and use it for both region sizes.

The native TOML `samples` field takes an ID string, not a file path. Copy the
saved fragment's literal value into the dataset entry; no `samples_file` API
or automatic file expansion is needed. Native exclusion strings beginning
with `^` may also be passed through unchanged.

The dataset name deliberately omits a region-size suffix, so the final
10 Mbp configuration and the frozen cohort file retain the same name.

## Dataset configuration

During implementation, add this entry to `config.yaml`, keeping its existing
simulation entry and global HMM, cutoff, thread, and haplotype-comparison
settings:

```yaml
- name: tgp_chr20_n100
  zarr_path: ~/work/dphil-analysis/data/zarr_vcfs/tgp/chr20/data.zarr
  ts_path: null
  ancestral_state:
    field: variant_ancestral_allele
  include: 'POS >= 1000000 & POS < 2000000'
  samples: NA06989,NA06993,NA06995,NA07019,NA07031,NA07345,NA07347,NA10831,NA10840,NA10843,NA10851,NA10864,NA11917,NA11995,NA12005,NA12006,NA12045,NA12056,NA12274,NA12282,NA12286,NA12336,NA12400,NA12546,NA12716,NA12740,NA12750,NA12752,NA12753,NA12761,NA12877,NA12889,NA12890,NA18489,NA18497,NA18504,NA18507,NA18516,NA18518,NA18519,NA18528,NA18532,NA18533,NA18536,NA18538,NA18552,NA18557,NA18565,NA18566,NA18571,NA18574,NA18579,NA18582,NA18591,NA18593,NA18606,NA18609,NA18611,NA18617,NA18618,NA18619,NA18623,NA18625,NA18630,NA18637,NA18640,NA18641,NA18642,NA18644,NA18648,NA18745,NA18747,NA18748,NA18853,NA18858,NA18865,NA18869,NA18877,NA18881,NA18906,NA18909,NA18923,NA18924,NA18930,NA19092,NA19097,NA19102,NA19109,NA19128,NA19129,NA19141,NA19142,NA19149,NA19150,NA19189,NA19203,NA19207,NA19210,NA19211,NA19223
  metadata_source: csv
  csv_path: ~/work/dphil-analysis/data/misc/tgp_3202_pedigree.csv
  zarr_id_field: sample_id
  csv_id_field: sample_id
  pop_field: superpopulation
```

The source store contains chromosome-20 variants only. The POS predicate above
selects a 1 Mbp interval directly, without region-string conversion, a
`region_index`, or shifting stored coordinates. Read-only native iteration
confirmed that it selects 23,775 source rows. Duplicates in those rows are
tsinfer's responsibility; do not prefilter them here.

Omitted or null `include`, `exclude`, and `samples` settings are omitted from
the generated source section. Preserve native expression and sample-string
syntax. The inspected checkout reports tsinfer `0.5.2.dev317` and vcztools
`0.1.2` while exposing the new native configuration interface. In this backend,
`include` and `exclude` are alternatives: it rejects both in one source.
Support either key independently and let native validation enforce its
contract; do not combine or rewrite expressions to emulate simultaneous use.

For an exclude-only check, omit `include` and use:

```yaml
exclude: 'POS < 1000000 | POS >= 2000000'
```

Native iteration confirmed that this selects exactly the same rows as the
include example. The single `&` and `|` forms used here are supported by
the inspected backend. Do not add other filtering APIs or hidden quality
filters.

One existing native/input issue must be checked during integration: this VCZ
retains a 24-contig header although its variants are all chr20. The inspected
checkout reads the first declared contig for its output name/length, which is
chr1, rather than chr20's 64,444,167 bases. Verify the intended native tsinfer
version handles this input correctly. If it does not, report the upstream/input
issue instead of adding a pipeline contig-normalization workaround or accepting
incorrect chromosome metadata.

## Small implementation changes

### Native TOML and shared source

Extend `utils.write_inference_config` with the dataset's optional native
filter settings. Copy non-null `include`, `exclude`, and `samples` strings
directly into `[[source]]`, preserving them exactly. Use the same writer for
inferred and optional true-panel configurations.

Keep both match sources, scalar HMM settings, raw tree sequences, and
`path_compression = false`. Ancestor inference consumes the sample source
from this TOML. Ancestor matching retains it for the existing split handoff;
sample matching retains that exact source via the existing
`dataclasses.replace` construction. Do not apply the participant sample
string to the generated ancestor source.

Remove the current hardcoded
`exclude = "INFO/match_eval_singleton_mask == 1"` and its redundant singleton
annotation/readers. Native filtering and native inference determine the site
axis. This avoids conflicting source filters and introduces no replacement
expression or custom site-selection policy.

### Selected-cohort counts and local sample readers

Keep the existing regular copied sample store and output paths. Do not build
a compact VCZ, copy arrays through a new storage abstraction, or normalize
headers. Rename the count-preparation helper/rule if needed to reflect that
its purpose is writing `variant_match_eval_derived_ac` and
`variant_match_eval_derived_af`.

The existing counts must describe the selected 100 individuals. Resolve
columns with native `tsinfer.vcz.resolve_samples_selection(store, samples)`
and compute the current derived-count/frequency annotations from those calls.
Read the existing genotype chunks with selected columns to avoid loading the
full 3,202-person genotype tensor. This is count annotation, not variant
filtering; retain the store's original variant/sample axes.

Make the same small column-selection change in `matching.find_focal_ancestors`
and `haplotypes.construct_focal_ancestor_chunks`: use native selection for
sample IDs and genotype columns. Their site axis already comes from the
native inferred panel. No include/exclude evaluator or duplicate-position
handling is needed in these functions. Preserve their existing numerical
algorithms, candidate storage, and source/sample/ploidy joins.

Track the native filter strings in config-writing rule `params`, and
`samples` in annotation/focal/chunk rule `params`. Pass `None` through
intermediate calls and let the native resolver select all samples when
appropriate. Changing native site filters rebuilds the generated TOML and its
dependent inference/matching/analysis products. Changing the sample string
also refreshes cohort counts and haplotype rows.

### Population metadata

Retain optional per-dataset `metadata_source`:

- Omitted or null: no population enrichment.
- `ts`: use the existing `ts_path`, which must name an existing original TS.
- `csv`: require `csv_path`, `zarr_id_field`, `csv_id_field`, and `pop_field`.
  These fields are optional outside CSV mode. Read a standard headered CSV
  with string identifiers; no delimiter or identifier inference.

Add these small utilities:

```text
extract_ts_populations(samples_path, ts_path, samples=None) -> pandas.DataFrame
extract_csv_populations(samples_path, csv_path, zarr_id_field,
                        csv_id_field, pop_field, samples=None) -> pandas.DataFrame
read_population_metadata(samples_path, dataset) -> pandas.DataFrame | None
```

Both extractors use the native sample resolver and return one row per selected
haploid genome, indexed uniquely by `(source, sample_id, ploidy_index)`.
`source` is `samples`; `sample_id` is the canonical Zarr sample ID, even
when CSV lookup uses a different sample-dimensioned `zarr_id_field`.

For TS data, follow
`~/work/dphil-analysis/notebooks/ch5_initial_testing.py`:

1. Load the original TS and call `map_to_vcf_model()`.
2. Match selected Zarr IDs to `individuals_name` and use the corresponding
   `individuals_nodes` columns in ploidy order. The original TS may contain
   more individuals than the selected cohort; join by ID.
3. Read each original node's individual and population, returning
   `truth_node_id`, `truth_individual_id`, `population_id`, and
   `population = ts.population(node.population).metadata["name"]`.

The inspected simulation's individual metadata are empty: use their node
association and the population table as in the notebook. Keep each genome's
own node population. Require selected IDs/ploidy to match the original mapping
and require labels; do not infer identities from matched TS node numbers.
Read-only joins already verified this notebook approach against the existing
statistics and chunk outputs.

For CSV data, match selected `zarr_id_field` values to unique
`csv_id_field` values, then repeat the chosen `pop_field` label across each
individual's ploidy indices. Extra CSV participants are expected. Missing
selected IDs or labels are errors, not sample exclusions. The common output
column is `population`; for this dataset its values come from
`superpopulation`: EUR for CEU, EAS for CHB, and AFR for YRI. CSV data need
no artificial TS IDs.

Merge this small table into both dataframe outputs immediately before writing:

```text
{name}_inferred_focal_ancestor_stats.csv
{name}_inferred_focal_ancestor_chunks.csv
```

Use their existing sample-haplotype columns against the metadata index with
`how="left"`, `validate="many_to_one"`, and `sort=False`. Preserve all
row keys, row counts, ordering, and numerical values, including an empty chunk
dataframe. Disabled metadata preserves the existing schemas. TS mode adds its
truth/population IDs as well as the label. The site-level true-ancestor CSV
has no sample-haplotype key and does not receive this annotation.

The two dataframe rules take the metadata file as an optional explicit input
and track the metadata settings in `params`; the statistics rule also needs
the sample store as an input. A metadata-only change should rebuild these CSVs
without rerunning inference or matching. No new metadata artifact, cache,
population-filtering rule, or TS-table mutation is needed.

### Documentation

Update `README.md` and `config.yaml.example` with native filter pass-through,
selected-cohort counts, the frozen sample fragment, and the metadata settings.
Keep production changes confined to the existing utilities, sample-reading
leaves, and Snakefile wiring; do not modify tsinfer to implement pipeline
filtering.

## Minimal verification and final run

1. Implement the changes above. Use fresh real-data outputs, or force affected
   rules after library edits because library files are not Snakemake inputs.
   Run `uv run ruff check lib` and `uv run ruff format --check lib`.
2. Generate and parse the native TOML. Check literal filter/sample values and
   verify the same sample source reaches all three native stages. Check the
   exclude-only example by native position selection against the include
   example; a second complete pipeline run is unnecessary for this equivalence.
   The cohort file and this native equivalence check are already verified.
3. Add the 1 Mbp dataset entry above to `config.yaml`. Run
   `uv run snakemake --cores all --dry-run`, then
   `uv run snakemake --cores all` through both dataframe outputs. Confirm native
   chromosome identity/length before accepting results. Let tsinfer determine
   the inference-site and ancestor counts and report the actual counts; the
   earlier cohort/mask-based counts no longer apply.
4. Check the resulting 200 sample haplotypes and 200 native JSONL records by
   `(source, sample_id, ploidy_index)`. Compare decoded alleles with selected
   source calls on the native panel axis. Verify focal/chunk IDs refer to that
   same cohort, and the statistics have
   `200 * len(ac_cutoff)` rows. Evaluate copying coverage, switches, mismatches,
   and likelihood with the existing contracts and native sequence intervals.
5. Verify complete population labels in both CSVs. The selected metadata table
   must have 66 EUR, 66 EAS, and 68 AFR genome rows, with the finer pedigree
   populations still matching 33 CEU, 33 CHB, and 34 YRI individuals. With five
   cutoffs, statistics have 330 EUR, 330 EAS, and 340 AFR rows. Check the small
   existing simulation with `metadata_source: ts` and metadata disabled;
   preserve numerical values, keys, and order. Keep this to direct integration
   checks, without a new test suite or a broad configuration matrix.
6. Confirm a second dry run has no work pending and summarize the actual
   1 Mbp evaluation. Resolve native/input problems within the supported native
   contract; do not reintroduce custom filtering or preprocessing workarounds.
7. **After the 1 Mbp run is confirmed and evaluated, run the pipeline once on
   10 Mbp with the identical frozen 100 individuals.** Change only this
   dataset's include expression to:

   ```yaml
   include: 'POS >= 1000000 & POS < 11000000'
   ```

   This selects 10 Mbp of chr20. Rebuild the native-config-dependent products,
   run `uv run snakemake --cores all` through both dataframe outputs, and
   summarize the larger run's evaluation and population-label checks.
   Leave `tgp_chr20_n100` in `config.yaml` with this **10 Mbp** expression,
   unchanged `samples` string, and CSV metadata settings afterwards.
   Retain the cohort file and completed 10 Mbp outputs. Do not restore the
   temporary 1 Mbp configuration.

Completion means native filter pass-through, consistent selected-cohort
counts/haplotypes, both population-enriched dataframes, and the final successful
10 Mbp run retained in the configuration. Today's deliverables are the frozen
sample fragment and this simplified plan only.


## Implementation status (2026-09-30)

The pipeline changes are implemented: native source filters, selected-column
counts and local readers, TS/CSV population enrichment, Snakemake dependencies,
and configuration/documentation. No tests, compact stores, header normalization,
or tsinfer changes were added.

Lint and formatting checks pass. Generated native TOML preserves literal values,
omits null settings, and retains the same sample source through the split stages.
Native include/exclude iteration gives the same 23,775 rows. Counts were compared
to selected calls across that interval. The metadata join gives 66 EUR, 66 EAS,
and 68 AFR genomes, corresponding to 33 CEU, 33 CHB, and 34 YRI individuals.
Diagnostic focal/chunk preparation has 200 haplotypes, 295,057 seeds, and 885,171
chunk rows; labels are complete and merging preserves values and row order,
including empty output. These diagnostic chunks were checked in memory.

The existing n100 simulation completed through both CSVs with TS metadata
and with metadata disabled. Its 1,000 statistics rows and 4,284,249 chunk rows
have identical original columns, values, and ordering in both modes. A second
dry run has no pending simulation work; switching only metadata schedules only
the two dataframe rules. Original
TS metadata also joins completely to the existing n300 simulation's 3,000
statistics rows and 1,180,755 chunk rows without changing their values or order.

**On 2026-09-30, real-data completion was blocked by the native/input contig issue.**
Installed tsinfer `0.5.2.dev318` reads the first declared contig. The input has
`variant_contig == 19` (chr20), but declares chr1 first. Actual 1 Mbp ancestor
inference produced 7,478 sites and 3,618 ancestors, labeled `chr1` with length
248,956,422 instead of chr20's 64,444,167. Its native inference interval is
[1,000,026, 1,999,934). These are diagnostic outputs, not accepted results.
Real-data matching/evaluation and the conditional 10 Mbp run were therefore
not performed at that time. The temporary 1 Mbp configuration was retained
pending the upstream fix. This blocker was resolved and both runs completed
on 2026-10-01, as recorded below.


## Completed real-data evaluation (2026-10-01)

The user fixed native contig resolution in the adjacent tsinfer checkout. The
small bug reproducer now passes both inference and matching: chr20 and
64,444,167 bases. The pipeline performs no header normalization. Runs use
runtime tsinfer `0.5.2.dev318`, base commit
`a7043084c74ed491bf3be8cd8670638a21285f4e`, with the user's local contig fix.
Source hashes are retained in
[the revision record](../tmp/real_data_tsinfer_revision.json).

Forced real-data ancestor inference refreshed the previous diagnostic output.
The 1 Mbp run completed and was checked before changing the interval to 10 Mbp.
The full workflow's second dry run had no pending work. The identical frozen
100 individuals were then run over `POS >= 1000000 & POS < 11000000`.

The 10 Mbp run exposed a native cache-size requirement: its sample-source chunk
needs 501.2 MiB, exceeding the native 256 MiB default. Optional per-dataset
`matching_cache_size` now passes native cache sizing through both matching
leaves, with None passed through when omitted. The real dataset uses 1024 MiB;
this changes memory allocation, not HMM settings, genotype data, filters, or
matching contracts. Rule params track it. An initial cache-size failure was
resolved without modifying tsinfer or compacting the store. A fresh Snakemake
runtime source cache was used after rule edits to avoid stale cached rule code.

| Metric | 1 Mbp | 10 Mbp |
| --- | ---: | ---: |
| Native source rows | 23,775 | 254,189 |
| Inference sites | 7,478 | 82,072 |
| Inferred ancestors | 3,618 | 35,829 |
| Raw ancestor nodes | 3,620 | 35,831 |
| Raw matched nodes | 3,820 | 36,031 |
| Sample haplotypes / JSONL records | 200 / 200 | 200 / 200 |
| Statistics rows | 1,000 | 1,000 |
| Chunk rows | 885,171 | 9,763,548 |
| Evaluated bp per haplotype | 999,908 | 9,999,700 |
| Total sample switches | 53 | 179 |
| Total sample mismatches | 0 | 0 |
| Mean normalized path log likelihood | -3.389178 | -13.498112 |

Both tree sequences retain chr20's full 64,444,167-base length. Native sequence
intervals are [1,000,026, 1,999,934) and [1,000,026, 10,999,726), respectively;
evaluation uses these intervals rather than including empty selected-window
flanks. The likelihood is the existing normalized mismatch/switch score,
not an absolute sequence likelihood.

| Inclusive AC cutoff | 1 Mbp mean bp coverage | 10 Mbp mean bp coverage |
| --- | ---: | ---: |
| 3 | 99.128380% | 99.914866% |
| 5 | 99.184700% | 99.936379% |
| 10 | 99.327787% | 99.948248% |
| 100 | 99.495367% | 99.952567% |
| 200 | 99.495367% | 99.953116% |

Both runs have complete population labels: 66 EUR, 66 EAS, and 68 AFR genomes,
from 33 CEU, 33 CHB, and 34 YRI individuals. Each statistics CSV has 330 EUR,
330 EAS, and 340 AFR rows. Chunk rows are:

| Population | 1 Mbp chunk rows | 10 Mbp chunk rows |
| --- | ---: | ---: |
| EUR | 290,409 | 3,213,939 |
| EAS | 295,890 | 3,223,812 |
| AFR | 298,872 | 3,325,797 |

Direct integration checks join all 200 native JSONL records, raw sample nodes,
and focal rows by `(source, sample_id, ploidy_index)`. Decoded alleles match
selected source calls on every native panel site: 1,495,600 genome-site calls
for 1 Mbp and 16,414,400 for 10 Mbp. The 10 Mbp selected calls were also compared
directly to the untouched original store. Cohort AC/AF annotations agree with
the calls. Raw sample edges match native copying segments. Independent interval
clipping and site counting agree with every coverage row; switches, mismatches,
and normalized scores agree with the native records and configured HMM.

All chunk rows were checked for cohort/ancestor joins, carried focal calls,
AC/AF, budgets, bounds, and complete labels. An independent mismatch-rank and
missing-barrier oracle checked 1,200 first/last chunk rows across all haplotypes
and budgets in the 10 Mbp run. No new test suite was added. The prior simulation
metadata-enabled/disabled checks remain valid, and the existing default
simulation completed again as part of the final workflow.

`uv run ruff check lib`, `uv run ruff format --check lib`, and a final fresh-cache
`uv run snakemake --cores all --dry-run` pass; the dry run has no pending work.
The full successful final workflow ran through both dataframes in 2m39s.

Detailed checked summaries are retained as
[1 Mbp JSON](../tmp/real_data_1mb_summary.json) and
[10 Mbp JSON](../tmp/real_data_10mb_summary.json), with
[the direct integration checker](../tmp/validate_real_data.py) and run logs in
`tmp/`. Production outputs remain in `data/match_eval/` at their normal paths.
`config.yaml` and `config.yaml.example` retain the 10 Mbp expression, unchanged
frozen sample string, CSV metadata settings, and the native cache override.
The frozen cohort and completed 10 Mbp outputs are retained. This plan is complete.
