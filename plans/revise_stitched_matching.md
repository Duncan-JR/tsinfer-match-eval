# Simplify matching and generate diagnostics separately

## Scope and intended result

Keep ordinary matching focused on writing the full-HMM and stitched raw tree
sequences. Remove JSONL path output entirely. Generate focal masks and copying-path
comparison tables through an explicit `diagnostics` target, whose final output is
one portable `.tar.gz` archive.

Support **`af_cutoff` only**, using the current selected-cohort derived frequency
definition. Reuse the configured datasets and Cartesian product of `af_cutoff`
and `min_focal_region_size`; do not introduce a second experiment grid. Preserve
the existing exact-extension, connected-component filtering, independent gap-HMM,
sample selection, missing-call and allele-encoding semantics.

This is an implementation plan, not a request to run the HPC experiment. Keep
implementation in `lib`, configuration in YAML, and plotting in `dphil-analysis`.
Do not add compatibility shims, an extensible diagnostics framework, a new test
suite, or a second matching implementation.

## 1. Simplify core matching

### Append segments and squash once

In `lib/matching.py::_assemble_result`, remove the Python loop that merges touching
equal-parent runs. Concatenate the existing block arrays, convert their site-index
bounds to physical coordinates, and return the segments and mutations directly.
Keep the existing endpoint conventions and owned-result-array requirement.

`append_result` should append these segments directly to the edge table. After all
haplotypes have been appended, call:

```python
tables.edges.squash()
ts = tsinfer.matching.finalise_match_tables(tables)
ts.dump(output_path)
```

Call squash once per stitched output, including the shared runner used by
performance trials. Keep it inside finalisation/output accounting. Do not squash
per haplotype or batch, and do not simplify the raw TS: reference parent IDs must
remain available for path comparisons.

The installed `EdgeTable.squash()` sorts internally by parent, child and bounds;
that order is not necessarily valid tree-sequence edge order. Therefore squash
**before** the existing finaliser, which performs the required table sort before
indexing and mutation-parent computation. Do not add a preliminary table sort or
change the sibling tsinfer repository merely for this operation. Current matching
edges have no metadata, as required by squash.

This removes a Python pass and simplifies assembly. It does append some extra
temporary edge rows and performs a native squash over the accumulated table, so
measure the net effect rather than promising a speedup. Preserve immediate result
append/release, bounded pending work, the persistent matching pool and shared
mapped inputs. Do not otherwise redesign the core worker protocol.

### Remove JSONL throughout this repository

- Delete `path_output` from example and active configurations, including the local
  HPC/performance configs, and remove its README documentation.
- Remove JSONL outputs from `rule all`, matching rules and function arguments.
- Remove `write_result`, file handles, output branches and the coordinator
  arguments used only by JSONL. Keep arguments still needed for TS assembly.
- Stop passing `match_file` to native full-HMM matching; its default already
  disables that output. Do not remove the upstream tsinfer API.
- Remove JSONL handling from benchmark trial setup, `_resident_full_hmm` and
  `quality_check`. Retain sample identity, reference-node and called-allele checks
  against the saved trees.
- Simplify `ExitStack` usage where it existed solely to open the optional file.
  Keep the optional benchmark timing window with a plain context manager and
  `contextlib.nullcontext()` where appropriate. Remove unused imports.

Ordinary runs must neither construct masks nor compute comparison statistics.
Retain the existing opt-in performance measurement facility and its timing
boundaries. Diagnostics do not add flags or payloads to matching workers.

## 2. Configuration and workflow

Add a top-level `config_name` identifying the configured experiment and a
`population` subsection directly inside each existing dataset entry. Do not add
any `diagnostics` YAML heading, worker/compression settings, analysis-window
settings, population-group mapping or performance-directory list. Dataset-local
population settings avoid listing dataset names again elsewhere.

Illustrative configuration:

```yaml
config_name: hpc_params1

datasets:
  - name: out_of_africa_n300_10mbp_error
    # Existing input and matching settings, including ts_path, remain here.
    population:
      source: ts
      field: name

  - name: tgp_chr20_n100
    # Existing input and matching settings remain here.
    population:
      source: csv
      path: ~/work/dphil-analysis/data/misc/tgp_3202_pedigree.csv
      zarr_id_field: sample_id
      csv_id_field: sample_id
      field: superpopulation
```

Use `inference_threads` as the diagnostics rule's requested CPU budget and pass
the actual Snakemake-granted `threads` into the driver. Use that count for the
process pool and pigz, with serial numerical/codec helpers inside diagnostics
workers to avoid nested parallelism. Run compression after the diagnostic workers
have exited. This needs no new resource knob and respects `--cores` and scheduler
allocations. Paths follow the pipeline's existing working-directory-relative
convention and expand `~`. Pigz compression is always level 1.

Population configuration is required for diagnostics on each configured dataset;
it is not required for ordinary matching. Export original population labels;
derive broader groups later in analysis. Derive the physical analysis extent
from the first and last entries of the matched TS's `sites_position`, intersected
with its reference inference intervals. Check that full and stitched site axes
agree. Do not use the whole-chromosome `sequence_length` as a coverage denominator
or attempt to infer original selection boundaries beyond the observed sites.
Site-based statistics include both endpoint sites; physical coverage measures
the span between the endpoint positions. An empty physical denominator gives an
undefined percentage. Record this derived extent with the results.

Give each active/example configuration an explicit lowercase `config_name`.
Resolve diagnostics from the effective merged Snakemake config, regardless of
whether it came from the default YAML, `--configfile`, command-line overrides or
a combination. No parsing of the config filename or separate run selection is
needed. Use distinct names for experiments whose diagnostic outputs should coexist.

Use simple Snakemake orchestration:

1. Define the full-HMM and stitched output lists once, using the existing
   `utils.parameter_values` normalization and dataset list. Reuse these lists in
   `all` and diagnostics dependencies.
2. Add `rule diagnostics`, depending only on
   `{data_dir}/diagnostics/{config_name}_diagnostics.tar.gz`.
3. Add one producer rule that depends on the configured matched trees, prepared
   samples, ancestor panels/references, native configs and population sources.
   When `performance` is non-null, also depend on the performance suite's existing
   final products, obtained through `benchmark.output_directory(config)` and the
   same output list used by `rule performance`.
   Declare its mask and table products using the same parameter lists. Call a
   Python driver in `lib/diagnostics.py` to compute them and package the archive.
   Put combined CSVs, the config snapshot and manifest under
   `{data_dir}/diagnostics/{config_name}/` so different configs do not overwrite
   each other's tables. Keep masks in the parameterized `focal_masks` paths below.
   Keep archive creation in a small helper in that module; no separate packaging
   framework is needed.

No checkpoint rules, filesystem experiment discovery or per-haplotype Snakemake
jobs are needed. The target gathers every result specified by the effective
config. Missing matched outputs are produced through the normal DAG. Enabled
performance testing is a required dependency: finish that configured suite before
packaging it, or reuse it if already complete. With `performance: null`, include
no benchmark results and schedule no benchmarks. Do not collect unrelated old
experiments merely because their directories exist. Record the effective config
as rule parameters so changes invalidate the diagnostics archive and tables.

Command:

```sh
uv run snakemake diagnostics --configfile config.yaml.hpc_params1 --cores 196
```

## 3. Reconstruct focal masks without HMM matching

### Reuse the numeric algorithm

The current `matching.focal_runs` already implements the required exact extension,
stable sort and greedy component cover. Its `Worker` also owns HMM indexes and a
matcher, which diagnostics do not need.

Extract only the exact-matching scratch/state and small numeric entry points into
`lib/greedy.py`, using a concrete dataclass. Let the matching worker hold that
state alongside its HMM state. Reuse `greedy.extend_exact` and
`greedy.cover_components` unchanged. Avoid generic worker classes, optional HMM
fields and a large refactor of matching coordination.

Diagnostics should process one dataset/AF pair at a time:

- Use the native sample roster and existing `haplotypes.prepare_samples`,
  `prepare_reference`, reference packing and chunk-loading helpers. Preserve the
  exact reference site axis, cohort-derived AF, ancestral polarisation, focal
  choice and clipped ancestor supports used by matching.
- Pack the reference once for the AF pair. Use a spawn multiprocessing pool with
  its task/result queues to process physical sample chunks. Reuse the pool for
  independent reference-packing tasks if convenient, finishing packing before
  submitting sample work. Pass descriptors and chunk indices rather than large
  genotype matrices through queues.
- Each sample-chunk task reads its haplotypes once. For each haplotype, extend and
  stable-sort candidates once, then call the existing cover kernel for every
  configured minimum size. The kernel can reuse its output scratch. Union its
  touching output runs to obtain focal components, ignoring parent IDs.
- Reusing the cover kernel is preferable to implementing a new mask-only sweep.
  Computing its unused parent choices is acceptable. Sharing candidate extension
  and sorting across minimum sizes needs only a small helper boundary and avoids
  repeating the expensive input/extension work.
- Never initialise native matcher indexes, run gap HMMs, build sample TS tables
  or call `match_stitched_samples` to obtain masks.

Keep scheduling simple: a dataset/AF pair at a time, sample-chunk tasks distributed
over available workers, and a single coordinator writing outputs. Per-worker
memory includes one decoded chunk, exact scratch and the tree data needed for
analysis. Use the granted rule allocation; do not load a dense cohort-wide
genotype or parent matrix. Gather compact interval results for one pair, write
them, and release them before the next pair. Start with this implementation;
do not add shard caches, resume protocols or cross-AF shared catalogs without a
measured need.

### Mask representation and files

Use one NPZ per dataset/AF/minimum combination:

```text
focal_masks/af_{af_cutoff}/min_region_{min_focal_region_size}/{name}_focal_mask.npz
```

Represent the mask as maximal alternating focal/gap intervals, not a dense
haplotype-by-site Boolean matrix. A small `FocalMask` dataclass and its NPZ form
should contain:

- canonical `source`, `sample_id`, `ploidy_index` arrays, including every matched
  haplotype of every selected individual;
- `offsets` of length haplotypes + 1 into flat `left_site_index`,
  `right_site_index`, `is_gap` arrays;
- per-haplotype `domain_start`, `domain_end` on the inference-site axis;
- `num_sites`, `af_cutoff`, `min_focal_region_size`, `focal_choice`.

All intervals are half-open. `is_gap=True` means outside retained focal coverage,
consistent with a mask excluding focal sites. A dense focal-coverage indicator,
if needed for one haplotype's statistics, is the inverse of that gap mask.
Use numeric/string arrays and `allow_pickle=False`; no serialized Python objects.
Use ordinary `np.savez`, letting final pigz compression handle the archive rather
than adding per-file compression work by default.

The matching domain is first through last called inference site, or the full
reference axis for an all-missing haplotype. Partition that domain only. Empty
focal coverage becomes one gap interval. Preserve the actual kernel's component
filtering semantics, including touching unions; do not change them while deriving
the masks. Store the shared positions and inference intervals once per dataset
in `diagnostics/{config_name}/{name}_sites.npz` so the masks are directly interpretable on the
MacBook without sample/ancestor Zarr stores.

## 4. Population labels and analysis tables

### Restore the useful population joins

Git commit `26df0fb` introduced `extract_ts_populations`,
`extract_csv_populations` and `read_population_metadata` in `lib/utils.py`. The old
config used `metadata_source`, `csv_path`, `zarr_id_field`, `csv_id_field` and
`pop_field`. Reuse the straightforward join logic under the new diagnostics
dataset `population` configuration, without reinstating its old top-level
interface or enriching core matching inputs.

- `source: ts`: read the dataset's existing `ts_path`, use `map_to_vcf_model()` to
  map sample names and ploidy indices to original nodes, and read the configured
  population metadata field. Do not assume selected sample order equals truth
  node order. Export the truth-node mapping for later analysis.
- `source: csv`: use the selected sample's configured Zarr identifier field to
  join the CSV identifier column; export the configured population field for each
  actual ploidy index. Do not hardcode diploidy in the diagnostic join.
- Fail clearly on missing/duplicate selected identities or missing labels; do
  not guess labels. No broad metadata-source plugin system is needed.

Write `sample_populations.csv` once under the config-specific diagnostics folder,
keyed by `dataset, source, sample_id, ploidy_index`, with the population label.
Repeat labels in the per-haplotype results for convenient standalone plotting.
Truth is used only as an explicitly configured population/identity source and
as an archive artifact. No computed comparison metric depends on truth.

### Read copying paths directly from raw trees

Adapt the small metric functions in
`dphil-analysis/src/ch5_stitching.py` and `ch5_hyperparams.py`; do not import that
repository at runtime or reproduce its old JSONL/config loaders.

Read sample-child edges from full and stitched raw trees, sort each child's
segments by left bound, and evaluate parents at reference inference positions.
Align samples by `(source, sample_id, ploidy_index)` in node metadata. Compare
original reference parent IDs, without post-processing or simplification.
Group/index sample edges once per loaded tree, rather than scanning all edges
separately for every sample. Compute one haplotype's dense parent arrays at a
time and discard them after its rows are produced. Reuse the full-HMM parents
across minimum sizes within a task.Review tskit documentation in depth for
guidance on how to do this TS analysis efficiently and elegantly

Implement only the following outputs, covering the entire configured grid:

| Output under `diagnostics/{config_name}/` | Contents |
| --- | --- |
| `haplotype_results.csv` | One row per dataset/AF/minimum/haplotype, population labels, evaluated domain sizes, total and partitioned switches, parent agreement, focal/gap site counts and physical coverage. |
| `region_lengths.csv` | One row per maximal focal/gap region, with run/sample keys, site-index bounds, number of sites, clipped physical bounds and span in bp. |
| `run_summary.csv` | One row per dataset/AF/minimum: sample/haplotype counts, counts of undefined percentages and empty regions, and means/SEM/counts for the principal per-haplotype metrics. |
| `sample_populations.csv` | Selected roster, original population labels and available truth-node mappings. |

Use `af_cutoff` and `min_focal_region_size` as run keys; do not retain legacy AC,
mismatch-budget or focal-method columns that no longer describe a grid axis.
Keep the raw counts needed to recompute percentages, rather than only summaries.

Metric definitions should match pt6 where its complete-domain assumptions hold:

- `num_extra_switches = stitched_switches - full_switches`.
- `switch_increase_percentage = 100 * num_extra_switches / full_switches`;
  undefined when the full count is zero. Preserve pt6's additional
  `extra_switch_fraction` only with its explicit denominator of stitched switches.
- Count switches from changes in adjacent valid reference parent IDs, never from
  edge/segment counts alone. Classify both methods' switches against the same
  reconstructed mask: focal/focal, gap/gap, or a change of region at a seam.
  Equal parents across a seam are not a switch.
- Export total parent agreement and focal/gap agreement separately, using the
  respective region's evaluated site count as denominator. Export disagreement
  counts and a `parents_identical` indicator. Empty regions have undefined
  percentages, not zero agreement.
- Export region site counts/lengths, physical coverage in bp and percentages.
  Convert bounds using the matcher's physical endpoint conventions, then intersect
  with the extent derived from the output site positions and reference inference
  intervals. Clip terminal edges to that extent rather than chromosome bounds.
- Evaluate path comparisons only where both paths have a parent within the
  haplotype's matched domain. Do not count a missing parent as agreement or a
  switch. Do not count transitions across excluded inference intervals. Record
  excluded/unmatched sites and physical span explicitly; terminal missing
  ancestry is not silently relabelled as an HMM gap.
- For physical coverage, report focal, gap and unmatched span within the analysis
  extent's inference intervals, and record the denominator. Thus focal + gap is
  100% for a fully matched domain; otherwise unmatched span explains the remainder.
  Split region-length rows at excluded physical intervals rather than counting
  their span. Keep the saved masks on the original inference-site axis.
- Summary means weight haplotypes equally. Calculate the mean of per-haplotype
  percentage increases, not the ratio of summed counts. SEM describes haplotype
  variability and does not account for within-individual dependence. Leave
  population-specific regrouping and confidence intervals to the notebook.

These rules remove the old loader's assumption that every path spans the whole
chromosome, while retaining the existing definitions on the usual complete data.
Do not compute edge/tree counts, ARG distances, ARF/TPR, topology statistics,
dating, mutation-age correlations or truth comparisons. The archive's trees allow
the user to perform those analyses separately.

## 5. One portable analysis archive

Write `{data_dir}/diagnostics/{config_name}_diagnostics.tar.gz` last. Use a small
manifest-driven file list derived from the same configured run inventory and generated outputs.
Include each source file once, with relative archive paths:

```text
config.yaml                         # Effective merged config used for diagnostics
manifest.json                       # Run inventory, file mapping, versions/revisions
configs/                            # Native inference/matching TOMLs
matches/                            # All configured full/stitched outputs, tszipped
ancestors/                          # One tszipped raw reference per dataset
truth/                              # Each configured simulated TS, tszipped
focal_masks/                        # Every dataset/AF/minimum mask
diagnostics/                        # All CSV tables and per-dataset site axes
performance/                        # Configured suite's files and tszipped trial trees
```

The full-HMM output and ancestor reference are included once per dataset, rather
than repeated per parameter pair. Deduplicate repeated source paths, including
shared truth inputs, and record every dataset's mapping in the manifest. Preserve
distinct benchmark trial outputs: their trial identities are meaningful even
when their scientific contents happen to agree.

When performance is enabled, obtain its directory from
`benchmark.output_directory(config)`; the current location is
`data/performance/{experiment_id}`, independently of `data_dir`. Once the suite's
declared final products are complete, include all its files: manifests/configs,
run/stage/quality tables, memory samples, logs and trial trees. Preserve the
experiment structure. Do not add a directory-selection option, scan other
experiments or package an incomplete suite as a completed result. When performance
is disabled, there is no performance directory to collect for this config.

Never include Zarr directories, even if found beneath an included performance
root. Do not include temporary mapped inputs, the archive itself or stale masks
and trees for parameter combinations outside the current run inventory. Optional
`ts_path: null` means no truth file; a configured non-null truth path must exist.

Record the effective config, dataset/run-to-file mapping, cohort roster, analysis
extents derived from output sites, reference site axes, population-label provenance
and available pipeline, tsinfer and dependency versions. Benchmark manifests retain their own original
versions; label newly captured versions as diagnostics-time information, not
proof of which code produced older trees. Include resolved population tables so
the MacBook does not need the original CSV paths or HPC paths. Include native
configs as provenance, with the portable manifest as the authoritative file map.

Before archiving, compress every included tree sequence with `tszip`, including
full-HMM, stitched, ancestor-reference, truth and performance-trial trees. Add
`tszip` as a dependency. Stage the compressed `.trees.tsz` files in a temporary
directory, preserving their relative archive paths; compress each unique source
tree once. Use a simple pool over independent files bounded by the granted rule
allocation, after the analysis pool has exited. Keep the original raw outputs.
Include only the tszipped copies in the archive, never both formats. Record each
source-to-archive mapping in the manifest, so benchmark records referring to raw
trial filenames can be resolved after extraction. Do not simplify or otherwise
transform trees as part of compression.

Use Python `tarfile` streaming mode (`w|`) into
`pigz -p <allocated_workers> -1`, writing a temporary sibling archive and renaming
it after successful compression. Stream other files from their existing paths;
only the tszip outputs need staging. This avoids copying all source files or shell
quoting source paths. Check the compressor exit status and report a missing `pigz`
clearly; no silent serial gzip fallback. Remove temporary tszip files after
packaging. Do not add a compression-level configuration option.

The archive supports detailed path, region, population and tree analyses without
Zarr. It cannot regenerate exact masks or restore omitted/non-inference sites for
fresh dating without the original input stores. Document the output schemas and
this boundary concisely in the repository README. Do not generate or include an
`analysis_readme.md` in the archive. Include the effective merged configuration as
`config.yaml`, even when several configuration sources contributed to it.

## 6. Implementation order and verification

1. Remove JSONL support and replace Python coalescing with one final squash. Check
   the ordinary and benchmark call sites together so signatures remain consistent.
2. Extract the minimal exact scratch/helpers, implement reconstruction and the
   population joins in `lib/diagnostics.py`, and add `config_name` and dataset
   `population` settings.
3. Add tree-based comparison tables, sharing the run inventory and mask results.
4. Add the simple diagnostics rule, configured performance dependency, manifest,
   tszip preparation and pigz archive helper. Document
   configuration, target invocation and output schemas concisely in the README.
5. Verify with `uv run` tooling and small manual fixtures/existing local data;
   do not create a test suite or run the full HPC grid as part of implementation.

Acceptance checks:

- Matching still reconstructs called alleles, preserves reference/sample identity
  and gives the same copying parents and mutations as before. Compare scientific
  tables/paths rather than file bytes or provenance timestamps. Adjacent
  equal-parent/child edges are squashed, including across focal/gap seams.
- Check a small AF/minimum grid, serial and multicore diagnostics, all-focal and
  all-gap cases, rejected short components, touching candidates, missing terminal
  calls, all-missing calls and separated inference intervals. Reconstructed masks
  must agree with the matching algorithm's focal runs before assembly.
- Per-run totals equal the configured selected haplotype count. Population joins
  match known simulated and CSV labels. Partitioned switch counts sum to their
  method total; extra switches equal the total difference; zero denominators are
  represented and counted explicitly. Physical coverage reconciles with the
  recorded domain and unmatched span.
- Read masks and tszip-decompressed trees from the extracted archive and recompute
  a few diagnostic rows without access to Zarr or original HPC paths. Check expected run coverage,
  optional truth/performance inclusion, no duplicate baseline entries and no Zarr
  members. Check performance enabled with initially missing results (the DAG must
  require their generation), enabled with complete results, and disabled with
  unrelated old performance directories present (they must not be collected).
  Check distinct `config_name` values produce distinct archive/table destinations,
  and that the archived `config.yaml` reflects effective config overrides.
- Dry-run `all`, `performance` and `diagnostics`. Ordinary matching must have no
  diagnostics dependency or JSONL products; diagnostics must require benchmarks
  exactly when the effective config enables performance testing.
  Run one small timed before/after core comparison to detect a material regression
  from the extra temporary edges/native squash. Stop optimisation once this is
  correct and diagnostics run in reasonable time.

The intended net effect is fewer branches, arguments and assembly lines in core
matching, plus one focused diagnostics module and a small explicit workflow.
