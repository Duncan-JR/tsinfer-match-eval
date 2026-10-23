# tsinfer matching evaluation

This Snakemake pipeline generates ancestor references and compares full-HMM
sample matching with greedy exact copying plus full-HMM gap matching. Analysis
belongs in separate notebooks.

## Running

Use phased, biallelic, single-contig VCF Zarr inputs and the editable tsinfer
checkout at `../tsinfer`. The runner uses its regional `AncestorMatcher`,
retaining reference topology across independent gap calls, and raw table APIs.

```sh
cp config.yaml.example config.yaml
# Set input paths and the options below.
uv sync
uv run snakemake --dry-run --cores all
uv run snakemake --cores all
```

`af_cutoff` and `min_region_size_kb` accept a scalar or a list. The default
target runs their Cartesian product, sharing annotated samples, ancestor panels,
references and one full-HMM baseline per dataset. Use separate output directories
when changing inputs or HMM settings. Logs record the actual tsinfer revision.

## Matching

1. Copy each input store and annotate derived allele counts and frequencies for
   the selected cohort. Missing calls are excluded; singletons are excluded from
   inference. The annotated store retains both original axes.
2. Infer ancestors and match them to make a raw reference tree sequence. True
   ancestor extraction remains available through explicit `true` targets.
3. Run the independent native full-HMM baseline.
4. Select each ancestor's leftmost or rightmost focal site as its seed. Keep
   ancestors with annotated derived AF at most `af_cutoff`. AF is derived copies
   divided by called haplotypes in the selected cohort at the seed. Pack their supports,
   clipped to the containing inference interval, into one temporary mapped file.
5. Load selected samples by physical Zarr chunks into temporary mapped slabs.
   Each worker matches one complete haplotype. A derived sample seed activates
   its ancestor. Skip missing sample observations while extending in both
   directions; called disagreements and unknown ancestor calls stop extension,
   as do support ends and inference boundaries. Clip supports to the sample's
   called domain before extension and minimum filtering. Different ancestors
   may share a seed; missing seeds activate none.
6. Convert `min_region_size_kb` once per matching run to a rounded mean site
   count: `max(1, round(kb * 1000 * num_sites / (last_position - first_position)))`,
   using the reference's `sites_position` array. Stable-sort intervals by left
   bound and repeatedly copy the interval reaching furthest from the current
   cursor. Filter complete connected unions by that site count, including
   touching intervals. Short intervals can bridge a retained union. Equal
   reaches keep the first interval in stable panel order. This wrapper uses
   genome-wide mean density, rather than checking each union's physical length.
7. Split uncovered gaps at inference boundaries. For a one-site gap (one
   inference-site index), copy a supported parent with a known allele: prefer
   the left retained parent, then the right, then the activated ancestor with
   the furthest support end in stable support-start order. Candidates rejected
   by the minimum remain eligible for this choice. Correct called disagreements
   with sample mutations; missing observations need no mutation. Match remaining
   gaps independently with the full HMM. Append completed results, squash adjacent
   equal-parent/child edges once, then finalise and sort tables.

The greedy cover minimises copying runs within each retained exact union.
Gap matches start independent likelihoods; this does not optimise global HMM
likelihood or switches across gaps. Storage chunks never limit extension.
The matching domain runs from first through last called site, or the full
reference axis for an all-missing haplotype.

One persistent spawn pool prepares the reference and then matches samples.
One loader thread supplies complete slabs; the coordinator owns outputs.
`max_pending_haplotypes` bounds submitted jobs whose results have not yet been
appended. `sample_prefetch_chunks` separately bounds every live slab, including
loading, ready and in-use slabs. Temporary files disappear after children stop.
Mappings share input pages but still incur filesystem I/O. Native indexes are
private to workers, and final tables grow with output size.

## Configuration

Paths are relative to the working directory; input paths, `data_dir`,
`progress_dir` and `temp_dir` expand `~`.

| Option | Meaning |
| --- | --- |
| `config_name` | Lowercase experiment label using letters, digits and underscores. Distinct names keep diagnostic tables and archives separate. |
| `data_dir` | Ordinary output root. Use a separate directory for each biological configuration. |
| `progress_dir` | Per-rule log root. |
| `focal_choice` | `left` (default) or `right` endpoint focal seed. |
| `hmm.recombination` | Constant HMM recombination probability between inference sites, strictly between 0 and 1. |
| `hmm.mismatch` | Mismatch probability for remaining HMM calls, strictly between 0 and 1. Exact extension stops at called disagreements; one-site-gap copies can require sample mutations. |
| `ancestor_chunk_size` | Physical ancestor-column chunk size for explicit true-ancestor extraction only. |
| `inference_threads` | Native ancestor inference workers and requested diagnostics CPU budget. Diagnostics uses the actual Snakemake-granted threads. |
| `matching_threads` | Native ancestor/baseline threads and greedy process workers. |
| `sample_prefetch_chunks` | Maximum live decoded sample slabs; start with `2`. |
| `max_pending_haplotypes` | Maximum queued, running and completed-but-unappended haplotype jobs; also bounds preparation admission. `null` uses the requested worker count. An explicit limit must be at least that count. |
| `temp_dir` | Temporary mapped-file parent, or `null` for the system temporary directory. |
| `helper_threads.blosc_threads` | Codec threads per process; start with `1`. |
| `helper_threads.zarr_async_concurrency` | Zarr async I/O concurrency per process; start with `1`. |
| `helper_threads.zarr_threading_max_workers` | Zarr helper pool size per process; start with `1`. |
| `helper_threads.blas_threads` | Numerical-library threads inherited by spawned children; start with `1`. Numba kernels are serial. |

Snakemake reserves matching workers plus loader/decoding helpers. With all helper
limits set to `1`, greedy requires `max(2 * workers, workers + 3)` CPUs. Increase
`--cores` to meet this reservation, or reduce `matching_threads`.

Each `datasets` entry has:

| Option | Meaning |
| --- | --- |
| `name` | Unique lowercase label with underscores, used in filenames. |
| `af_cutoff` | Inclusive maximum selected-cohort derived AF at the seed, between `0` and `1`; a scalar or nonempty list of distinct values. Missing calls do not enter the denominator. |
| `min_region_size_kb` | Finite nonnegative number or nonempty list of distinct values; requested minimum connected focal-region size in kb. Converted to a rounded mean inference-site count over the first-to-last reference-site span, with a minimum of one site. `0` keeps all regions. Replace the old `min_focal_region_size` config key rather than reusing its site-count values as kb. |
| `zarr_path` | Input VCZ store. |
| `ts_path` | Optional truth tree sequence, used for explicit true-ancestor extraction, configured population joins and archive inclusion; otherwise `null`. |
| `ancestral_state.field` | Array of ancestral allele strings. |
| `ancestral_state.is_reference` | Alternative to `field`: `true` treats REF as ancestral. |
| `include` | Optional native site filter, e.g. `'POS >= 1000000 & POS < 11000000'`. |
| `exclude` | Alternative to `include`. Singleton exclusion is added automatically. |
| `sample_list` | Optional headerless CSV of sample IDs. The same cohort is used for annotation, inference and matching; all ploidy indices are retained. |
| `population.source` | Required only for diagnostics: `ts` joins names/ploidy through `ts_path.map_to_vcf_model()`; `csv` joins selected Zarr identifiers to a CSV. |
| `population.field` | Original population metadata field (`ts`) or label column (`csv`). Labels are exported without regrouping. |
| `population.path` | Population CSV path when `source: csv`. |
| `population.zarr_id_field` | Sample-dimensioned Zarr join identifier when `source: csv`, e.g. `sample_id`. |
| `population.csv_id_field` | CSV identifier column when `source: csv`, e.g. `sample_id`. Selected identifiers and labels must exist and be unique. |
| `matching_cache_size` | Native ancestor/full-HMM reader cache in MiB; `null` uses tsinfer's default. Greedy uses its slab bound instead. |

Coordinates remain absolute when selecting a region. Main products are:

- `samples/{name}_samples_masked.zarr`: count-annotated sample copy. Annotation
  counts selected columns one physical variant/sample tile at a time.
- `configs/{name}_ancestor_inference.toml`: native inference/matching config.
- `ancestors/{name}_inferred_ancestors.zarr` and `.trees`: panel and raw reference.
- `matches/{name}_inferred_samples_raw.trees`: native full-HMM baseline.
- `matches/af_{af_cutoff}/min_region_kb_{min_region_size_kb}/{name}_inferred_greedy_samples_raw.trees`: greedy stitched samples for each parameter pair.

Each greedy tree stores the requested kb minimum and resolved site count in
`ts.metadata["match_eval"]["min_region_size_kb"]` and
`ts.metadata["match_eval"]["min_focal_region_size"]`, alongside AF and focal
choice. Logs and greedy benchmark workload records also report the site count.

For a parameter search, configure for example:

```yaml
datasets:
  - name: example_dataset
    af_cutoff: [0.01, 0.03, 0.05]
    min_region_size_kb: [0, 50, 100, 200]
    # Existing paths and metadata follow.
```

`performance: null` disables the benchmark target. Ordinary matching does not
collect benchmark timings or memory samples.

## Diagnostics

With performance enabled, diagnostics reuses completed trial trees selected by
`performance.diagnostics_run` (`mode`, `workers`, zero-based `repeat`). The suite
must cover every diagnostic dataset and its settings with both methods. Missing
coverage or trees raises an error. Diagnostics depends on suite completion and
prepared inputs, and schedules no ordinary sample matching. With
`performance: null`, it uses ordinary trees under `matches/`. Selected saved run
specifications are recorded in the archive manifest. The ordinary `all` target
always enumerates ordinary outputs independently of this selection.

Ordinary matching writes raw trees only. Reconstruct focal masks and compare raw
copying parents across the configured dataset/AF/minimum grid explicitly.
Regenerate stitched trees before rebuilding masks or diagnostics with this policy:

```sh
uv run snakemake diagnostics --cores all
```

Install `pigz` on PATH and sync the Python dependencies (including `tszip`). The
driver prepares shared copying-path indexes once, then uses spawn workers to decode
bounded waves of physical sample chunks and analyse individual haplotypes. Exact
extension, sorting, activated-support lookup and baseline support are shared across
minimum sizes. It does
not run HMM matching. Numerical/codec
helpers are serial; tree compression and level-1 pigz run after analysis workers
exit, within the granted CPU budget. An enabled `performance` suite is a required
dependency and is included in full. `performance: null` schedules and includes no
benchmarks, regardless of other experiments on disk. The archived `config.yaml`
is the effective merged Snakemake configuration, including command-line overrides.

Outputs under `{data_dir}/diagnostics/{config_name}/` are:

| File | Schema/content |
| --- | --- |
| `haplotype_results.csv` | Keys `dataset, af_cutoff, min_region_size_kb, source, sample_id, ploidy_index`; resolved integer `min_focal_region_size`; original `population`; domain/evaluated/excluded/unmatched site counts; full/stitched total and focal/gap/seam switches; disagreement counts, agreement percentages and `parents_identical`; focal/gap/unmatched bp spans and coverage percentages. |
| `region_lengths.csv` | The same keys/population, mask `region_index`, `is_gap`, half-open site-index bounds, `num_sites`, clipped `left_bp`, `right_bp`, and `span_bp`. Regions split at excluded or unmatched physical intervals. |
| `run_summary.csv` | One row per grid run: sample/haplotype and empty-region counts; per-haplotype metric `_mean`, `_sem`, `_count`; explicit `_undefined_count` for percentages/fractions. |
| `sample_populations.csv` | Selected roster keyed by dataset/source/sample/ploidy, original population, and available `truth_node_id`, `truth_individual_id`, `population_id`. |
| `{name}_sites.npz` | Shared `positions`, reference `inference_intervals`, clipped `analysis_intervals`, observed `analysis_start`/`analysis_end`, and `sequence_length` for interpreting matcher endpoints. |
| `config.yaml`, `manifest.json` | Effective config, run/file map, cohort and population provenance, observed extents, and explicitly labelled diagnostics-time versions/revisions. |

Masks are ordinary NPZ files at
`focal_masks/af_{af_cutoff}/min_region_kb_{min_region_size_kb}/{name}_focal_mask.npz`.
Each contains canonical `source`, `sample_id`, `ploidy_index`, ragged `offsets`,
flat `left_site_index`, `right_site_index`, `is_gap`, per-haplotype `domain_start`
and `domain_end`, plus `num_sites`, `af_cutoff`, `min_region_size_kb`, the resolved
`min_focal_region_size` and
`focal_choice`. Bounds are half-open on the original inference-site axis;
`is_gap=True` excludes focal coverage. Load NPZs with `allow_pickle=False`.

Site comparisons include both observed endpoint sites and require a parent in
both paths within the matched domain. Switches are adjacent valid parent changes,
partitioned by the same focal mask; excluded inference intervals break adjacency.
`num_extra_switches = stitched_switches - full_switches`;
`switch_increase_percentage` divides by full switches, while
`extra_switch_fraction` divides by stitched switches. Region agreement uses that
region's evaluated site count. Undefined ratios are CSV `nan` (and counted in
summaries); an unevaluated `parents_identical` is blank. Physical coverage uses
the span between observed endpoint positions intersected with inference intervals,
not chromosome length. Focal, gap and unmatched spans reconcile to that denominator.
Summary means weight haplotypes equally; SEM describes haplotype variability and
does not account for within-individual dependence.

The final `{data_dir}/diagnostics/{config_name}_diagnostics.tar.gz` contains
`config.yaml`, `manifest.json`, native `configs/`, `matches/`, raw reference
`ancestors/`, optional `truth/`, `focal_masks/`, `diagnostics/`, and the configured
`performance/` experiment. Trees are losslessly tszipped `.trees.tsz` copies;
original raw outputs remain. The manifest maps source filenames to portable
archive paths, deduplicating shared baselines/references/truth without merging
benchmark trial identities. No Zarr stores or temporary mapped inputs are included.
Use `tszip.decompress(path)` to load trees after extraction. The archive supports
path, region, population and tree analysis without Zarr; regenerating exact masks
or restoring omitted/non-inference sites for fresh dating requires original stores.

## Performance

```sh
uv run snakemake performance --cores 12
# Or use already prepared references:
uv run -m lib.benchmark suite config.yaml --allocation 12
```

Performance is opt-in and runs fresh trial processes sequentially. Each
experiment directory must be empty. Output is under
`data/performance/{experiment_id}/`: configuration and dependency/source
manifest, `runs.csv`, `stages.csv`, optional `quality.json`, trial logs,
process-tree memory samples and matched trees. Whole-trial memory includes
validation; use `window_peak_private_bytes` for matching-memory comparisons.
Validation timing is recorded separately, and completed matching measurements
are retained if validation fails.
Each performance dataset accepts scalar or list `af_cutoff` and requires scalar
`min_region_size_kb`. Ordinary matching and diagnostics use each dataset's own
Cartesian product. Both keys are required on datasets; top-level keys are rejected. Each greedy cutoff gets a separate trial; Full HMM runs
once per dataset/mode/worker/repeat and supplies the baseline for all cutoffs.
`runs.csv` records each greedy trial's AF and kb setting, plus its resolved
`min_focal_region_size` in sites. AF and kb are blank for Full HMM trials.

Worker counts are never silently reduced. Every dataset must have at least as
many haplotypes as the largest requested worker count. The suite checks its CPU
reservation against the process's effective CPU affinity on Linux. Its manifest
records host CPU/memory information, allowed CPU/core/socket/NUMA topology and
selected scheduler allocation and binding variables.

For the 100, 1,000 and 10,000-individual simulated cohorts in
`../tsinfer-anc-eval/config.yaml.matching`, `config.yaml.hpc_perf1` measures greedy
AF cutoffs 0.01 and 0.1 at 150 kb, plus one Full HMM baseline, at 32 workers
(nine trials). Generate their inputs from the `tsinfer-anc-eval` directory:

```sh
uv run snakemake --configfile config.yaml.matching --cores 196 \
    --resources mem_mb=950000 --until add_errors
```

Inside an exclusive 196-CPU allocation, set `TMPDIR` to node-local scratch, then
run from this directory:

```sh
uv run snakemake performance --configfile config.yaml.hpc_perf1 --cores 196
```

This target prepares all references before starting sequential measurements.
The ordinary default target can run matching jobs concurrently and should not
be used for benchmark timings. Keep CPU binding, cache size, queue limits and
storage policy fixed across cohorts; use a new experiment ID for each attempt.

| Performance option | Meaning |
| --- | --- |
| `experiment_id` | New output directory name. |
| `datasets` | Configured dataset names to measure. |
| `methods` | `full_hmm`, `greedy`, or both. |
| `modes` | `from_reference`, `resident_match`, or both. |
| `worker_counts` | Positive worker counts, used within the allocated CPU reservation. |
| `repeats` | Repeats of every dataset/mode/method/worker combination. |
| `seed` | Randomises method/worker order within each repeat. |
| `rss_sample_interval_s` | Launcher sampling interval in seconds for private process-tree memory. |
| `trial_timeout_s` | Timeout in seconds; terminates the whole trial process group, including workers. |
| `collect_stages` | Enable diagnostic aggregate setup, packing, loading, numeric matching, gap/assembly and output timings. Use `false` for headline totals. |
| `diagnostics_run` | Explicit `mode`, `workers` and zero-based `repeat` selecting completed trees for diagnostics; must belong to the suite. Diagnostic coverage is checked only when diagnostics is requested. |
| `quality_checks` | Exhaustive reference/sample metadata and called-allele reconstruction checks in the isolated trial, after the matching window. The trial timeout covers validation. |
| `quality_max_matrix_mib` | Positive decoded validation-call matrix budget in MiB, batching whole samples within physical chunks. Must fit one sample. Excludes reference/output trees, one decoded source tile, remapping buffers and decoder scratch. Splitting a chunk may reread tiles. |
| `resident_max_matrix_mib` | Maximum decoded sample matrix for resident trials. |
| `resident_max_private_mib` | Maximum summed private process-tree memory for resident trials. |

`from_reference` is the primary comparison: it times the deployed runner from
existing ancestor trees through finished outputs, including startup, packing,
loading, JIT cache loading/compilation, worker shutdown and output. Matching
workers and the loader stop before edge squashing and output finalisation. Both methods use identical
inputs, biological settings, CPU reservations and requested products.

`resident_match` excludes input decoding and HMM/JIT setup. Greedy includes
candidate discovery, exact extension, covering, gap matching and immediate
assembly; final table sorting/writing is outside. Resident trials preload all
sample slabs and therefore use the explicit matrix limit instead of the ordinary
slab bound. Native resident HMM retains results until assembly. Historical
resident greedy numbers excluded precomputed candidate/interval work and are
not equal-work comparisons.

Totals include process-tree CPU. Benchmark processes self-sample private USS;
the launcher sums their snapshots and reports mapped-input capacity separately,
avoiding multiplication of shared input pages by worker count. Sampling threads
add measurement overhead and can miss short peaks. Filesystem
caches are uncontrolled and the on-disk Numba cache is reused; fresh processes
do not imply cold caches. Diagnostic worker service times overlap and must not
be added to wall time. Performance improvements require measurement.
