# tsinfer matching evaluation

This pipeline generates ancestor references and compares full-HMM sample matching
with greedy exact copying followed by HMM gap matching. Analysis belongs in
separate notebooks.

## Running

Use Python 3.12 or newer, phased biallelic single-contig VCF Zarr inputs, and
the sibling TSInfer and TSLS repositories. Both native dependencies are pinned
to committed local Git sources in `pyproject.toml` and `uv.lock`. Set those
source locations for your checkout before resolving dependencies.

```sh
mkdir -p configs
cp config.yaml.example configs/my_run.yaml
# Set input paths and configuration options.
uv sync --locked
uv run snakemake --configfile configs/my_run.yaml --dry-run --cores all
uv run snakemake --configfile configs/my_run.yaml --cores all
```

Personal configurations live in ignored `configs/`. Paths are relative to the
repository working directory; input paths, `data_dir`, `progress_dir` and
`temp_dir` expand `~`. Use fresh outputs when changing inputs or HMM settings.
VCZ arrays must have consistent variant chunks.

## Matching

1. Annotate derived allele counts and frequencies for the selected cohort.
   Missing calls do not enter counts. Exclude singletons from inference.
2. Infer and match ancestors to create a reference. Explicit `true` targets
   can extract true ancestors when a truth tree sequence is available.
3. Run an independent full-HMM sample baseline.
4. Prepare the focal ancestor catalog and supported calls under
   `data_dir/ancestors/focal/` once the ancestor tree is ready. Select each
   eligible ancestor's leftmost or rightmost focal seed, applying
   the selected-cohort derived-AF cutoff. Pack support clipped to its containing
   inference interval. A derived sample call at the seed activates the ancestor.
5. Extend exact matches, skipping missing observations and stopping at called
   disagreements, unknown reference calls, support ends or the called domain.
   Stable-sort intervals and greedily take the furthest reach. Equal reaches
   keep the first ancestor in stable panel order. Apply the minimum to whole
   touching focal unions, so short intervals can bridge retained unions.
   Then remove every whole touching focal union containing a site adjacent to
   a real zero-recombination transition, including unions extending into a flat
   block. Use exact zeros in the actual HMM probabilities and leave the map
   unchanged; the artificial first probability is excluded.
6. Partition copied spans and gaps within the domain and inference boundaries.
   Resolve supported one-site gaps outside zero-rate blocks separately: prefer
   the original left neighbour, then the original right neighbour, then the
   activated ancestor
   with furthest support end in stable support-start order. An unsupported
   fallback leaves an HMM gap. Correct called disagreements with mutations;
   missing observations need no correction. Zero-rate one-site pieces created
   by inference boundaries or the called domain remain HMM gaps.
7. Merge compatible copied spans and construct residual windows. Condition on
   adjacent copied parents with one flanking observation per available anchor.
   Inference boundaries prevent padding; anchors stay outside zero-rate blocks.
   `condition_gaps: false` removes pins and padding while retaining one-site copying.
8. Match all residual windows of one haplotype in one native batch, with
   independent likelihood resets. Clip paths to true gaps and assemble paths
   and mutations once. Append results, squash edges, sort and finalise output.

The catalog and packed calls are reused across minimum lengths, worker counts
and gap-conditioning settings. Updating the ancestor panel, reference tree or
cohort AF input triggers preparation. Each configured cutoff sweep shares one
payload. A match reads the prefix containing spans admitted by its cutoff.

The minimum in kb becomes
`max(1, round(kb * 1000 * num_sites / (last_position - first_position)))`
on the reference site axis. This uses mean site density. The domain spans the
first through last called site, or the full axis for an all-missing haplotype.
Greedy copying stays fixed; each residual HMM optimum is conditional on its
available anchors and native numerical settings.

The coordinator opens the shared genotypes and writable reference mapping.
Packing and worker setup finish before bounded sample loading starts; the
mapping is read-only during matching. One persistent matching thread pool owns
private HMMs and scratch. Loading continues while matching. Queue and live-slab
limits are separate, and temporary mappings close after workers stop.
Only the coordinator appends output tables.

## Configuration

`af_cutoff` and `min_region_size_kb` accept scalars or lists on each dataset.
Ordinary matching runs their Cartesian product and shares one full-HMM baseline.

| Option | Meaning |
| --- | --- |
| `config_name` | Lowercase experiment label using letters, digits and underscores. Distinct names keep diagnostic tables and manifests separate. |
| `data_dir` | Ordinary output root. Use a separate directory for each biological configuration. |
| `progress_dir` | Per-rule log root. |
| `focal_choice` | `left` (default) or `right` endpoint focal seed. |
| `condition_gaps` | Default `true`; condition residual greedy HMM gaps on adjacent copied parents. `false` removes conditioning and flanking observation padding. One-site-gap copying is unchanged. |
| `hmm.recombination` | Constant HMM recombination probability between inference sites, strictly between 0 and 1. |
| `hmm.recombination_rate` | Alternative to `hmm.recombination`: a nonnegative rate per bp per generation, or a HapMap file path. tsinfer converts genetic distances between inference sites to probabilities using Haldane's map function. Choose exactly one recombination setting. |
| `hmm.mismatch` | Sample mismatch probability, strictly between 0 and 1. Mutually exclusive with `hmm.mismatch_epsilon`. Exact extension stops at called disagreements; one-site-gap copies can require sample mutations. |
| `hmm.mismatch_epsilon` | Alternative per-haploid-base epsilon. Resolve sample mismatch as `epsilon * mismatch_span_bp / selected_source_sites`, before singleton exclusion. |
| `hmm.mismatch_span_bp` | Positive calibration span in bp, required with epsilon. Set explicitly for the simulated/selected interval rather than using whole-contig length. |
| `hmm.likelihood_threshold` | Optional sample likelihood floor; omitted/null retains the native default. Zero disables flooring. |
| `hmm.prune_threshold` | Optional Viterbi pruning cutoff; omitted/null uses zero (disabled). |
| `hmm.relative_prune_threshold` | Optional relative pruning cutoff; omitted/null uses zero. All numerical settings pass through unchanged with gap conditioning. |
| `ancestor_chunk_size` | Physical ancestor-column chunk size for explicit true-ancestor extraction only. |
| `inference_threads` | Native ancestor inference workers and requested diagnostics CPU budget. Diagnostics uses the actual Snakemake-granted threads. |
| `matching_threads` | Native ancestor/baseline threads and greedy matching threads. |
| `sample_reader_threads` | Greedy physical variant-block readers per slab; default `1`. Blocks write disjoint columns and publish together. Allow for decoder scratch. Shared Zarr async and thread limits remain separate from reader count and are included in the CPU budget. |
| `sample_prefetch_chunks` | Maximum live decoded sample slabs; start with `2`. |
| `max_pending_haplotypes` | Maximum queued, running and completed-but-unappended haplotype jobs; also bounds preparation admission. `null` uses the requested worker count. An explicit limit must be at least that count. |
| `temp_dir` | Temporary mapped-file parent, or `null` for the system temporary directory. |
| `helper_threads.blosc_threads` | Codec threads per process; start with `1`. |
| `helper_threads.zarr_async_concurrency` | Maximum concurrent Zarr requests within an array read. |
| `helper_threads.zarr_threading_max_workers` | Shared I/O/decode pool per process. Multiple workers permit parallel chunk reads; the pool is included in the CPU reservation. |
| `helper_threads.blas_threads` | Numerical-library thread limit; start with `1`. Numba kernels are serial. |

Sample mismatch calibration uses selected source sites before singleton
exclusion and an explicit positive finite span. Choose exactly one of mismatch
or epsilon, with probabilities strictly between zero and one. Ancestor matching
uses stock probabilities without sample maps or numerical overrides. Both
stages use the selected haploid cohort size.

HapMap files have a header and chromosome, position (bp), rate (cM/Mb), and
optional cumulative position (cM) columns. Three-column maps are integrated
using each row's rate until the next row. Native configuration writes a
four-column map beside the TOML. Coordinates stay absolute under site selection.
Set the unused recombination or mismatch choice to `null` when overlaying YAML.

Snakemake reserves matching workers plus loader and decoder helpers. With all
helper limits at one, greedy reserves `max(2 * workers, workers + 3)` CPUs.
Worker counts are never silently reduced.

Each `datasets` entry has:

| Option | Meaning |
| --- | --- |
| `name` | Unique lowercase label with underscores, used in filenames. |
| `af_cutoff` | Inclusive maximum selected-cohort derived AF at the seed, between `0` and `1`; a scalar or nonempty list of distinct values. Missing calls do not enter the denominator. |
| `min_region_size_kb` | Finite nonnegative number or nonempty list of distinct values; requested minimum connected focal-region size in kb. Converted to a rounded mean inference-site count over the first-to-last reference-site span, with a minimum of one site. `0` keeps all regions. |
| `zarr_path` | Input VCZ store. |
| `ts_path` | Optional truth tree sequence, used for explicit true-ancestor extraction, configured population joins and diagnostic provenance; otherwise `null`. |
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
| `matching_cache_size` | Ancestor reader cache in MiB; `null` uses tsinfer's default. Both sample runners use bounded decoded slabs. |

Products beneath `data_dir` include annotated `samples/`, native TOML in
`configs/`, ancestor panels/references in `ancestors/`, and raw sample trees
in `matches/`. Greedy outputs use
`matches/af_{af_cutoff}/min_region_kb_{min_region_size_kb}/`.
Their `match_eval` metadata records the requested minimum and resolved site count.

## Diagnostics

```sh
uv run snakemake diagnostics --configfile configs/my_run.yaml --cores all
```

The default target includes diagnostics. Diagnostics reconstructs the same copying mask without
running HMM matching, then compares saved paths. With `performance: null`, it
uses ordinary matching outputs. Otherwise `performance.diagnostics_run` selects
completed trial trees; every diagnostic dataset and setting must have both
methods in the selected mode, worker count and repeat.

Outputs beneath `diagnostics/{config_name}/` include `haplotype_results.csv`,
`region_lengths.csv`, `run_summary.csv`, `sample_populations.csv`, shared site
axes, effective configuration and a provenance manifest. Masks in `focal_masks/`
contain canonical haplotype identities, ragged half-open site-index intervals,
`is_gap`, domain bounds and matching parameters. Load NPZ files with
`allow_pickle=False`. Site comparisons require parents in both paths; excluded
inference intervals break switch adjacency.

The manifest records the original paths of the trees, tables and other inputs.

## Performance and regional profiling

```sh
uv run snakemake performance --configfile configs/my_run.yaml --cores all
# Or measure already prepared references:
uv run -m lib.benchmark suite configs/my_run.yaml --allocation 12
# Replay a bounded selection of regional windows:
uv run -m lib.profile_regions configs/my_run.yaml example_dataset data/regional_replay
```

Performance is opt-in; `performance: null` disables it. References are prepared
before fresh trial processes run sequentially. Each experiment directory must
be new. Results under `data/performance/{experiment_id}/` contain manifests,
matched trees, logs, workload/stage CSVs and memory observations.
Benchmark datasets require a scalar minimum and may have multiple AF cutoffs.
Full HMM runs once per dataset/mode/workers/repeat, shared across cutoffs.

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
| `collect_haplotype_timings` | Default `false`; save task service times independently of regional stage timers. Available for both matching methods. |
| `collect_stages` | Enable aggregate preparation, coordinator, loading, native gap matching and assembly timings. `false` retains workload counts without stage timers. |
| `diagnostics_run` | Explicit `mode`, `workers` and zero-based `repeat` selecting completed trees for diagnostics; must belong to the suite. Diagnostic coverage is checked only when diagnostics is requested. |
| `quality_checks` | Exhaustive reference/sample metadata and called-allele reconstruction checks in the isolated trial, after the matching window. The trial timeout covers validation. |
| `quality_max_matrix_mib` | Positive decoded validation-call matrix budget in MiB, batching whole samples within physical chunks. Must fit one sample. Excludes reference/output trees, one decoded source tile, remapping buffers and decoder scratch. Splitting a chunk may reread tiles. |
| `resident_max_matrix_mib` | Maximum decoded sample matrix for resident trials. |
| `resident_max_private_mib` | Maximum summed private process-tree memory for resident trials. |

`regional_profile.haplotypes` bounds replay selection;
`regional_profile.overlap_probe_sites` sets additional overlapping probe widths.
Replay records windows, scores, matched alleles, rates, queries and a Python profile.

`from_reference` includes preparation, packing, sample loading, HMM/JIT setup,
matching, shutdown and final output. `resident_match` preloads samples and
excludes decoding/setup and final table writing; greedy still includes
candidate discovery, covering, bridges, HMM matching and immediate assembly.
Validation runs after the measured matching window.

Coordinator stages measure elapsed phase windows. Worker stages report summed
service time and executing-thread CPU, which overlap coordinator time.
`gap_matching_assembly` includes interval preparation, the regional batch and
final assembly; do not add it to its children. `gap_calls` counts logical
windows and `hmm_batches` counts native calls. Optional
`haplotype_timings.csv` records identities, workers, call start/end offsets,
service/CPU times and workload counts.

Greedy workers share a packed ancestor reference and decoded sample slabs in
memory. Full HMM sample slabs and process-based diagnostics use mapped files.
Private process-tree memory is sampled separately from mapped- and resident-input
capacities. Traceback allocator
bytes measure that allocator alone. Sampling can miss short peaks; filesystem
and Numba caches are reused.

## Recovery

```sh
uv run -m lib.benchmark resume data/performance/my_run \
    --allocation 12 --backup-dir data/performance/my_run_failed_trials
```

Recovery reuses successful trials and retries failed ones with saved settings.
The new backup directory must be outside the experiment. It preserves failed
files and records current provenance before rebuilding summaries and completion
markers. If Snakemake retains incomplete metadata, clear it for the saved suite
files before requesting diagnostics with that suite's effective configuration.
