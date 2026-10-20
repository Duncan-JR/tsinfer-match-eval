# tsinfer matching evaluation

This Snakemake pipeline generates ancestor references and compares full-HMM
sample matching with greedy exact copying plus full-HMM gap matching. Analysis
belongs in separate notebooks.

## Running

Use phased, biallelic, single-contig VCF Zarr inputs and the editable tsinfer
checkout at `../tsinfer`. The runner uses its regional `AncestorMatcher`,
`reset_topology()` and raw table APIs directly.

```sh
cp config.yaml.example config.yaml
# Set input paths and the options below.
uv sync
uv run snakemake --dry-run --cores all
uv run snakemake --cores all
```

`af_cutoff` and `min_focal_region_size` accept a scalar or a list. The default
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
   its ancestor; extend exactly in both directions until mismatch, missing call,
   support end or inference boundary. Different ancestors may share a seed.
6. Stable-sort intervals by left bound and repeatedly copy the interval reaching
   furthest from the current cursor. Filter complete connected unions by their
   inference-site count, including touching intervals. Short intervals can bridge
   a retained union. Equal reaches keep the first interval in stable panel order.
7. Match uncovered gaps independently with the full HMM, splitting calls at
   inference boundaries. Merge touching final runs with equal parents, append
   completed results immediately, then finalise tables once.

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
| `data_dir` | Ordinary output root. Use a separate directory for each biological configuration. |
| `progress_dir` | Per-rule log root. |
| `focal_choice` | `left` (default) or `right` endpoint focal seed. |
| `af_cutoff` | Inclusive maximum selected-cohort derived AF at the seed, between `0` and `1`; a scalar or nonempty list of distinct values. Missing calls do not enter the denominator. |
| `min_focal_region_size` | Positive integer or nonempty list of distinct positive integers; minimum inference sites in a complete touching union. `1` keeps all. These are site counts, not base pairs. |
| `hmm.recombination` | Constant HMM recombination probability between inference sites, strictly between 0 and 1. |
| `hmm.mismatch` | HMM mismatch probability, strictly between 0 and 1. Exact focal copying always allows zero mismatches. |
| `path_output` | Write optional native-shaped JSONL paths/mutations for both runners. Greedy records may be in completion order; use `haplotype_index` for canonical order. |
| `ancestor_chunk_size` | Physical ancestor-column chunk size for explicit true-ancestor extraction only. |
| `inference_threads` | Native ancestor inference workers. |
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
| `zarr_path` | Input VCZ store. |
| `ts_path` | Truth tree sequence for explicit true-ancestor extraction; otherwise `null`. |
| `ancestral_state.field` | Array of ancestral allele strings. |
| `ancestral_state.is_reference` | Alternative to `field`: `true` treats REF as ancestral. |
| `include` | Optional native site filter, e.g. `'POS >= 1000000 & POS < 11000000'`. |
| `exclude` | Alternative to `include`. Singleton exclusion is added automatically. |
| `sample_list` | Optional headerless CSV of sample IDs. The same cohort is used for annotation, inference and matching; both ploidies are retained. |
| `matching_cache_size` | Native ancestor/full-HMM reader cache in MiB; `null` uses tsinfer's default. Greedy uses its slab bound instead. |

Coordinates remain absolute when selecting a region. Main products are:

- `samples/{name}_samples_masked.zarr`: count-annotated sample copy.
- `configs/{name}_ancestor_inference.toml`: native inference/matching config.
- `ancestors/{name}_inferred_ancestors.zarr` and `.trees`: panel and raw reference.
- `matches/{name}_inferred_samples_raw.trees`: native full-HMM baseline.
- `matches/af_{af_cutoff}/min_region_{min_focal_region_size}/{name}_inferred_greedy_samples_raw.trees`: greedy stitched samples for each parameter pair.
- Corresponding `*_samples_matches.jsonl` when `path_output` is enabled.

`config.yaml.hpc_params1` selects the 10 Mb simulated n=300 and n=3000 cohorts
from `../tsinfer-anc-eval/config.yaml.matching`. It runs AF cutoffs
`[0.01, 0.03, 0.05]` and minimum sizes `[1, 100, 300, 1000]`: 24 greedy outputs
plus two full-HMM baselines. With `TMPDIR` set to node-local scratch inside the
196-CPU, 1024-GB allocation, run:

```sh
uv run snakemake all --configfile config.yaml.hpc_params1 --cores 196
```

It uses 48 matching workers per job, allowing two greedy jobs to run concurrently
within their helper-inclusive CPU reservations. `performance: null` disables the
benchmark target, including any performance settings inherited from `config.yaml`.
Ordinary matching does not collect benchmark timings or memory samples.

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
process-tree memory samples and matched trees.
Performance trials require scalar `af_cutoff` and `min_focal_region_size` settings.

Worker counts are never silently reduced. Every dataset must have at least as
many haplotypes as the largest requested worker count. The suite checks its CPU
reservation against the process's effective CPU affinity on Linux. Its manifest
records host CPU/memory information, allowed CPU/core/socket/NUMA topology and
selected scheduler allocation and binding variables.

For the five simulated cohorts in `../tsinfer-anc-eval/config.yaml.matching`,
`config.yaml.hpc_perf1` runs both methods once at 32 workers, using identical references
and settings. Generate just their inputs from the `tsinfer-anc-eval` directory:

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
| `quality_checks` | Untimed metadata, called-allele reconstruction and optional tree/JSONL agreement checks. |
| `resident_max_matrix_mib` | Maximum decoded sample matrix for resident trials. |
| `resident_max_private_mib` | Maximum summed private process-tree memory for resident trials. |

`from_reference` is the primary comparison: it times the deployed runner from
existing ancestor trees through finished outputs, including startup, packing,
loading, JIT cache loading/compilation and output. Both methods use identical
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
