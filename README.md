# tsinfer matching evaluation

A Snakemake pipeline for comparing sample matching against a full ancestor
reference with matching that uses focal ancestors where possible. It generates
panels, copying paths, and evaluation data; analysis is done separately.

## Setup and running

The pipeline uses an editable tsinfer checkout at `../tsinfer`, including its
native VCZ reader and regional matching support. Supply phased, biallelic,
single-contig VCF Zarr (VCZ) inputs; the pipeline does not create these inputs.

From this repository:

```sh
cp config.yaml.example config.yaml
# Edit config.yaml: set input paths, datasets, and the options below.
uv sync
uv run snakemake --dry-run --cores all
uv run snakemake --cores all
```

The default run processes every configured dataset and requested AC cutoff,
mismatch budget, and focal method. Performance evaluation is a separate target:

```sh
uv run snakemake performance --cores 9
```

Nine cores suffice for the example performance configuration. See
[CPU allocation](#cpu-allocation) when changing concurrency settings.
Append `--configfile path/to/config.yaml` to use another configuration.
Rule logs are written under `progress_dir`.

## How the pipeline works

A **haplotype** is one chromosome copy. An **ancestor panel** contains inferred
ancestral haplotypes; matching these ancestors produces the tree-sequence
**reference**. A copying path describes which reference node a sample haplotype
copies along the sequence.

1. **Prepare samples.** Copy the input VCZ and add derived allele counts (AC)
   and frequencies for the selected samples. AC counts chromosome copies carrying
   the non-ancestral allele; missing calls are excluded. Sites with AC 1
   (singletons) are excluded from inference and matching.
2. **Build the reference.** Infer an ancestor panel, then match the ancestors
   to form a reference tree sequence.
3. **Run the baseline.** Match each sample haplotype against the full reference
   using the HMM (hidden Markov model).
4. **Find focal candidates.** Choose one focal site, or **seed**, per ancestor.
   An ancestor is a candidate when the sample haplotype carries the derived allele
   at that seed. Discard ancestors above the largest requested annotated derived
   AC before loading seed calls or expanding candidates. Ancestors without a seed
   contribute nothing.
5. **Find coverage.** For eligible candidates, extend left and right from the
   seed while comparing sample and ancestor calls. Stop at the mismatch budget,
   missing data, ancestor support, or an inference-interval boundary. Union these
   intervals into a mask of **focal regions** and uncovered **gaps**.
6. **Match and stitch.** Match focal regions with a reduced-reference HMM or a
   greedy exact interval cover. Match gaps with the full-reference HMM, then join
   the paths and write outputs.

The reduced HMM retains eligible candidates and their internal ancestry in a
smaller reference. Greedy repeatedly copies the interval reaching furthest from
the next uncovered site, minimising switches within the saved exact intervals.
Regional matches start independently, so this does not guarantee the fewest
switches across gap boundaries. The methods can choose different paths.

Candidates, intervals, and masks do not require baseline sample matches.
Stitched matching requires the ancestor reference but can run independently of
baseline sample matching.

## Pipeline configuration

[config.yaml.example](config.yaml.example) is a complete starting configuration.
Paths are relative to the working directory; `~` is expanded for input paths
and the main output/log directories.

### Output, focal selection, and HMM options

| Option              | Example           | Meaning                                                                                                                                                                                                                                                                                                                         |
|---------------------|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `data_dir`          | `data/match_eval` | Root directory for ordinary pipeline products. Performance results have a separate fixed root.                                                                                                                                                                                                                                  |
| `progress_dir`      | `.progress`       | Directory for per-rule logs.                                                                                                                                                                                                                                                                                                    |
| `focal_choice`      | `left`            | Use each ancestor's leftmost focal site as its seed; `right` uses the rightmost. Defaults to `left` if omitted; `null` is invalid. Changing it can change candidates and coverage.                                                                                                                                              |
| `ac_cutoff`         | `[10]`            | Keep candidates whose seed's derived AC is **at most** this value. Lower values restrict focal matching to rarer alleles. Supply a nonempty list of distinct positive integers; each gets separate masks and stitched outputs. Performance passes follow this list's order. |
| `max_mismatches`    | `[0]`             | Mismatches allowed **on each side of the seed** during interval extension. `0` gives exact intervals; `1` allows one mismatch left and one right, excluding the next mismatch. Supply a nonempty list of strictly increasing nonnegative integers; each gets separate outputs. This controls coverage, not HMM mismatch counts. |
| `focal_methods`     | `[hmm, greedy]`   | Focal matching methods: `hmm` uses the reduced reference; `greedy` uses exact intervals. Choose either or both without duplicates. Greedy requires `max_mismatches: [0]`. Gaps always use the full HMM.                                                                                                                         |
| `hmm.recombination` | `0.01`            | Constant HMM recombination probability between successive inference sites. Larger values make switching copying states less costly. This is a probability, not a per-base simulation rate.                                                                                                                                      |
| `hmm.mismatch`      | `1.0e-20`         | HMM mismatch probability at a site. Larger values make copying a different allele less costly. The example strongly favours agreement but does not forbid mismatches.                                                                                                                                                           |

The `hmm` block supplies both probabilities to ancestor and sample matching,
including reduced HMM and gap matching. Both must be strictly between 0 and 1.
Greedy focal copying does not use these probabilities.

Candidates and intervals are prepared once at the **largest** AC cutoff;
intervals cover the largest mismatch budget, and smaller values reuse those
comparisons. Both archives record their maximum supported cutoff and reject
larger requests. For example, this produces six masks
and six stitched tree/JSONL pairs per dataset:

```yaml
ac_cutoff: [3, 5, 10]
max_mismatches: [0, 1]
focal_methods: [hmm]
```

### Datasets and sample selection

`datasets` is a list of input datasets. Each entry has these options; optional
keys can be omitted or set to `null` unless stated otherwise.

| Dataset option                 | Meaning                                                                                                                                                                                                                                            |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`                         | Unique label used in filenames and `performance.datasets`. Use lowercase names with underscores.                                                                                                                                                   |
| `zarr_path`                    | Required path to the input VCZ store.                                                                                                                                                                                                              |
| `ts_path`                      | Simulated truth tree sequence, used only for explicit true-ancestor targets. Use `null` for real data or when truth extraction is not needed.                                                                                                      |
| `ancestral_state`              | Required block identifying the ancestral allele using one of the next two options.                                                                                                                                                                 |
| `ancestral_state.field`        | VCZ array containing ancestral allele strings, such as `variant_ancestral_state` or `variant_ancestral_allele`.                                                                                                                                    |
| `ancestral_state.is_reference` | Alternative to `field`: set to `true` to treat REF as ancestral. Use only when that assumption is appropriate; do not also set `field`.                                                                                                            |
| `include`                      | Native site filter: retain sites satisfying it, e.g. `'POS >= 1000000 & POS < 11000000'`. Singleton exclusion is added automatically.                                                                                                              |
| `exclude`                      | Alternative to `include`: discard sites satisfying it. Singleton exclusion is added automatically. Set at most one of these filters; omit both to apply only singleton exclusion.                                                                  |
| `sample_list`                  | Headerless CSV of sample IDs, normally one per line. Selects the same cohort for AC annotation, inference, and matching. Omission or `null` selects all samples; both chromosome copies are retained.                                              |
| `matching_cache_size`          | Native reader cache size in MiB for ancestor, baseline, and stitched matching, including performance trials. Omission or `null` uses tsinfer's default. Must fit at least one input chunk; larger caches retain more data but consume more memory. |

Filter coordinates remain absolute; selecting a region does not shift it to
zero. The copied sample store retains the original site and sample axes;
readers apply selection. Its historical name is `{name}_samples_masked.zarr`,
although it is a count-annotated copy.

### Chunking and workers

| Option | Example | Meaning |
| --- | --- | --- |
| `ancestor_chunk_size` | `5000` | Ancestors decoded together, and ancestor-axis Zarr chunk size, during **true-ancestor extraction only**. Larger chunks increase temporary memory. Does not control inferred ancestor generation. |
| `inference_threads` | `1` | Requested workers for inferred ancestor generation. Snakemake may reduce this to fit `--cores`. |
| `matching_threads` | `1` | Requested workers for ordinary ancestor and baseline sample matching. Does not set stitched or performance concurrency. |
| `haplotype_batch_size` | `64` | Positive number of haplotypes submitted together in stitched matching. Bounds pending tasks and compact results; actual stitched workers cannot exceed it. Keep it at least as large as the requested workers when possible. |

Ordinary interval and stitched jobs request the workflow's available cores.
Interval workers are capped by physical ancestor-column blocks; stitched workers
by haplotype count and batch size. Batching bounds tasks/results, while final
output tables, input archives, and reader caches still grow with the workload.

## Performance evaluation

The `performance` block enables the `performance` target; benchmark trials are
not part of the default run. The target prepares dependencies, then runs
**one fresh trial process at a time**.

It requires a nonempty list of distinct positive `ac_cutoff` values,
`max_mismatches: [0]`, and one nonempty
present-day sample group per dataset. Enable `hmm` in `focal_methods` to benchmark
`reduced_hmm`, and `greedy` to benchmark `greedy`. Trials reuse the main dataset
settings, focal choice, HMM probabilities, reader cache, and batch size.

The complete suite runs once per cutoff, in configured order. With
`ac_cutoff: [3, 5, 7, 10]` and `performance.repeats: 3`, all datasets, modes,
methods (including full HMM), worker counts and three repetitions finish at
AC 3 before AC 5 starts, followed by AC 7 and AC 10. Resident trials share
maximum-cutoff preparation and filter it downstream. Each from-reference
stitched trial prepares only its own cutoff inside the measured window.

### What is measured

`performance.modes` chooses the timing window:

| Mode | Included in the measured total | Outside timing |
| --- | --- | --- |
| `resident_match` | Matching and collecting results, including matching workers, per-haplotype reference reduction, gap matching, and stitching. | Load all haplotypes into RAM, prepare shared inputs/indexes, and warm greedy compilation beforehand. Assemble and write outputs afterwards. |
| `from_reference` | Sample matching from an existing reference through completed tree/JSONL files: loading, first-use compilation, matching, assembly, and writing. Stitched methods also regenerate candidates, intervals, and masks in each trial. | Sample count annotation, ancestor inference, and ancestor matching. |

Resident mode isolates matching cost but retains all haplotypes and completed
results in memory. Use `from_reference` to assess the deployed pipeline on larger
workloads. A fresh process does not imply a cold filesystem cache.

`performance.methods` chooses the matching algorithm:

| Method | Meaning |
| --- | --- |
| `full_hmm` | Baseline: match each haplotype against the full reference. |
| `reduced_hmm` | The pipeline's `hmm` focal method, with full-HMM gap matches. |
| `greedy` | The pipeline's greedy exact focal cover, with full-HMM gap matches. |

### Every performance option

Values below are those in `config.yaml.example`.

| Option under `performance` | Example | Meaning |
| --- | --- | --- |
| `experiment_id` | `sample_matching_pilot` | Output folder name: nonempty, using only lowercase letters, digits, and underscores. Choose a new name for each experiment or changed configuration; an existing nonempty folder cannot be overwritten. |
| `datasets` | `[tgp_chr20_n100]` | Dataset names to benchmark, selected from the main `datasets`. Nonempty, without duplicates; may select a subset. |
| `modes` | `[resident_match, from_reference]` | Timing windows described above. Choose either or both without duplicates; they run in list order. |
| `methods` | `[full_hmm, reduced_hmm, greedy]` | Algorithms described above. Nonempty, without duplicates. Include `full_hmm` for quality comparisons against the baseline. |
| `worker_counts` | `[1, 6]` | Matching workers to try for each method: here, one-worker and six-worker trials. Positive integers without duplicates. Actual workers are capped by haplotype count and, for stitched methods, batch size. |
| `repeats` | `1` | Positive number of trials for every dataset/mode/method/worker combination **at each cutoff**. Does not count passes through the cutoff list. Start with one pilot; use e.g. `3` in a new experiment to assess variability. |
| `seed` | `20261014` | Integer random seed for shuffling method/worker trial order within each dataset, mode, and repeat. Makes ordering reproducible; does not randomise input data. |
| `read_workers` | `1` | Positive number of background threads loading genotype chunks, fixed while matching workers change. Resident loading happens before timing; from-reference reading is timed. |
| `rss_sample_interval_s` | `0.02` | Positive interval in seconds between samples of trial-process RAM usage (RSS): about every 20 ms here. Shorter intervals give finer observations with more sampling overhead; brief spikes can still be missed. |
| `trial_timeout_s` | `3600` | Positive wall-time limit in seconds for the **entire trial process**, including startup, preparation, and output outside the measured window. Here, one hour. Exceeding it kills the trial and records failure. |
| `collect_stages` | `true` | Collect stage timings alongside totals. `false` retains total measurements and workload counts but omits stage timers. Use a separate experiment with the same workload to assess timer overhead. |
| `quality_checks` | `true` | After trials, validate successful outputs and write `quality.json`: paths, mutations, identities, allele reconstruction, switches, mismatches, likelihoods, and parent agreement with a corresponding successful full-HMM trial when available. Outside trial timing/RSS, but adds suite runtime and memory use. `false` skips this. |
| `resident_max_matrix_mib` | `512` | Positive resident haplotype-matrix limit in MiB. Size is haplotypes × reference sites × one byte. Rejects a resident trial before loading if exceeded; excludes references, results, and other memory. |
| `resident_max_rss_mib` | `8192` | Positive limit in MiB on total resident trial-process RAM, including preparation and output. Sampled RSS above it kills the trial; timing-boundary samples are also checked. Separate from the matrix limit. Neither resident limit applies to `from_reference`. |
| `helper_threads` | See below | Supporting-library concurrency limits, separate from matching/read workers. These also affect CPU reservation. |

MiB means 1,048,576 bytes. Trial count is cutoffs × datasets × modes × methods ×
worker counts × repeats: the single-cutoff example runs **12 trials**.
Method/worker pairs are shuffled within each dataset/mode/repeat and run
sequentially; the random order restarts from the configured seed for each cutoff.

### Helper limits

All five keys belong inside `performance.helper_threads`. Each is a positive
integer; the example sets each to `1` to keep supporting concurrency fixed.

| Option | Meaning |
| --- | --- |
| `blosc_threads` | Blosc threads for decompressing/compressing array chunks. Raising it can speed codec work but adds CPU demand alongside readers and matchers. |
| `zarr_async_concurrency` | Maximum concurrent asynchronous Zarr I/O operations. An I/O limit, not a CPU-thread count; does not directly add to CPU reservation. |
| `zarr_threading_max_workers` | Maximum size of Zarr's helper thread pool. Raising it permits more supporting work and increases reserved CPUs. |
| `blas_threads` | Numerical-library thread limit, applied through BLAS/OpenMP environment settings before trial imports. Raising it can add parallelism inside numerical calls. |
| `numba_threads` | Thread limit for Numba parallel execution, applied before trial imports. Separate from matching workers; raising it does not parallelise a serial compiled function. |

### CPU allocation

The suite reserves the largest matching worker count, readers, a Blosc reserve
for readers and synchronous matching reads, the Zarr helper pool, and extra
BLAS/Numba threads. It refuses a smaller Snakemake allocation. With all helper
limits at `1`, the reservation is:

```text
max(worker_counts) + 2 × read_workers + 1
```

Thus `[1, 6]` with one reader needs `6 + 2 + 1 = 9` CPUs. For other settings,
print the exact requirement from your configuration:

```sh
uv run - <<'PY'
import pathlib
import yaml
from lib import benchmark

config = yaml.safe_load(pathlib.Path("config.yaml").read_text())
print(benchmark.required_allocation(config))
PY
```

Pass at least that number to `--cores`. Performance matching concurrency comes
from `worker_counts`, independently of `matching_threads` and visible host CPUs.
Keep batch size, cache size, and helper settings fixed when comparing workers.

### Results and interpretation

Each pass writes to `data/performance/{experiment_id}/ac{cutoff}/`, independently
of `data_dir`, including single-cutoff invocations. Each folder contains:

| Product | Purpose |
| --- | --- |
| `manifest.json` and saved configs/rosters/`uv.lock` | Record workload, sample identities, settings, source revisions, packages, machine, and allocation. |
| `runs.csv` | One row per trial: cutoff, status/error, wall and process CPU times, peak RSS, requested/actual workers, workload counts, and output paths. Failures have empty timing fields; subsequent trials still run. Run IDs and trial metadata include the cutoff. |
| `{run_id}/per_haplotype_worker_times.csv` | Reduced HMM and Greedy only, when `collect_stages: true`. One row per haplotype: `sample_id`, `ploidy_index`, and `service_s` (summed worker-stage elapsed seconds, including simplification/setup/stitching). Written after the timing window; join to `runs.csv` by the trial folder name for method, mode and repeat. |
| `stages.csv` | Stage breakdown. Worker service time/thread CPU are nested diagnostics, not additional total time; parallel service sums can exceed wall time. Contains only a header when stage collection is disabled. |
| `quality.json` | Separate scientific checks, when enabled. Different methods may legitimately choose different parents. |
| Each trial directory | Tree/JSONL outputs, specification/result JSON, `trial.log`, `native.log`, and `rss_samples.csv`. From-reference stitched trials also retain regenerated focal products. |

The experiment root receives `complete.json` only after every requested pass's
trials and enabled quality checks succeed. A failed suite exits with an error.

`window_peak_rss_bytes` covers the measured window; `trial_peak_rss_bytes`
covers the whole trial, including preparation and output. Both are sampled
maxima plus timing-boundary observations and can miss spikes. Resident window
RSS includes data loaded before timing. CPU time is `user_s + system_s`;
dividing by `wall_s` gives average CPU utilisation in cores. Compare successful
rows with identical workloads and timing modes, checking actual worker counts.
Do not add stage times to the measured total.

## Ordinary outputs and rerunning

Paths below are under `data_dir`, for dataset `{name}`. `{ac}`, `{k}`, and
`{method}` are a configured cutoff, mismatch budget, and focal method;
`{max_ac}` is the largest configured cutoff.

| Path | Contents |
| --- | --- |
| `samples/{name}_samples_masked.zarr/` | Input copy with selected-cohort derived AC/AF annotations. |
| `configs/{name}_ancestor_inference.toml` | Generated native tsinfer configuration. |
| `ancestors/{name}_inferred_ancestors.zarr/` | Inferred ancestor panel. |
| `ancestors/{name}_inferred_ancestors.trees` | Matched ancestor reference. |
| `focal_ancestors/{name}_inferred_max_ac{max_ac}_focal_ancestors.npz` | AC-filtered candidates and saved seeds for each haplotype, with the maximum supported cutoff. |
| `haplotype_intervals/{name}_inferred_max_ac{max_ac}_focal_ancestor_intervals.npz` | Candidate identities, seed AC, interval bounds for each budget, and maximum supported cutoff. |
| `focal_masks/{name}_inferred_ac{ac}_k{k}_focal_mask.npz` | Covered runs and gaps. |
| `matches/{name}_inferred_samples_raw.trees` and `…_samples_matches.jsonl` | Full-reference baseline outputs. |
| `matches/{name}_inferred_ac{ac}_k{k}_{method}_stitched_samples_raw.trees` and `…_stitched_samples_matches.jsonl` | Stitched outputs for each cutoff/budget/method. |

Trees are raw and unsimplified, preserving reference node IDs; path compression
is disabled. JSONL records contain haplotype identity, copying path segments
(`left`, `right`, `parent`), and mutations (`position`, `derived_state`). Path
coordinates are absolute, half-open base-pair bounds; mutation states are
canonical allele indices, while tree-sequence states are allele strings. Join
records by `(source, sample_id, ploidy_index)`. NPZ bounds are half-open indices
on the inference-site axis; load with `numpy.load(..., allow_pickle=False)`.

Snakemake reuses current products and rebuilds affected dependencies when tracked
inputs or rule parameters change. Library files are not tracked inputs: after
editing `lib`, force the affected rule, for example:

```sh
uv run snakemake --cores all --forcerun find_focal_ancestors make_ancestor_intervals
```

Likewise, force `make_focal_mask` after changing mask logic or
`match_stitched_samples` after changing stitched matching. Use a new performance
experiment ID to measure changed code or configuration.

True-ancestor extraction and matching require explicit file targets and a
non-null dataset `ts_path`. For example:

```sh
uv run snakemake --cores all data/match_eval/ancestors/out_of_africa_n300_1mbp_true_ancestors.zarr
uv run snakemake --cores all data/match_eval/matches/out_of_africa_n300_1mbp_true_samples_raw.trees
```

These produce a truth-derived panel on the inferred site axis, a
`dataframes/{name}_true_ancestors.csv` mapping, and full-reference matches with
`true` replacing `inferred` in filenames. There is no true-panel focal branch.

For implementation details, see [lib/haplotypes.py](lib/haplotypes.py),
[lib/matching.py](lib/matching.py), and [lib/benchmark.py](lib/benchmark.py).
[plans/performance_testing.md](plans/performance_testing.md) describes the detailed
performance measurement contracts.
