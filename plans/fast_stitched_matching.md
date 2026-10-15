# Fast stitched matching: implementation plan

Repository: `tsinfer-match-eval`. Implement the following three commits in order. Each commit includes focused agent-run correctness checks and local before/after timings using existing checks and small ad hoc scripts.

## Commit 1: `perf: filter focal candidates by AC before expansion`

**Files:** `lib/haplotypes.py`, `Snakefile`, preparation/path handling in `lib/benchmark.py`.

Add an optional `max_ac_cutoff` argument to `find_focal_ancestors`. Apply the existing annotated derived-AC eligibility test to anchored ancestors **before** computing unique seed sites, loading seed genotypes, or expanding per-haplotype candidate lists. Use the existing AC annotations rather than recomputing AC from the selected sample cohort.

Thread the limit through candidate discovery and interval preparation. For shared workflow preparation, use `max(ac_cutoff)` and filter to each requested cutoff downstream. For an individual `from_reference` benchmark trial, prepare at that trial's cutoff inside its preparation window.

Give filtered candidate archives and dependent interval products cutoff-scoped paths and metadata recording their maximum supported cutoff. Reject requests exceeding that maximum. Wire matching dependencies directly to these products so producing them does not also require building the unrestricted candidate archive. Include the maximum cutoff in Snakemake dependency/path handling to prevent stale products being reused after configuration changes.

**Agent checks:** Compare candidates with the eligible projection of the old output, then compare downstream intervals, masks and matched paths at the same cutoff. Cover empty candidate sets, shared seed sites, missing seed observations and AC at the threshold. Record candidate/preparation time, selected seed count, association count and archive size before and after.

## Commit 2: `perf: polarise sample calls with numeric allele lookup`

**Files:** `_load_sample_haplotypes` in `lib/haplotypes.py`.

Replace the genotype-to-allele-string expansion with a small per-site allele-index lookup containing ancestral/derived codes. Compare each site's allele table with its ancestral allele once, then gather numeric codes directly using the genotype indices. This removes the per-genotype string temporary.

Handle missing genotype indices explicitly before gathering and restore the missing-call code afterwards. Do not clamp invalid positive indices into range. Return the required numeric dtype and shape directly, retaining site, sample and ploidy ordering. Apply the same implementation to seed-only and full-site preparation loads.

**Agent checks:** Compare values, dtype and shape with the old helper for REF- and ALT-ancestral sites, multiallelic sites, missing calls, reordered samples and empty selections. Check downstream preparation output equivalence. Time conversion on already-loaded arrays and the complete loader, and compare peak memory use.

## Commit 3: `feat(perf): run all configured AC cutoffs sequentially`

**Files:** `lib/benchmark.py`, `Snakefile`, relevant README/configuration documentation.

Use the existing top-level `ac_cutoff` list as the sequence of benchmark passes. Validate that it is nonempty and contains valid, distinct cutoffs; preserve the configured order.

Implement a thin outer loop around the single-cutoff benchmark runner. For each cutoff, derive a configuration containing only that cutoff and execute the complete configured suite. Finish all datasets, modes, methods, worker counts and repetitions for that cutoff before starting the next. `performance.repeats` applies independently to every cutoff; it does not count passes through the cutoff list.

For example, `ac_cutoff: [3, 5, 7, 10]` with `performance.repeats: 3` runs the complete three-repeat suite at AC 3, then AC 5, AC 7 and AC 10. Methods selected in the configuration, including full HMM, run in each pass. Reuse the existing trial subprocess runner rather than introducing another scheduler.

Put each pass in `data/performance/<experiment_id>/ac<cutoff>/`, with its own configuration, manifest, `runs.csv`, `stages.csv`, `quality.json` and trial outputs. Include the cutoff in run identifiers and run metadata. Use the same directory structure for a single-cutoff invocation. This lets each pass remain an ordinary single-cutoff result set without mixing records or overwriting outputs.

Update Snakemake input/output expansion to cover every requested cutoff and select the correct candidate, interval and mask products for each pass. Shared resident-input preparation can use the maximum-cutoff products from commit 1; each `from_reference` trial must perform its own cutoff-specific preparation within its measured window. Mark the overall benchmark complete only after every requested pass succeeds. Document the sequence, output layout and relationship between cutoffs and repetitions.

**Agent checks:** Run a small two-cutoff, multi-repeat example and verify execution order, expected trial count, cutoff-specific inputs, unique outputs and correct quality-record associations. Compare each pass's scientific outputs with a standalone run at that cutoff, and check single-cutoff execution and invalid-list handling.
