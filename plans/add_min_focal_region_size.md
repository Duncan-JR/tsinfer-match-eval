# Add `min_focal_region_size`

## Contract and scope

Add a required top-level list, `min_focal_region_size: [1]`, alongside the
existing `ac_cutoff` list (do not rename `ac_cutoff` to `max_ac`). For each
haplotype, AC cutoff and mismatch budget, first union eligible ancestor
intervals, merging overlaps and touching bounds as today. Retain a resulting
focal region iff `right_site_index - left_site_index >= min_focal_region_size`.
Measure inference sites, not base pairs or individual ancestor intervals.
Removed regions become ordinary full-HMM gaps; rebuild the complement so
adjacent gaps coalesce. A minimum of 1 is the unfiltered control.

Generate the full Cartesian product of AC cutoffs, minimum sizes, mismatch
budgets and focal methods. Keep candidate discovery, interval extension,
reference construction and matching algorithms unchanged. Assume valid
configuration: a nonempty list of distinct positive integer minimum sizes.
Add no compatibility paths, fallback defaults or new validation framework.

## Implementation

1. **Filter masks in `lib/haplotypes.py`.** Add a scalar
   `min_focal_region_size` argument to `make_focal_mask` and `_write_focal_mask`,
   and a list argument to `make_focal_masks`. Extend the existing batch product
   to `(ac_cutoffs, max_mismatches, min_focal_region_sizes)`, in that order, and
   pass the scalar to each writer. In `_write_focal_mask`, filter `covered`
   **after merging and before emitting focal/gap runs**. Reuse the existing
   cursor/complement code, including its all-gap case. Save the scalar minimum
   in each mask NPZ and update these functions' interlinked docstrings.

2. **Expand ordinary workflow outputs in `Snakefile`.** Read the required list
   and use `{min_size}` in minimum-dependent paths:
   - Masks: `{name}_inferred_ac{ac}_k{k}_min{min_size}_focal_mask.npz`.
   - Stitched products:
     `{name}_inferred_ac{ac}_k{k}_min{min_size}_{method}_stitched_samples_{raw.trees,matches.jsonl}`.
   Update `rule all`, mask batch outputs/arguments, stitched mask inputs,
   stitched outputs, wildcard constraints and stitched log names. Match mask
   expansion order to the batch product above. Each stitched job handles one
   `(dataset, ac, min_size, method)` across the existing mismatch-budget list.
   **Never pass multiple minimum sizes to one matching invocation:** masks are
   indexed by mismatch budget inside matching. Keep `lib/matching.py` and its
   API unchanged; it already follows the supplied masks for both HMM and greedy.
   Candidate/interval paths and their maximum-AC sharing remain unchanged.

3. **Extend the existing benchmark passes in `lib/benchmark.py`.**
   - Iterate `itertools.product(config["ac_cutoff"],
     config["min_focal_region_size"])`, preserving configured order, AC first.
     Each pass config contains one-element lists for both parameters. Finish
     every dataset/mode/method/worker/repeat for a pair before the next pair;
     keep the existing sequential subprocess launcher and full-HMM baseline
     in every pass. `performance.repeats` applies independently to each pair.
   - Use `data/performance/{experiment_id}/ac{ac}_min{min_size}/`, including
     single-pair runs. Update expected-pass-directory handling and Snakemake's
     declared performance outputs and resident-mask dependencies accordingly.
   - Update `prepared_inputs` to select the minimum-specific resident mask.
     In `_focal_preparation`, pass the selected scalar minimum to
     `make_focal_mask` and include it in the local mask name. Resident masks
     stay outside timing; from-reference preparation/filtering stays inside.
     Continue sharing maximum-AC candidate/interval inputs for resident runs.
   - Add `min_focal_region_size` to `Run`, run IDs, trial metadata and quality
     records (success and failure). Extend the existing single-pass trial
     consistency check and full-HMM baseline association to include the minimum.
     Saved pass configs/manifests must identify both settings and the correct
     mask. Record both parameter lists in root `complete.json`, publishing it
     only after every pair succeeds. Keep the existing benchmark restriction
     `max_mismatches: [0]`.

4. **Configuration and documentation.** Add `[1]` to `config.yaml.example` and
   the active local `config.yaml` (gitignored). Update the README's option table, grid example,
   ordinary output paths, benchmark pass/repeat description and output layout.
   Explain merge-before-filter semantics and the unfiltered control briefly;
   keep experiment findings out of the README. Do not modify notebooks or
   unrelated/untracked configuration files.

## Verification after implementation

Use `uv run` and small ad hoc checks; do not add a test suite. Check that
minimum 1 preserves existing mask arrays and scientific matching outputs;
exact-threshold regions survive; touching short intervals whose union reaches
the threshold survive; removing a middle focal region merges its surrounding
gaps; and removing every focal region produces one full-axis gap per haplotype.
Keep identities, offsets and complete axis coverage intact. Check HMM budgets
0/1 and greedy at budget 0 against the same filtered masks.

Dry-run a two-AC/two-minimum workflow and run a small benchmark with both axes
and multiple repeats. Expect four distinct pass directories and
`n_ac * n_min * n_datasets * n_modes * n_methods * n_workers * repeats` trial
rows. Verify pair order, unique outputs/logs, correct resident and local masks,
quality-baseline associations and completion handling. Compare each pass with
a standalone invocation of that pair; use existing allele reconstruction and
TS/JSONL consistency checks. Do not require different minimums to choose the
same copying paths.

## Initial experiment

Review of [pt3](../../dphil-analysis/notebooks/ch5_stitching_pt3_hyperparams.py)
and [pt4](../../dphil-analysis/notebooks/ch5_stitching_pt4_perf.py), including
their saved tables, supports trying `min_focal_region_size: [1, 50, 100, 200]`
with AC 10, 20 and 40, initially using exact intervals. On pt3's 10 Mb datasets,
these non-control minima remove approximately 0.08–1.09% of existing focal
base-pair coverage in TGP and 0.18–3.02% in simulation across those ACs. A minimum
of 500 loses roughly 13–14% in simulation, so defer that larger cutoff.

Higher AC improves simulated truth ARF while modestly reducing truth TPR;
agreement with Full HMM also decreases. Pt4 measures AC 3 only: resident greedy
is faster, but both stitched methods are slower from reference. Neither saved
notebook measures the effect of this size filter. Treat speed and accuracy
improvements as hypotheses to evaluate with the grid, not established results.
