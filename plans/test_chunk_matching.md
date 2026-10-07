# Test stitching focal matches and gap matches

## Aim

Compare two results for each sample haplotype:

- **Stitched match:** keep perfect focal-ancestor pieces from a whole-sequence
  match against a simplified reference, then fill every remaining chunk by
  matching against the full reference.
- **Full match:** match the same haplotype against the original full reference
  over the whole sequence.

The reduced match is an intermediate used to construct the stitched match.
It is not a third result to compare or a separate output file. The question is
whether stitching reproduces the full HMM's path, mutations, and path score.
There is no timing, profiling, or performance comparison at this stage.

This follows the [1 October supervisor meeting](/Users/duncan/exo/documents/meeting_notes/2026-10-01_gil_yan_jerome.md): first try independent matching of gaps and inspect the result, before adding boundary conditioning.

This document describes subsequent implementation. For now, change only this
plan; do not create code, configuration, or data products.

## Locations and inputs

Put the standalone analysis in:

```text
~/work/dphil-analysis/experiments/ch5/
    test_chunk_matching.py
    chunk_matching_config.yaml
    data/
        match_eval/                    # existing input organisation
        chunk_matching_experiment/     # results, grouped by dataset
```

Use existing products from `~/work/tsinfer-match-eval/data/match_eval`. Place
copies of the selected datasets' required products in the local
`experiments/ch5/data/match_eval` tree, preserving their relative organisation
and leaving the originals unchanged. No input generation or pipeline changes
are needed. Prepare these copies when implementing the experiment, not during
this plan revision.

For dataset `{name}`, the required products are:

| Relative path beneath `data/match_eval` | Purpose |
| --- | --- |
| `ancestors/{name}_inferred_ancestors.trees` | Original full ancestor reference |
| `ancestors/{name}_inferred_ancestors.zarr` | Panel IDs, alleles, and ancestor support |
| `samples/{name}_samples_masked.zarr` | Observed sample genotypes |
| `focal_ancestors/{name}_inferred_focal_ancestors.npz` | Per-haplotype focal candidates and AC |
| `configs/{name}_ancestor_inference.toml` | Sample selection and HMM parameters |

The copied TOML can contain absolute paths to the original products. Read its
sample-selection and parameter values, but open inputs using the local paths
above. Use the reference's ancestral states to polarize sample calls. Do not
invoke `pipeline.match` with the copied TOML or follow its output paths.

The chunk CSVs and existing matched sample tree sequences are not required.
Run a fresh full match with the same observations and parameters as the
stitched calculation, so the comparison has a consistent baseline.

## Configuration and execution

Keep experiment settings in the separate YAML. A small initial run is:

```yaml
input_dir: data/match_eval
output_dir: data/chunk_matching_experiment
ac_cutoff: 10
workers: 1
score_tolerance: 1.0e-8
datasets:
  - name: out_of_africa_n300_1mbp
    haplotypes:
      - sample_id: tsk_0
        ploidy_index: 0
      - sample_id: tsk_0
        ploidy_index: 1
```

Resolve relative input/output paths against the YAML's directory. Expand `~`
using `pathlib.Path`. `haplotypes: null` selects all focal-NPZ haplotypes in
their stored order. `workers: null` uses available CPU cores, capped by the
number of selected haplotypes; compute that default at process launch.

Read `recombination` and `mismatch` from the TOML's
`match.sources.samples`. Use the same positive probabilities, allele counts,
and default engine settings for reduced, gap, and full matches. The current
tiny nonzero mismatch probability allows the whole-sequence reduced match to
pass through regions where no exact match exists. Zero mismatch probability
can raise `MatchImpossible` instead of returning gaps.

Use the existing tsinfer-match-eval environment, which includes the editable
tsinfer checkout and required dependencies, to run the future analysis:

```sh
uv run --project ~/work/tsinfer-match-eval \
  ~/work/dphil-analysis/experiments/ch5/test_chunk_matching.py \
  --config ~/work/dphil-analysis/experiments/ch5/chunk_matching_config.yaml
```

Keep the implementation self-contained in the analysis script, following
[dphil-analysis's AGENTS.md](/Users/duncan/work/dphil-analysis/AGENTS.md).
No Snakemake rules, production-library edits, or new test suite are needed.

## Prepare shared inputs

Load the original reference, panel metadata, and NPZ once per worker. Use
`tskit.load`, read-only `tsinfer.vcz.open_store`, and
`numpy.load(..., allow_pickle=False)`. Start with the existing inferred,
biallelic, uncompressed references with one inference interval.

Use the reference's site positions as the common axis and require them to
equal the panel's positions. Locate these positions in the sample store with
`searchsorted`, checking exact equality. Resolve the TOML's sample selection
against the local sample store and read the selected observed haplotypes.
Encode them as ancestral `0`, derived `1`, missing `-1`, using allele strings
and the reference's ancestral states. Preserve missing calls and check that
the panel's allele order agrees. Do not obtain observations from a previously
matched sample TS, where missing calls may have been imputed.

Join the NPZ's panel `ancestor_id` strings to original reference nodes using
node metadata `(source="ancestors", sample_id, ploidy_index=0)`. Require a
unique join. NPZ ancestor indices are Zarr panel columns, not TS node IDs.
Also retain each panel ancestor's `sample_start_position` and
`sample_end_position` for the acceptance check.

Build full-reference matcher indexes once per worker and reuse them. All HMM
calls use full-length arrays on this original inference-site axis. Here
"whole sequence" means every inference site, `[0, S)`, including neutral
missing calls. Both the reduced pass and full baseline use exactly this range.

## One complete job per haplotype

Each job performs the following stages sequentially for one haplotype:

```text
select focal ancestors
    → simplify reference
    → immediately match the whole haplotype against that reduced reference
    → extract accepted pieces and every gap
    → match the gaps against the full reference
    → stitch
    → run the whole-sequence full match and compare
```

Do not create all sample-specific references first and match them in a second
pass. A worker constructs only its current haplotype's reduced reference,
uses it immediately, and releases it after extracting the accepted pieces.
If processing several haplotypes in parallel, queue complete haplotype jobs
using multiprocessing; the simplification and matching stages remain together
inside each job. Process that haplotype's gaps sequentially in the same worker.

### 1. Simplify and immediately match

For NPZ haplotype row `r`, its candidate panel columns are:

```text
ancestor_index[offsets[r]:offsets[r + 1]]
```

Keep columns whose panel-aligned `derived_ac <= ac_cutoff`, then map them to
original focal-node IDs. Simplify the full reference to these nodes, retaining
synthetic roots 0 and 1 first:

```python
retained_nodes = [0, 1, *focal_nodes]
reduced_ts, old_to_new = reference_ts.simplify(
    retained_nodes, map_nodes=True, filter_sites=False
)
```

Those two roots and their order are required by the current raw-reference
engine. Confirm that they and all selected focal nodes survive, and that
positions and ancestral states remain unchanged. Create reduced indexes with
`vestigial_root=False`, since the required roots are already present.

Immediately use `tsinfer.matching.AncestorMatcher` to match the sample
haplotype against `reduced_ts` over `[0, S)`. This is the whole-sequence
matching pass against the simplified ancestor reference. Retain its site
parents and copied alleles, and invert `old_to_new` to map parents back to
original-reference node IDs.

Simplification also retains internal ancestral nodes. These are available to
the matcher, but only parents in this haplotype's selected focal set can be
accepted. The synthetic root is not an unmatched-region indicator.

If there are no focal candidates, skip the reduced pass and treat `[0, S)`
as one gap.

### 2. Extract accepted pieces and every gap

Use the NPZ to identify eligible focal parents and the HMM's copied alleles
to identify exact matches. The NPZ contains candidate identities, not interval
bounds. There is no need to load chunk CSVs or run another haplotype sweep.

Construct one Boolean acceptance mask of length `S`. Site `i` is accepted
exactly when:

```text
h[i] is called
and original_parent[i] is in this haplotype's eligible focal set
and reduced_copied_allele[i] == h[i]
and the site's position is inside that ancestor's original support
```

Use one vectorized parent-membership calculation, then vectorized allele
comparison and support checks. Look up support only for eligible focal
parents. An inferred TS can supply alleles outside the ancestor's generated
span, so equality alone is insufficient there.

This deliberately accepts exact portions of the chosen focal path without
requiring each portion to contain a focal seed or imposing a new minimum
length. It tests the simplest stitching method first. It does not claim to
find all possible perfect matches to every focal ancestor.

Find every maximal false run in the mask by padding its complement with false
sentinels, locating Boolean transitions, and pairing the transition indices
into half-open `[gap_start, gap_end)` intervals. This is a linear scan of the
site mask and includes leading and trailing gaps. Missing sample calls,
mismatches, internal-ancestor copying, and copying outside support all become
gaps. Every site belongs to either accepted coverage or exactly one gap.

Retain accepted site parents and copied alleles directly in the arrays that
will become the stitched result. Parent changes within accepted coverage are
preserved; they do not create a gap.

### 3. Match each gap and stitch

Create one full-reference `AncestorMatcher` for this haplotype's gaps and
reuse it for successive calls:

```python
left, right, parent = matcher.find_path(h, gap_start, gap_end, match_out)
```

Pass the full haplotype array and original site indices, not sliced and
renumbered haplotypes. The returned bounds are absolute genomic coordinates.
Convert the path to site parents and copy only the requested gap's parent and
copied-allele slices into the stitched arrays before the next call overwrites
the match buffer.

Use independent gap matching with unconstrained endpoints. No padding,
boundary conditioning, or separate gap-process scheduling is needed.

After filling every gap, derive mutations from the complete stitched copied
alleles: one mutation for each called sample site whose observed allele differs
from the copied allele. Missing calls contribute none. Accepted sites must
contribute none. This avoids retaining mutations from discarded reduced pieces.

Run-length encode the finished original-parent array into a path, merging
adjacent equal parents even when one piece was accepted and the next came from
a gap. Place parent changes at inference-site positions. Use the engine's
whole-sequence endpoints, `0` and `sequence_length`, for the first and last
segments. All stitching and switch counting use this site-based convention.

Between-site spaces contain no observations and require no extra HMM calls or
artificial switches. The path's conventional BP span is not a claim of perfect
focal evidence beyond the original ancestor support. Use inference-site
agreement as the primary comparison; no separate BP coverage analysis is
required for this initial test.

### 4. Run the full match and compare

Run `AncestorMatcher` against the original reference over `[0, S)` with the
same sample calls, parameters, and allele counts. This is the only baseline.
Compare its original parent at each site and its mutation records with the
stitched result.

Report exact parent-path equality, fraction of agreeing sites, disagreement
counts inside accepted coverage and inside gaps, mutation equality, mismatch
counts, and switch counts. Separating accepted and gap disagreements helps
distinguish focal selection differences from independent-gap boundary effects.

Score both complete paths under the same full-reference model:

```text
score = m * log(mu / (1 - mu))
      + k * log((rho / n) / (1 - rho + rho / n))
```

Here `m` is the number of called mismatches, `k` is the number of changes
between consecutive site parents, and `n` is the original full reference's
node count, including synthetic roots. `mu` and `rho` are the TOML sample
probabilities. Count all stitched seams after merging adjacent equal parents.
Do not sum separately scored gaps or use the reduced node count to score the
stitched path.

This is the normalized path score already used in
[lib/evaluation.py](../lib/evaluation.py), with common terms removed. Report
`score_delta = stitched_score - full_score`; use the configured tolerance
only to classify score equality. A different parent path with the same score
can be an equally good solution. A positive score delta requires checking the
scoring and matching assumptions before interpreting it as an improvement.

## Minimal outputs

Write only these files beneath
`experiments/ch5/data/chunk_matching_experiment/{dataset}/`:

| File | Contents |
| --- | --- |
| `full_matches.jsonl` | Whole-sequence full paths and mutations |
| `stitched_matches.jsonl` | Accepted reduced pieces plus filled gaps, as complete paths and mutations |
| `comparison.csv` | One row per haplotype comparing these two results |

Key records by dataset, source, sample ID, ploidy index, and cutoff. JSONL paths
use `left`, `right`, and original-reference `parent`; mutations use `position`
and canonical `derived_state`. Include the accepted site intervals and gap
site intervals in the stitched record so every retained and replaced region
can be inspected without additional diagnostic files.

The comparison table needs only:

```text
num_focal_ancestors, num_sites, accepted_sites, gap_sites, num_gaps,
parents_identical, fraction_parent_agreement,
accepted_disagreement_sites, gap_disagreement_sites, mutations_identical,
full_mismatches, stitched_mismatches, full_switches, stitched_switches,
full_score, stitched_score, score_delta, scores_equal
```

Preserve selected NPZ order in outputs. Keep reduced matches, gap-match
buffers, and reduced references as transient job state. Add no timing fields,
extra matching modes, intermediate TS files, or separate reduced-match tables.

## Correctness checks and first run

Use direct assertions or clear errors during actual experiment runs:

1. Sample/panel identities, site axes, allele polarity, and original node
   mappings agree; simplified references retain the required roots and sites.
2. Accepted sites meet the exact acceptance predicate. Accepted and gap
   intervals are disjoint and cover `[0, S)` completely.
3. Every stitched site has exactly one original-reference parent and copied
   allele, and every mutation corresponds to a called mismatch.
4. Reconstructing site parents from the saved paths reproduces the arrays;
   switch counts include all seams and agree with merged path segment counts.
5. Both scores use identical full-model parameters and the same site range.

Start with the two n300 1 Mb haplotypes in the example YAML and confirm that
gap matching actually occurs. Inspect the stitched and full paths, mutations,
and score differences. Then expand to a few contrasting haplotypes and the
existing `out_of_africa_n100_10mbp` and `tgp_chr20_n100` datasets.

If differences concentrate at gap seams, boundary conditioning is the next
experiment. If they occur inside accepted pieces, reconsider which focal
pieces should be fixed. The present implementation should answer those
correctness questions using the two paths and their comparison table.
