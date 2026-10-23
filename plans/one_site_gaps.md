# Missing calls and one-site gaps

Implement only the first two behaviour changes from
[further_stitching_improvements.md](further_stitching_improvements.md), as two
independently reviewable commits. Stop after these; orchestration, reference
reduction and analysis-notebook changes are separate work. This is an
implementation plan, not a claim that the changes have been tested or measured.

Use these exact one-line commit messages, with no prefix or body:

1. `Ignore missing calls during greedy matching`
2. `Copy across supported single-site gaps`

The second message preserves the wording of the combined plan. Throughout this
plan, a **one-site gap** means one inference-site index, not one base pair or a
mutation-frequency category. Copying across it may introduce a recurrent or back
mutation on the sample node, but only when needed to preserve a called allele.

## Commit 1: Ignore missing calls during greedy matching

### Extension and matching domain

In both directions of `lib/greedy.py::extend_exact`, skip missing sample calls
when the ancestor call is known. Keep an unknown raw ancestor call as a boundary:

```python
if ancestor_call < 0:
    break
if sample_call >= 0 and sample_call != ancestor_call:
    break
```

Missing sample observations supply no comparison and do not consume a mismatch
allowance. This follows the full HMM's treatment of a missing query call as
uninformative. Keep the original site axis, current AF selection, stable candidate
ordering and minimum-component rule. A candidate still requires a called derived
sample allele at its focal seed; packing already requires the ancestor seed to
be called and derived. Missing seeds do not activate candidates.

Compute the existing sample domain once in each matching/diagnostics caller:
first through last called site, or the whole axis for an all-missing sample.
Pass these bounds into candidate preparation and intersect them with each
ancestor's maximum support **before extension and minimum filtering**. Otherwise
terminal missing stretches could increase coverage outside the domain or make a
short component pass the minimum. Catalog supports remain immutable and already
respect inference boundaries. An all-missing sample activates no candidates and
retains the existing HMM fallback.

### Use the existing HMM for unknown ancestor calls

The local panel audit below found no unknown calls inside ancestor supports.
Keep the existing packed input and output flow. There is no reason to prepare
inherited-allele arrays, traverse reference trees during packing, or scan focal
runs for extra corrections in this MVP.

The distinction is between a missing **query observation** and a copied reference
allele. `tsinfer` can produce an unknown raw ancestor call when all contributing
sample calls are missing at an older site. During ancestor matching, that call
is an uninformative query; the HMM copies a concrete reference allele and emits
no mutation for the missing observation. Later sample matching copies that
resolved allele from the reference tree, rather than treating the original
panel `-1` as a wildcard reference state.

Stopping greedy extension at an unknown ancestor call leaves it to the existing
gap HMM, which already uses the resolved reference alleles. This preserves called
sample genotypes without adding a resolver for a case absent from the checked
panels. Do not reject the whole input or assume such calls are impossible.

Keep native reference restrictions and output finalisation unchanged. The
current index builder rejects multiple mutations at a reference site; this
does not prevent adding recurrent or back mutations to the sample output and
computing their parents during finalisation.

Update both callers of candidate preparation, including diagnostics' reuse
across minima and worker warm-up. Keep the README description concise: missing
sample observations are skipped, while called disagreements and unknown
ancestor calls stop extension. This commit must be usable and verifiable
without the one-site-gap change.

## Commit 2: Copy across supported one-site gaps

### Gap policy

First apply the existing exact-component minimum filter. Split uncovered
intervals at existing inference boundaries, then handle each resulting one-site
gap at site `s` using this deterministic preference. A copying choice must have
support at `s` and a known packed allele:

1. The left retained focal parent, if its maximum support contains `s`.
2. Otherwise the right retained focal parent, if supported.
3. Otherwise another sample-activated focal ancestor whose support contains `s`.
4. Otherwise the existing HMM fallback.

Support means `starts[a] <= s < stops[a]` within the sample domain. The eligible
set is all sample-activated candidates from commit 1, including candidates
superseded by the cover or rejected by the minimum. Retained neighbours are
those from the exact cover; a newly inserted one-site run does not become a
new candidate or change the preference for later gaps.

On the checked panels, every supported choice has a known allele, so a
supported one-site gap bypasses the HMM. A missing sample call is not a fallback
condition. If a raw ancestor call is unknown, try the next choice, then leave
the gap to the HMM if none is usable. A rejected candidate may supply this
one-site copy without restoring its rejected component. Leading
and trailing gaps, one-site pieces created by inference-boundary splitting,
and a one-site interval with no retained runs use the same rule. Larger
uncovered intervals remain HMM intervals; do not break them into individual
sites to apply the shortcut.

### Bounded support lookup

Use the combined plan's prefix lookup rather than scanning every candidate at
every gap:

- Stable-sort catalog IDs by maximum-support start once during preparation.
- For each haplotype, sweep that order and fill worker-owned prefix scratch with
  the activated ancestor having the furthest stop so far, or no ancestor. Keep
  the first in stable order on equal stops. Activation must use the same called
  derived seed rule as candidate discovery, not the retained-run list.
- If both neighbours fail, binary-search the last support start at or before
  `s`. Its prefix-best ancestor is supported exactly when its stop exceeds `s`.
  Check its packed call before copying. If unknown, use the HMM; do not add an
  exhaustive search or a more elaborate index for alternative known alleles.

Use immutable catalog supports for this lookup: queries are already inside the
sample domain, so allocating another set of domain-clipped support arrays adds
nothing. The cost is `O(A)` work and reusable scratch per haplotype plus
`O(log A)` per third-parent query, where `A` is catalog size. Prepare the prefix
once per haplotype and reuse it across diagnostic minima. Keep the sweep in the
existing numeric-kernel style; no interval-tree dependency, lazy cache or dense
site-by-ancestor table is needed.

### One shared traversal for matching and diagnostics

Keep extension and exact covering free of gap and mutation policy. Add a small
shared traversal in `lib/greedy.py` that takes retained runs, sample bounds and
inference boundaries and produces ordered, disjoint copied/HMM intervals.
Copied intervals retain catalog ancestor IDs; translate to reference node IDs
only for matching output. Use a simple named record for interval results.

This replaces matching's gap-policy loop and diagnostics' independent mask
partitioning. Move boundary splitting into this traversal so matching calls
`_match_region` once per emitted HMM interval; do not leave a second splitter in
`_match_gap`. Diagnostics builds and coalesces masks from the same decisions.
No additional whole-site traversal or node-ID reverse map is needed.

Extend a supported adjacent parent's endpoint, or emit a one-site copying run
for a third parent. Preserve inference boundaries and the original cover scratch
needed by diagnostics' next minimum. Do not force new runs into scratch sized
only for the original exact cover. Let the traversal hold the pending interval
needed for endpoint extension and emit completed intervals in order; use the
existing output block assembly rather than a second path representation or
another covering pass.

For each bridged site, compare the sample with the selected parent's known
packed call. Add a mutation site only when the sample is called and differs.
Pass these sites through `_assemble_result`, `append_result` and native
`finalise_match_tables`, which already computes recurrent/back-mutation parents.
Do not insert mutations directly or use stale HMM `match_out` scratch to infer
the focal allele. Bridged sites and HMM intervals are disjoint, so each site is
collected once without a deduplication set, copied-haplotype buffer or full-site
mismatch scan. Missing sample calls and matching alleles need no mutation.

Count copied interval lengths as `focal_sites`, keeping
`focal_sites + gap_sites == domain_sites`. Count only actual native calls in
`gap_calls`. Define `focal_runs` as the final greedy copying runs emitted before
output-table edge squashing; third-parent choices may increase it. Add a simple
`one_site_gap_sites` count through the existing collector for comparison.
No new tuning parameter is required.

Update the concise README algorithm and `hmm.mismatch` descriptions: that
setting controls remaining HMM calls, while one-site-gap copies can require
sample mutations. Regenerate masks and diagnostics with new stitched results;
do not apply the new mask policy to old paths.

## Verification and MVP limits

Use temporary targeted checks and an existing small pipeline run, with all
Python tooling invoked through `uv run`. Do not add a test suite or a benchmark
framework. Evaluate commit 1 separately before adding commit 2.

- Check missing calls in either extension direction, called/missing seeds,
  terminal missing stretches and the all-missing domain. Confirm clipping
  precedes minimum filtering and short components remain rejected.
- Check that an unknown raw ancestor call stops extension and is ineligible
  for a one-site copy. Verify called-genotype reconstruction through the HMM
  fallback and after finalisation of a one-site-gap back mutation.
- Check internal, repeated, leading and trailing one-site gaps; left preference,
  right fallback, a third activated parent and no supporting parent. Include a
  rejected candidate supplying a one-site copy and a gap split at a boundary.
  Confirm larger intervals retain HMM matching.
- Confirm complete nonoverlapping domain coverage, matching/diagnostic mask
  agreement across minima, one mutation per needed correction, and counters
  consistent with actual HMM calls and copying intervals.
- Compare the baseline, commit 1 and commits 1–2 on identical inputs using the
  existing measurements. Record total elapsed time, peak memory, coverage,
  mutation counts and HMM calls, including preparation costs. Parent paths may
  change; called-genotype reconstruction is the correctness requirement.

These checks establish the intended behaviour, not a guaranteed speedup or a
claim about dating accuracy. Keep experiment settings in configuration and
experiment findings out of the README. Preserve existing multiprocessing,
bounded sample loading and output ownership. Do not add worker sessions,
persistent reference caches, retries, new native APIs or general-purpose
fallback machinery for hypothetical reference formats.

## Local evidence reviewed, 23 October 2026

Scanned complete raw panel supports, not just selected AF cutoffs. These are
the local inputs used by the smaller analysis experiments; raw panels are in
`tsinfer-match-eval/data`, while the corresponding analyses and archived
reference trees are in `dphil-analysis/data`.

| Panels | Ancestor-site cells inside supports | Unknown calls inside supports |
| --- | ---: | ---: |
| Simulation, 300 individuals, 10 Mb, with errors | 112,905,733 | 0 |
| TGP, 300 individuals, 10 Mb | 127,855,221 | 0 |
| TGP, 300 individuals, 30 Mb | 456,675,880 | 0 |
| Four 1 Mb panels: simulations with 100/300 individuals, TGP with 100, and simulation truth with 300 | 12,492,737 | 0 |
| Total | 709,929,571 | 0 |

All calls outside these supports were missing padding. The simulation-with-error
sample store and the TGP sample store also had no missing calls. The archived
simulation-100 and TGP-320 ancestor reference trees in `dphil-analysis` each had
one mutation per site, consistent with the current native index restriction.

Reviewed `fix_reference_ts` at `67710ca`. Relevant code is
`tsinfer/lib/ancestor_builder.c` (consensus and all-missing contributing sets),
`tsinfer/lib/ancestor_matcher.c` (missing-query emissions and copied alleles during
traceback), and `tsinfer/tsinfer/matching.py` (called-only mutation detection and
finalisation). Temporary native checks produced an ancestor `[1, -1, 0]` from
missing contributors and matched query `[1, -1, 0]` to copied `[1, 0, 0]` without
a mutation. Calling the middle query allele `1` instead required a mutation.
An index-construction check also confirmed rejection of two mutations at one
reference site. These checks confirm the behaviour above without adding
production machinery.
