# Simplify direct focal-ancestor comparisons

## Aim and scope

Replace the large focal-chunk CSV with a compressed numeric archive of raw
`[left, right)` site intervals and exact focal allele counts. Choose one focal
site per ancestor before comparison: its leftmost non-padding focal position.
The pipeline generates intervals; `dphil-analysis` continues to calculate
coverage statistics and make plots.

This is a direct rewrite of the existing branch. This document supersedes the
multi-focal output and dataframe requirements in
[haplotype_compare.md](haplotype_compare.md) and
[haplotype_compare_prompt.md](haplotype_compare_prompt.md). It describes future
implementation only. No code, configuration, notebooks, or data products are
changed by this planning task.

Keep the implementation specific to inferred ancestors and the current inputs.
Use the existing alignment, genotype polarity, support boundaries, directional
Numba sweeps, and physical ancestor-block reading. Remove output machinery and
state that exist only to construct the dataframe. Do not introduce endpoint
aggregation, interval multiplicities, dense coverage arrays, per-cutoff files,
new dependencies, or a general storage abstraction.

## Leftmost focal site is the convention

Select the leftmost focal position once per ancestor, before expanding its
associations with sample haplotypes and before either directional sweep. Each
eligible sample–ancestor association consequently has exactly one seed. The
analysis must no longer choose or deduplicate focal seeds.

This preserves the existing chapter 5 convention: `load_coverage_data` currently
sorts all generated focal seeds and keeps the lowest focal-site index for each
haplotype, ancestor, and mismatch budget.

The justification is this explicit convention, rather than equivalence of all
focal seeds. Identical carrier patterns at focal sites do not imply identical
sample and ancestor genotypes between them. A read-only check of the existing
`out_of_africa_n100_1mbp` stores found the following for sample `tsk_0`, ploidy
index 0, ancestor column 2:

| Chosen focal-site index | K = 0 interval | K = 1 interval |
| ---: | --- | --- |
| 4672, the leftmost focal site | `[4669, 4674)` | `[4651, 4685)` |
| 4712, the rightmost focal site | `[4710, 4726)` | `[4704, 4744)` |

Its focal sites are 4672, 4694, and 4712, with mismatches at 4674, 4685, 4703,
and 4709 between the outer focal sites. These bounds were obtained directly
from mismatch ranks in the stored haplotypes, without running the production
sweep. Do not assume focal-seed equivalence in code or verification.

The existing focal NPZ associates ancestors with haplotypes carrying their
focal pattern. A retained association must have matching derived sample and
ancestor calls at the chosen leftmost focal site. An inconsistent association
is an input error; do not search for a different anchor as a fallback.

## Raw interval output

Write one file per dataset:

```text
{data_dir}/haplotype_intervals/{name}_inferred_focal_ancestor_intervals.npz
```

Use `numpy.savez_compressed` and load with `allow_pickle=False`. NPZ is already
used for focal candidates and needs no new dependency or storage framework.
Let `H` be the full haplotype roster, `P` the eligible sample–ancestor
associations, and `B = max_mismatches + 1`.

| Array | Shape | Meaning |
| --- | --- | --- |
| `sample_id` | `(H,)` | Sample IDs in focal-NPZ haplotype order, stored as Unicode strings |
| `ploidy_index` | `(H,)` | Haplotype index within each sample |
| `offsets` | `(H + 1,)` | Ragged slices into the association arrays |
| `focal_ac` | `(P,)` | Exact focal AC for each association, not a cumulative cutoff |
| `left_site_index` | `(P, B)` | Inclusive left bound; column `k` is budget `k` |
| `right_site_index` | `(P, B)` | Exclusive right bound; column `k` is budget `k` |
| `num_sites` | scalar | Length of the inferred-panel comparison axis |
| `max_ac_cutoff` | scalar | Inclusive maximum AC used for generation |
| `max_mismatches` | scalar | Largest generated directional budget |

Use int64 for numeric arrays and scalars. Keep one fixed layout rather than
adding dtype selection or format versions. Order associations by haplotype and
then increasing ancestor column. Budget columns avoid repeating the AC and
haplotype information once per budget.

There is one association entry per eligible ancestor for each haplotype. Two
different ancestors with identical AC and bounds remain two entries: they
contribute coverage twice. Do not deduplicate by interval coordinates.

The archive does not store ancestor IDs, ancestor columns, focal indices,
focal AF, BP bounds, populations, or repeated dataset/source/panel labels.
Ancestor columns and focal indices are still needed internally during
comparison, but not after bounds have been obtained. The dataset name comes
from the requested dataset and filename. The analysis obtains site positions
from the panel and populations through its existing metadata loading.

Include every haplotype in the roster, even when its slice is empty. Require
`offsets[0] == 0`, `offsets[-1] == P`, and nondecreasing offsets. An entirely
empty selection is a valid archive with zero association entries and bounds
of shape `(0, B)`. Each populated bound satisfies
`0 <= left <= chosen_focal < right <= num_sites` during generation.

## Comparison semantics

Generate once at `max(config["ac_cutoff"])`, retaining exact AC from the
panel-aligned focal NPZ. Eligibility is inclusive. A lower analysis cutoff
selects associations with `focal_ac <= cutoff`; no genotype comparisons are
repeated and no cutoff-specific intervals are generated.

Retain the present mismatch contract: budget `k` permits `k` called mismatches
on each side of the chosen focal site, potentially `2k` overall. Find and
exclude the `(k + 1)`th mismatch in each direction. The focal call matches and
uses neither budget. Missing sample or ancestor calls are terminal barriers.
Bounds remain on the inferred-panel site axis, not the original sample axis.

Intersect ancestor support with the inference interval containing the chosen
focal site, then convert those support limits to site indices. BP coordinates
are needed only for this input-support calculation. No output BP conversion
is required. Increasing budgets produces nested intervals; changing the AC
cutoff changes eligibility, not the bounds of a retained association.

## Simplify `lib/haplotypes.py`

Replace the public dataframe-producing function with
`construct_focal_ancestor_intervals`. Give it an output path alongside the
current sample-store, ancestor-store, focal-NPZ, ancestral-state, maximum-AC,
maximum-budget, thread-allocation, and sample-selection arguments. It prepares
numeric inputs, computes bounds, and writes the archive once after successful
completion. It returns `None`.

In preparation:

1. Keep exact sample/panel site alignment, canonical genotype polarity, sample
   selection, and identity/order checks against the focal NPZ.
2. For each ancestor with focal positions, discard padding, take the minimum
   position, and locate that one position on the panel axis. Ancestors with no
   focal positions create no seeds.
3. Walk each haplotype's distinct candidate columns, filter by maximum AC, and
   append one association and chosen focal index per eligible candidate.
   Construct output offsets during this traversal. Validate the chosen focal
   call and its AC against the existing sample annotation.
4. Compute association support indices and group associations into the same
   physical ancestor-column blocks used now.

Remove all-focal expansion, focal-frequency loading, and retained BP-output
state. Keep ancestor identity arrays only as long as input validation needs
them; the sweep needs numeric columns and the panel column count.

Keep the current directional kernel and block reader where they already
express the required work clearly. Each physical ancestor block is reused
across sample associations; a full-height block serves both directions.
Retain site-block state for stores with multiple physical variant chunks.
The reduced seed count also reduces sweep work and bound-array memory.

Simplify process orchestration to a spawned `multiprocessing.Pool` with an
initializer and `imap_unordered(..., chunksize=1)` over physical block IDs.
Cap worker count by the allocated threads and number of blocks, and keep a
direct one-worker path. Workers return association indices and numeric bounds
in a small dataclass; the parent fills the canonical output arrays. This
removes the manual pending-futures scheduler. Do not add another parallel
layer or send genotype blocks through task queues.

Remove worker setup/JIT timing fields, the dummy pre-compilation call, and
first-result timing machinery. Keep ordinary module logging for completion,
association count, and allocated workers. Propagate worker failures and write
the archive only after all comparisons succeed.

Delete `_positions`, `_dataframe`, and the pandas import from this module.
Remove dataframe construction, categorical identifiers, repeated budget rows,
and dataframe return types. Preserve the project-level pandas dependency,
which other pipeline modules still use. Retain only small dataclasses that
clarify genuinely multiple-valued internal results.

## Workflow and documentation

Rename the rule to `construct_focal_ancestor_intervals`, replace its output
with the NPZ path above, and replace the chunk-CSV target in `rule all`.
Pass the output path directly to the library function. Remove this rule's
population enrichment, metadata input/parameters, dataframe label insertion,
and CSV write. Its inputs are the annotated samples, inferred ancestor store,
and existing focal-candidate NPZ. Other rules still use metadata enrichment.

Keep the existing `haplotype_compare.max_mismatches` and maximum-AC parameters.
Changing smaller configured cutoffs does not invalidate intervals when the
maximum stays fixed. Maximum-AC or maximum-budget changes regenerate this
branch. Existing upstream inference-TOML invalidation remains as documented;
the interval rule itself reads no HMM, matching, truth, or tree-sequence input.

Replace the README's active CSV instructions with the NPZ target, compact
array contract, leftmost-anchor convention, and inclusive AC filtering.
Remove the superseded multi-focal/BP output description. Report any newly
measured output sizes as NPZ measurements, rather than carrying forward CSV
size or performance claims. Do not modify the earlier plan files as part of
this rewrite.

## Rewrite the `dphil-analysis` consumer

Keep `summarise_coverage`, its numeric endpoint sweep, and multiprocessing in
`~/work/dphil-analysis/src/ch5_analysis.py`. The pipeline must not compute
coverage counts, coverage summaries, or population statistics.

Replace `CoverageData.chunks` with the raw numeric association arrays and
offsets, plus the existing haplotype metadata, site count, dataset, and site
position limits. Record the effective maximum analysis AC and the loaded
budget-column labels. Polars remains useful for metadata and summary tables;
do not rebuild a large interval dataframe just to call the existing sweep.

Rewrite `load_coverage_data` to read the NPZ, validate its roster against the
existing focal NPZ and its site count against the panel, and select the
requested budget columns. Reject requested AC limits or budgets beyond those
generated. Keep the current population lookup from the statistics CSV and
original truth TS fallback. Read the full compact association arrays; there
is no need to filter and repack the ragged offsets at load time.

Delete CSV scanning, focal-site sorting/deduplication, ancestor-key grouping,
and interval-to-haplotype joins. Haplotypes are already identified by the
archive roster and offsets. Keep direct checks for valid offsets, array
dimensions, and site bounds, with errors describing the interval archive.

For each requested budget and haplotype, slice its associations using offsets,
sort that slice by exact AC, and send the existing numeric `SweepTask` to the
coverage worker. Check that requested summary cutoffs do not exceed the loaded
analysis AC limit. `_sweep_haplotype` can retain its present algorithm:

- Add `+1` at each eligible interval's left endpoint and `-1` at its right.
- Add associations incrementally as the inclusive AC cutoff increases.
- Include endpoints 0 and `num_sites` so uncovered sites enter the denominator.
- Weight constant-coverage segments by their numbers of sites.
- Return the current coverage means, extrema, percentiles, and covered fraction.

Retain the summary-table columns used by the notebook. In raw data `focal_ac`
is an exact AC; in the summary table it continues to label the cumulative
maximum cutoff. Empty haplotype slices still produce zero-coverage summaries.

Update `notebooks/ch5_ancestor_coverage.py` to describe and load intervals,
remove the claim that it selects one focal chunk per ancestor, and replace
`data.chunks.height` diagnostics with association counts and loaded budgets.
Keep the coverage plots and candidate-set-size calculation. Edit the jupytext
`.py` source and synchronise the notebook through the existing project tools.
`ch5_initial_testing.py` continues to consume the separate HMM statistics and
focal NPZ; it needs no change. The stitching experiment also remains
independent of these interval products.

## Verification and completion

Before replacing the implementation, capture the existing chapter 5 summaries
in memory or temporary files for the available simulation and 1KGP datasets,
using budgets 0 and 1 and the notebook's cutoffs. These are reference results
for the rewrite, not a new supported input format.

Verify regenerated intervals against direct mismatch-rank calculations at the
leftmost focal site, and compare all resulting coverage summaries with the
captured results. Exercise empty candidate slices, repeated interval
coordinates belonging to different ancestors, missing-call barriers, support
ends, and zero mismatch budget through read-only/manual probes. Verify that
one-worker and multiple-worker generation give identical numeric arrays.

Check that generating at a smaller maximum AC gives exactly the associations
and bounds selected from a larger-maximum archive. Increasing K must preserve
the existing budget columns. Record association counts, NPZ sizes, and
generation time for the existing n300 1 Mb simulation and n100 10 Mb simulation
and 1KGP products, without creating a benchmarking subsystem.

Rewrite the existing `tests/test_ch5_analysis.py` fixture and affected
assertions to use the new array contract, retaining its coverage-oracle and
serial/parallel checks. Do not add a new test suite or compatibility cases.
Run the existing chapter 5 tests and the appropriate lint/format checks with
`uv run`. Run the coverage notebook for the simulation and 1KGP datasets and
confirm that all existing plots can still be produced.

Dry-run the new explicit NPZ target and the default workflow. Confirm that the
NPZ is the direct-comparison product, the chunk CSV is no longer a target or
output, and the explicit interval target requires no matching branch. Check
that only the new archive writer and reader are used by active code.

Completion means that one leftmost seed per association is chosen before
sweeping, generation emits only compact raw intervals with exact AC, the
chapter 5 results agree with the previous leftmost-seed analysis, and the
dataframe/BP-output/multi-seed machinery has been removed from match-eval.
