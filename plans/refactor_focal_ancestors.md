# Refactor focal ancestor generation

## Scope and priorities

This plan is for implementation in the current branch of `tsinfer-match-eval`.
The present implementation remains recoverable from main. Simplicity,
readability, and a consistent focal-seed definition take priority over preserving
previous results, APIs, or archive formats.

Do not change anything in `dphil-analysis`. Do not adapt its readers, notebooks,
experiments, tests, or metadata loading. Existing analyses may stop working when
the statistics output disappears or the focal archive changes; that is accepted.
Do not add compatibility readers, aliases, migration code, or duplicate outputs.

Implement inferred focal ancestors only. Keep existing true-panel extraction
and ancestor/sample matching code dormant and outside the default workflow.
Remove the true-panel branch from focal lookup rather than carrying it into the
new inferred implementation. Do not add stitching, coverage summaries, population
outputs, or new matching algorithms in this refactor.

## Final behaviour

For every inferred ancestor, select exactly one non-padding focal position:

- `focal_choice: left`: the smallest focal position.
- `focal_choice: right`: the largest focal position.

A sample haplotype has that ancestor as a focal candidate if and only if its
call at this selected site is derived. A missing or ancestral call excludes the
ancestor. Calls at the ancestor's other focal sites do not affect membership.

Use this selected site as the interval seed for the same association. Focal
lookup must no longer collect candidates from all focal sites while interval
generation selects only one. An ancestor with no non-padding focal positions
has no seed and contributes no candidates or intervals.

This changes candidate membership intentionally. It is not an attempt to recover
the previous all-focal candidate sets. Neither endpoint choice is assumed to
give the same tract as the other.

## Configuration and function contracts

Add a top-level option to `config.yaml.example` and the working `config.yaml`:

```yaml
focal_choice: left
```

When loading configuration, the Snakefile reads
`config.get("focal_choice", "left")` and validates it exactly once, before
constructing rules or launching jobs. Only the exact strings `left` and `right`
are valid. Reject any other value, including null; do not accept synonyms,
arbitrary focal indices, or a best/longest/all mode. Omission means `left`.
Pass the resolved value unchanged to both focal rules.

All downstream functions assume this validated contract. Do not validate the
option again in public functions, helpers, preparation, or worker initialisers.
Do not compare it against archive metadata. Saved `focal_choice` fields record
provenance only; Snakemake dependencies and params keep the products consistent.
Direct library callers are responsible for supplying a valid choice.

In general, give each necessary check one owner at the earliest relevant input
boundary. Configuration values are checked when configuration is loaded;
cross-file relationships are checked at their loading boundary. Do not repeat
those checks in callers, consumers, sweeps, or workers, and do not introduce
downstream validation merely to defend against violations of these contracts.

Keep `ac_cutoff` as the existing list. Only its maximum controls interval
generation. Keep `haplotype_compare.max_mismatches`, sample selection, and
allocated worker counts as existing inputs.

Move `find_focal_ancestors` from `lib/matching.py` to `lib/haplotypes.py`, with
this public interface:

```python
find_focal_ancestors(
    samples_path: pathlib.Path,
    ancestors_path: pathlib.Path,
    output_path: pathlib.Path,
    focal_choice: str = "left",
    sample_selection: str | None = None,
) -> None
```

Remove `ancestral_state` and `true_dataframe_path` from this function. Its panel
is inferred, and its ancestral allele comes from the panel.

Use this interface for interval generation:

```python
construct_focal_ancestor_intervals(
    samples_path: pathlib.Path,
    ancestors_path: pathlib.Path,
    focal_ancestors_path: pathlib.Path,
    output_path: pathlib.Path,
    max_ac_cutoff: int,
    max_mismatches: int,
    threads: int,
    focal_choice: str = "left",
    sample_selection: str | None = None,
) -> None
```

Remove its `ancestral_state` argument. Do not retain wrappers for the previous
interfaces.

## Shared sample loading and focal selection

Put the following small helpers in `lib/haplotypes.py`. Do not introduce a new
utility module, generic allele mapper, or framework.

### `_load_sample_haplotypes(samples, panel, sample_selection)`

Both public functions open their input stores once and call this helper with
the opened stores. Return a `SampleHaplotypes` dataclass with:

- `calls`: site-by-haplotype canonical numeric calls.
- `sample_id`: sample IDs repeated in haplotype order.
- `ploidy_index`: increasing ploidy indices within each selected sample.
- `sample_rows`: sample-store row indices for the panel's positions.

The helper performs these operations in separate, readable statements:

1. Read panel positions and sample positions; map panel positions with
   `np.searchsorted`.
2. Check once that the mapped positions actually match, including the case
   where an insertion index falls beyond the sample axis. Raise `ValueError`
   for a missing panel position. Do not repeat this check in either caller.
3. Resolve sample columns with `tsinfer.vcz.resolve_samples_selection` and load
   their genotypes and the sample allele strings at the mapped rows.
4. Read ancestral allele strings from `panel["variant_allele"][:, 0]`.
5. Follow the existing focal lookup's string-based approach: preserve the
   called mask, replace missing genotype indices with zero only for allele
   lookup, and obtain called allele strings with `np.take_along_axis`.
6. Encode called ancestral alleles as 0, called non-ancestral alleles as 1, and
   missing calls as -1. Store the resulting numeric calls as `int8`.
7. Flatten sample and ploidy dimensions in sample-major order and construct
   the corresponding identity arrays.

The panel's canonical encoding is authoritative. Do not read the ancestral-state
annotation here, reconstruct canonical allele pairs, validate every allele
string against the panel, or require raw sample genotype codes to already be
canonical. The existing pipeline input and inferred-panel contracts suffice.

### `_select_focal_sites(panel, focal_choice)`

Read `sample_focal_positions`, discard negative padding, and choose the minimum
when `focal_choice == "left"`, otherwise the maximum. The configuration loader
has already established that the other choice is `right`; do not add an
invalid-choice branch or assertion here. Convert chosen
positions to panel site indices. Return one `int64` array in panel-column order,
with -1 for ancestors without focal positions.

Use a straightforward loop over ancestors; avoid sentinel arithmetic or a
generalised selection abstraction. Trust generated focal positions to lie on
the generated panel axis. This helper is used by focal lookup; interval
generation reuses its saved result rather than selecting seeds again.

## Rewrite focal lookup as numeric candidate generation

Delete `_inferred_focal_dataframe` entirely. Do not move it or preserve its
dataframe construction, ancestor-time loading, derived-frequency loading,
string-ID joins, or expansion of all focal positions.

The new `find_focal_ancestors` does the following:

1. Open samples and panel, then load `SampleHaplotypes`.
2. Select one seed per ancestor with `_select_focal_sites`.
3. Read exact derived AC from `variant_match_eval_derived_ac` at each selected
   seed, using `sample_rows` to address the sample store. Build a panel-aligned
   `derived_ac` array, using 0 for unanchored ancestors. Such ancestors are
   never candidates.
4. Build the sorted panel columns that have a seed. For each haplotype, retain
   those columns whose selected-seed call is 1. Append the retained columns
   and update ragged offsets. The columns are already unique and ordered;
   do not collect duplicate candidates and deduplicate afterward.
5. Write the focal archive directly from numeric arrays and identity arrays.

Do not filter by AC in focal lookup. The archive contains all selected-seed
candidates; interval generation applies the inclusive maximum AC separately.
Keep both ploidy copies and preserve empty candidate rows.

The focal archive contains exactly these fields:

| Field | Shape and meaning |
| --- | --- |
| `sample_id` | `(H,)`, Unicode haplotype roster |
| `ploidy_index` | `(H,)`, int64 chromosome indices |
| `ancestor_id` | `(A,)`, Unicode panel IDs in panel-column order |
| `offsets` | `(H + 1,)`, int64 ragged boundaries |
| `ancestor_index` | `(P,)`, int64 sorted candidate columns per haplotype |
| `derived_ac` | `(A,)`, int64 exact selected-seed AC; 0 for unanchored columns |
| `focal_site_index` | `(A,)`, int64 selected seeds; -1 for unanchored columns |
| `focal_choice` | Unicode scalar, `left` or `right` |

Here H is the number of haplotypes, A the number of panel ancestors, and P the
number of candidate associations before AC filtering. Use
`np.savez_compressed`; archives must load with `allow_pickle=False`.

## Simplify interval preparation and retain the existing sweep

Rewrite `_prepare` to perform only interval preparation:

1. Open the sample and panel stores and call `_load_sample_haplotypes`.
2. Read the focal archive. Keep one compact identity/order check comparing
   `sample_id`, `ploidy_index`, and `ancestor_id` with the loaded stores.
3. Trust the focal archive's saved seeds. Do not validate `focal_choice` or
   compare the requested choice with the archive's provenance field. Do not
   infer the convention from old archives or fall back to a seed choice.
4. For each haplotype slice, select candidates with `derived_ac <= max_ac_cutoff`
   and expand the numeric haplotype and ancestor association arrays.
5. Obtain each association's seed directly from the archive's
   `focal_site_index`. Obtain its AC from `derived_ac`.
6. Intersect ancestor support with the inference interval containing that seed,
   convert support boundaries to panel site indices, and group associations
   by physical ancestor-column block for the existing comparison sweep.

The seed is not recomputed. Candidate membership was defined from this exact
seed using the same sample loader, so remove the sample-seed-derived check.
Also remove the ancestor-seed-derived check from `_sweep_block`: the inferred
ancestor is generated with the derived call at its own focal seed.

Delete the following downstream revalidation:

- Ancestral annotation membership in the first two sample alleles.
- Full reconstructed canonical allele comparisons.
- Whole-array scans checking raw genotype code ranges.
- Focal NPZ count-array shape, offset layout, index range, and candidate sorting
  checks: the writer owns those invariants.
- Reloading sample AC and comparing it with the focal archive per association.
- Focal-position existence checks after selection.
- Checks that generated seeds lie inside generated support or inference
  intervals, and the final post-sweep bound-validation pass.

Validate the AC configuration and nonnegative maximum K once when configuration
is loaded, alongside `focal_choice`. Check worker-count arguments once where
work is launched; do not repeat them in helpers or workers. Remove the existing
public-function checks that duplicate these boundary checks. Preserve direct
exceptions from stores and NumPy; do not add recovery, schema validators, or
compatibility handling. The identity/order check belongs only to the focal
archive loading boundary; do not repeat it in the sweep or worker code.

Retain the existing Numba directional sweep, physical block reuse, streamed
site-block state, and queued multiprocessing. Do not replace them with Python
per-site loops or load the entire ancestor genotype matrix. Use dataclasses
only for coherent returned data; do not introduce new intermediate containers
without a concrete need.

Preserve interval semantics:

- K mismatches independently on each side of the selected matching seed.
- Missing sample or ancestor calls stop extension regardless of K.
- The next mismatch beyond the budget is excluded.
- Half-open bounds on the panel site axis, clipped to support and the containing
  inference interval.
- Budget columns 0 through `max_mismatches`, exact AC, and separate associations
  for distinct ancestors even when their bounds coincide.

Keep the interval archive's current numeric fields and roster fields, adding
the Unicode scalar `focal_choice`. Association ordering remains haplotype order
then increasing ancestor column. Do not add coverage vectors, BP summaries,
population labels, or per-seed alternative intervals.

## Remove the statistics branch and all dedicated support

Delete `lib/evaluation.py` in its entirety: `EvaluationContext`, `HaplotypeStats`,
`_build_context`, `_compute_haplotype_stats`, and `compute_focal_ancestor_stats`.
Remove the `evaluation` import and the `compute_focal_ancestor_stats` rule from
the Snakefile. The statistics CSV is no longer an output or default target.

Remove the Snakefile helpers `metadata_input`, `metadata_params`, and
`enrich_populations`. Delete `utils.read_population_metadata`,
`utils.extract_ts_populations`, and `utils.extract_csv_populations`; they exist
only for this branch. Remove unused pandas and tskit imports from `utils.py`.

Keep shared utilities that remaining rules use, including `setup_log`,
`annotate_derived_counts`, and `write_inference_config`. Keep derived AF
annotation: dormant true-panel extraction still consumes it. Keep pandas as
a project dependency because `lib/ancestors.py` still uses it.

Remove `metadata_source`, `csv_path`, `zarr_id_field`, `csv_id_field`, and
`pop_field` from the working configuration and example. Keep `ts_path` for
dormant true-panel extraction and `sample_list` for actual cohort selection.
Do not replace population enrichment with another metadata output.

## Snakefile and default targets

Change the `find_focal_ancestors` rule to an inferred-only rule, with explicit
`{name}_inferred_ancestors.zarr` input and
`{name}_inferred_focal_ancestors.npz` output. Call
`haplotypes.find_focal_ancestors`. Remove its true-dataframe input, true-branch
logic, and ancestral-state parameter; delete the now-unused `focal_dataframe`
helper. Leave the shared ancestor/sample matching rules capable of handling
their existing dormant true-panel targets.

Add `focal_choice` to the `params` of both focal rules. Pass it into the library
functions. Remove the interval rule's ancestral-state parameter. Retain its
sample selection, maximum AC, maximum K, allocated cores, and log setup.

For every configured dataset, `rule all` must require exactly these products:

1. Inferred ancestor Zarr.
2. Inferred ancestor reference trees.
3. Inferred focal ancestor NPZ.
4. Full-reference raw sample trees.
5. Full-reference sample match JSONL.
6. Inferred focal ancestor interval NPZ.

There must be no statistics CSV or true-panel target in `rule all`. Focal
candidates and intervals must remain independent of ancestor/sample matching;
an explicit interval target requires annotated samples and inferred ancestors,
but no reference trees or sample matches.

Changing `focal_choice` must invalidate candidates through Snakemake params
and consequently intervals, without regenerating inferred ancestors or full
matches. A lower maximum AC or a changed maximum K invalidates intervals only.
Changing smaller members of `ac_cutoff` with the maximum fixed has no effect
on either focal product. Existing sample-selection and inference-setting
dependencies remain in place.

## Documentation and implementation order

Implement in this order:

1. Shared loader, selected-seed helper, and inferred focal writer in
   `lib/haplotypes.py`; remove focal functions and unused pandas import from
   `lib/matching.py`.
2. Simplified interval preparation and removal of redundant sweep checks.
3. Statistics module and dedicated population-helper deletion.
4. Snakefile calls, parameters, and default targets; configuration changes.
5. README and affected docstrings.

Update README instructions to describe selected-seed candidate membership,
both endpoint choices, the new archive fields, and removal of statistics and
population enrichment. Remove active commands and output tables for the deleted
statistics branch. Update stale `lib.matching.find_focal_ancestors` docstring
references, including the obsolete true-focal consumer description in
`lib/ancestors.py`. Keep historical plans unchanged; this plan supersedes their
conflicting focal-selection and statistics instructions.

Existing generated files are not a migration target. Do not automatically
delete old trees, archives, CSVs, logs, or data directories. Use temporary output
locations for verification, and document that candidates and intervals must
be regenerated after implementation. Do not preserve numerical equality with
old analyses as an acceptance condition.

## Verification and completion

Do not add a test suite. Use temporary manual verification scripts and manifests,
run with `uv run`, without touching `dphil-analysis`.

- Run `uv run ruff check lib` and `uv run ruff format --check lib`.
- Generate focal products for both `left` and `right` on existing simulation
  and real-data panels containing multi-focal ancestors. Independently verify
  that membership is exactly a derived call at the chosen endpoint seed.
- Check canonical loading for reversed REF/ALT ancestry and missing calls,
  using existing data where present and small temporary arrays otherwise.
- Compare generated interval bounds with independent outward scans from the
  saved seed at K=0, 1, and 2. Include missing barriers, support boundaries,
  empty haplotype slices, and distinct ancestors with identical bounds.
- Confirm one-worker and multiple-worker results agree for the same inputs.
- Confirm lower maximum AC selects the same associations and bounds from a
  higher-maximum run, and increasing maximum K preserves earlier columns.
- Confirm default omission equals explicit `left`, and unsupported choices
  fail at configuration loading before jobs launch. Inspect downstream code
  to confirm there are no repeated option checks or archive/config convention
  comparisons.
- Dry-run the default workflow and an explicit interval target. Verify that
  statistics and true-panel outputs are absent from default targets, and the
  interval branch requires no matching products.
- After generating temporary products, dry-run a change from `left` to `right`:
  candidates and intervals must rerun, while ancestors and full matches remain
  current. Check maximum-AC and K changes separately.
- Search active code and documentation for deleted statistics functions and
  stale focal API references. Historical plans may retain those names.

Completion requires a single shared sample-polarisation implementation,
selected-seed membership and extension using the same saved seed, one-time
configuration validation with no downstream option rechecks, removal of
the statistics branch and its exclusive helpers, and the exact inferred-only
default outputs above. No backward-compatibility work or `dphil-analysis`
changes are part of completion.
