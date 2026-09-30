# Matching evaluation implementation plan

## Scope and outputs

Evaluate only the inferred ancestor panel for now. For each sample haplotype
and configured cumulative allele-count cutoff, report the bases and inference
sites copied from that haplotype's eligible focal ancestors, the corresponding
coverage fractions, switches, mismatches, and path log likelihood. Classification
by cutoff uses the existing full-panel matching result; it does not rerun the
HMM against restricted panels.

Produce one file per dataset:

```text
{data_dir}/dataframes/{name}_inferred_focal_ancestor_stats.csv
```

Each dataframe has exactly `num_haplotypes * len(ac_cutoff)` rows. A sample in
this diagnostic means one haplotype, keyed by `(source, sample_id, ploidy_index)`.
Diploid individuals therefore have two sets of cutoff rows.

Change `rule all` to request inferred products only: the inferred ancestor Zarr,
reference TS, focal NPZ, raw sample TS, match JSONL, and statistics CSV for each
dataset. Remove the true ancestor Zarr/CSV and all true matching/focal products
from its inputs. Keep the existing true rules callable through explicit output
targets and leave existing true files on disk. True-panel evaluation is future
work; the default inferred DAG must not depend on truth even when `ts_path`
is supplied.

The one preparation for both ancestor kinds is adding `derived_ac` to their
dataframes, as detailed below. This revision changes only this plan. Production
implementation is future work and follows
[matching_setup.md](matching_setup.md), [initial_mvp.md](initial_mvp.md), and
[../AGENTS.md](../AGENTS.md).

## Configuration and inferred-ancestor eligibility

Add these required settings to `config.yaml` and `config.yaml.example` during
implementation:

```yaml
ac_cutoff: [3, 5, 10, 100, 600]
hmm:
  recombination: 0.01
  mismatch: 1.0e-20
```

Cutoffs are inclusive and cumulative: 3 includes doubletons and tripletons, 5
includes those plus four- and five-carrier focal ancestors. Require a nonempty
list of positive integers in strictly increasing order; reject duplicates,
booleans, and noninteger values. Values above the dataset's haplotype count
are valid and include all eligible candidates. Keep defaults in YAML.

The inferred builder groups focal sites by the same time and encoded genotype
pattern. Consequently an inferred ancestor's focal sites share an observed
carrier set and allele count. Store a single `derived_ac` per ancestor. There
is no sample-specific minimum count or heterogeneous multi-focal reduction.
An ancestor is eligible for haplotype `h` at cutoff `f` exactly when:

```text
ancestor is in h's existing focal candidate set
and ancestor.derived_ac <= f
```

Multiple focal sites do not multiply candidate membership or copied coverage.
Eligibility applies to all evaluated portions of that parent's copying
segments, not just to the positions of its focal sites. Precompute each
ancestor's first eligible cutoff once for the entire dataset.

The local inference code assigns `sample_time = derived_ac / num_haplotypes`.
The pipeline's existing `derived_af` annotation instead divides by the number
of called haplotypes. These equal node time on fully called inputs, including
the current example, but differ when calls are missing. Preserve the established
`derived_af` meaning and use exact `derived_ac` for cutoffs. The fixed-pattern
property still gives each inferred ancestor one count and one frequency.
Do not recover counts by multiplying AF by the total haplotype count.

## Allele counts and dataframes

`utils.add_singleton_mask` already computes the exact observed derived count
while polarising the current sample genotypes. Persist that existing array as
`variant_match_eval_derived_ac` in the copied sample VCZ. Use int64, the existing
variant chunks and `_ARRAY_DIMENSIONS=["variants"]`, and refresh metadata with
the existing Zarr v3 APIs. Missing calls contribute neither a derived call nor
a called haplotype. This adds no genotype read or additional polarisation code.

Extend `matching._inferred_focal_dataframe` to accept both `ancestors_path` and
`samples_path`, and return the existing in-memory focal-relation dataframe with
these columns:

```text
focal_position, inferred_ancestor_id, inferred_node_time, derived_af, derived_ac
```

Continue emitting one row per non-padding focal position, so multi-focal
ancestors have repeated rows with the same time, frequency, and count. Read
`inferred_node_time` from panel `sample_time`; join sample-store AF/count
annotations by exact focal position. `derived_ac` is an integer and
`derived_af`/time are floats. This dataframe need not become another persisted
CSV or workflow rule. In `find_focal_ancestors`, retain the first relation per
ancestor to extract its count, then align counts to panel `sample_id` order.
No minimum or aggregation policy is needed for inferred ancestors.

Also add integer `derived_ac` immediately after `derived_af` in the site-level
CSV written by `ancestors.extract_true_ancestors`. Read it from the same sample
annotation on the inference-site axis used for `derived_af`. Preserve every
existing row, singleton-origin flag, ancestor association, sort order, and
other column. Thus the count is available for both the true and inferred IDs
in each CSV row. It describes the observed site count, not the number of truth
descendants. Different focal rows belonging to a true ancestor may still have
different counts; do not collapse them or assign a true-ancestor cutoff yet.

The true extraction rule remains outside `rule all`. Validate this schema
change with an explicit true-CSV output target on a small example, rather than
reactivating true matching or evaluation.

## Focal NPZ and joins

Preserve the existing inferred NPZ path, rows, candidate membership, and ragged
storage. Add one panel-level array instead of the previously proposed
per-association thresholds:

| Array | Shape and dtype | Meaning |
| --- | --- | --- |
| `sample_id` | `(h,)`, Unicode | Existing sample ID for each haplotype row |
| `ploidy_index` | `(h,)`, integer | Existing chromosome index |
| `ancestor_id` | `(a,)`, Unicode | Existing panel IDs in Zarr column order |
| `offsets` | `(h + 1,)`, int64 | Existing candidate slice boundaries |
| `ancestor_index` | `(k,)`, int64 | Existing sorted distinct panel-column indices |
| `derived_ac` | `(a,)`, int64 | One observed focal allele count per inferred ancestor, aligned to `ancestor_id` |

Row `i` still takes its candidates from
`ancestor_index[offsets[i]:offsets[i + 1]]`. Their counts come from indexing
the panel-level `derived_ac` array with those columns. Save all candidates,
independent of cutoffs. Empty sets retain equal offsets. Keep the current
collect-and-unique implementation of candidate membership. Load NPZs with
`allow_pickle=False`; no object arrays, new sample-by-ancestor matrices, or
per-cutoff files are needed.

This extension applies only to inferred NPZs. Existing explicitly requested
true focal files keep their current format. The evaluator requires the inferred
count array and must give a clear rebuild instruction if it is absent.

Build a reference-node-to-panel-column array once by joining node metadata
`(source="ancestors", sample_id, ploidy_index=0)` to `ancestor_id`. Preserve
the metadata-free synthetic roots as noncandidates. Validate that every panel
ID joins to exactly one ancestor node. Do not subtract a root count, interpret
ID suffixes, or equate TS node IDs with Zarr columns.

Join native match records to focal rows by
`(source="samples", sample_id, ploidy_index)`, never by line order, `group`, or
`haplotype_index`. Require exactly one record for every focal row: reject
duplicates/unknown keys while streaming and missing keys at end of file. Check
the basic NPZ offset, alignment, sorted-distinct candidate, and index invariants
on load. Error messages should name the affected key or input and the necessary
rebuild; do not introduce a generic validation framework.

## Coverage denominators and boundaries

Use the raw inferred ancestor reference TS's `sites_position` and top-level
`sequence_intervals` metadata. For each haplotype, intersect its copying path
with those evaluation intervals before accumulating bases or sites. The
denominator includes copying from every parent, including synthetic roots;
only eligible focal parents contribute covered bases/sites.

The existing example has contig length 64,444,167 but evaluation interval
`[81,353, 999,745)`, spanning 918,392 bases. Native paths can extend from zero
to the contig end. Counting those untrimmed spans would mostly measure padded
sequence with no inference sites.

For each nonempty clipped fragment `[left, right)`:

```text
fragment_bp = right - left
left_site = searchsorted(sites_position, left, side="left")
right_site = searchsorted(sites_position, right, side="left")
fragment_sites = right_site - left_site
```

A site at `left` is included and one at `right` belongs to the next segment.
Compute site coverage explicitly; site density prevents conversion of a base
fraction directly into a site fraction. Exclude gaps between evaluation
intervals and uncovered leading/trailing path regions from both denominators.
Include internal missing-call sites when their reference position is covered
by the path: this diagnostic measures copying decisions, not observed-genotype
completeness. Sites absent from the reference are not evaluated.

Empty candidate sets give zero coverage at positive denominators. A zero
denominator gives a zero numerator and `NaN` fraction. Validate increasing site
positions and ordered, disjoint evaluation intervals once. Sort each native
path by genomic left coordinate and clip it with a two-pointer sweep; no
special path-array dataclass is necessary.

## Switch count, mismatches, and likelihood

### What the current matcher exposes

Inspected the editable sibling tsinfer checkout at commit `a704308`, including
the C implementation and Python bindings:

| Question | Source and finding |
| --- | --- |
| Is there a switch-count accessor? | `_tsinfermodule.c` exposes `find_path`, `get_traceback`, `mean_traceback_size`, and `total_memory`, with no switch-count accessor; `MatchResult` contains only `path` and `mutations` |
| Does path length supply the count? | `lib/ancestor_matcher.c:ancestor_matcher_run_traceback` starts one segment and starts a new segment for each traceback recombination; the low-level binding returns that path length and the wrapper retains exactly those segments |
| What is `n`? | `matcher_indexes_alloc` sets `num_nodes` to `tables->nodes.num_rows`; `ancestor_matcher_update_site_likelihood_values` uses that value when weighting by `n` |
| Is weighting by `n` enabled? | The binding defaults `weight_by_n=1`; the pipeline does not override it |
| Is an additional vestigial root introduced? | `tsinfer.matching.Matcher` builds `MatcherIndexes(ts, vestigial_root=False)`, so the matching index has exactly the reference TS's nodes |
| What are match/mismatch emissions? | The C likelihood update uses `mu` for a mismatch and `1 - (num_alleles - 1) * mu` for a match or missing call |
| Where are the probabilities? | `Matcher._match_one` uses scalar source-config overrides, otherwise literal defaults `recombination=1e-2` and `mismatch=1e-20` |

Accordingly use:

```text
num_switches = max(len(record["path"]) - 1, 0)
num_mismatches = len(record["mutations"])
num_reference_nodes = reference_ts.num_nodes
```

This is the simplest way to reuse the matcher's traceback count without
modifying tsinfer, rerunning matching, or walking its private traceback maps.
The current pipeline saves native paths directly, without splitting or
postprocessing their segments. Validate this contract in integration checks:
segments are contiguous, consecutive parents differ, and the segment-derived
count equals an independent count of parent changes. Do not use the segment
count on clipped fragments, which may be split by evaluation intervals.

Mismatches include all native nonmissing disagreements, irrespective of the
canonical target allele; do not count only derived-state 1 or all mutations
in the final sample TS. Switches and mismatches describe the full recorded HMM
path, even if coverage is clipped to evaluation intervals.

For `n`, include every node in the reference, including its synthetic roots.
Do not use panel column count, the number of nodes active at one site, the
sample count, or the final raw sample TS's node count. The current inferred
example has 4,639 ancestor haplotypes and `n=4,641` reference nodes.

All contemporary samples currently match in a single sample-only group, before
sample nodes are appended. Thus every record uses the same reference and `n`.
Reject multiple sample match groups in this evaluator rather than silently
scoring later groups with the initial reference's node count. Supporting
additional sample-time groups would require recording the actual per-group
reference size and is outside this inferred contemporary-sample diagnostic.

### Path log likelihood

For the current binary, biallelic input contract, the matching emission is
`q = 1 - mu`. Add a `path_log_likelihood` column with exactly the requested
normalised path score:

\[
\log \widetilde L
= m\log(\mu/q)
+ k\log\left(\frac{\rho/n}{1-\rho+\rho/n}\right).
\]

Here `m` and `k` are the original path counts, and `mu`/`rho` are the scalar
per-site HMM probabilities used for sample matching. They are not rates per
base. This score has factored out the common match-emission and no-switch
baseline; do not label it an absolute sequence probability or posterior.
It repeats unchanged for every cutoff of a haplotype. It is a metric column,
so adding it does not add rows or change the sample-by-cutoff row contract.

Read sample probabilities from the native TOML's
`cfg.match.sources["samples"].mismatch` and `.recombination`. The defaults in
`_match_one` are not exposed through a simple default-value API. Avoid source
parsing or callable introspection: mirror the verified defaults in YAML's
`hmm` setting and write them explicitly to native match-source configuration
using `utils.write_inference_config`. Apply these configured values to both
the ancestor and sample source settings. Their initial values reproduce
current matching, while later edits keep matching and scoring consistent.
The evaluator requires explicit probabilities in its matching TOML and does
not silently apply a different fallback to old match products.

Validate `0 < mu < 1` and `0 < rho < 1` for this initial scoring contract.
Precompute two penalties once using stable natural-log expressions:

```text
log_mismatch_penalty = log(mu) - log1p(-mu)
log_switch_penalty = log(rho) - log(n) - log1p(-rho + rho/n)
path_log_likelihood = m * log_mismatch_penalty + k * log_switch_penalty
```

Using `log1p(-mu)` preserves the small matching-emission correction even when
floating-point `1 - 1e-20` rounds to one. A path with no mismatches or switches
has score zero. Site-specific probabilities, multiallelic emissions, disabled
weighting by `n`, and altered traceback formats are future extensions, not
cases to infer implicitly. Log `n`, `mu`, `rho`, and the existing tsinfer commit
along with evaluation dimensions for auditability.

## Returned dataframe and functions

`evaluation.compute_focal_ancestor_stats` returns these columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `source` | string | `samples` |
| `sample_id` | string | Actual sample-store ID |
| `ploidy_index` | integer | Haplotype index within the sample |
| `ac_cutoff` | integer | Inclusive allele-count threshold |
| `evaluated_bp` | float64 | Clipped path span, including root copying |
| `covered_bp` | float64 | Evaluated bases copied from eligible focal parents |
| `fraction_covered_bp` | float64 | `covered_bp / evaluated_bp`, or `NaN` |
| `evaluated_sites` | int64 | Reference sites covered by the clipped path |
| `covered_sites` | int64 | Such sites copied from eligible focal parents |
| `fraction_covered_sites` | float64 | `covered_sites / evaluated_sites`, or `NaN` |
| `num_switches` | int64 | Native traceback switch count |
| `num_mismatches` | int64 | Native mismatch events |
| `path_log_likelihood` | float64 | Normalised score defined above |
| `num_focal_ancestors` | int64 | Eligible candidates, including those never selected |

Numerators, denominators, and candidate counts make results auditable without
additional matching. Return focal-NPZ row order, with increasing cutoffs within
each row. The rule prepends `dataset` and `panel_kind="inferred"`, then writes
CSV with `index=False`. Preserve sample IDs as strings on readback.

Keep existing matching/focal work in `lib/matching.py`; put evaluation in
`lib/evaluation.py`. The inferred-only simplification needs just three new
functions and two small dataclasses:

| Function or dataclass | Inputs | Output and responsibility |
| --- | --- | --- |
| `utils.add_singleton_mask` | Existing input/output paths and ancestral-state mapping | Existing masked copy plus the exact derived-count annotation |
| `utils.write_inference_config` | Existing paths and ancestral-state mapping, plus `hmm: dict` | Native TOML with explicit scalar matching probabilities |
| `ancestors.extract_true_ancestors` | Existing arguments | Existing panel and site-level CSV, with added integer `derived_ac` |
| `matching._inferred_focal_dataframe` | `ancestors_path: pathlib.Path`, `samples_path: pathlib.Path` | In-memory relation dataframe including frequency, count, and time |
| `matching.find_focal_ancestors` | Existing arguments | Existing candidate NPZ with inferred-only panel-aligned `derived_ac` |
| `evaluation.EvaluationContext` | Shared numeric arrays and scoring values | Dataclass holding cutoffs, ancestor first-cutoff indices, reference parent mapping, positions, intervals, sequence length, reference node count, and the two log penalties |
| `evaluation._build_context` | Reference TS, NPZ ancestor IDs/counts, cutoff list, sample `MatchSourceConfig` | Validated `EvaluationContext`; computes joins, cutoff bins, and penalties once |
| `evaluation.HaplotypeStats` | Per-sample totals and cutoff vectors | Dataclass holding denominators, switches, mismatches, score, and vectors `covered_bp`, `covered_sites`, `num_focal_ancestors` |
| `evaluation._compute_haplotype_stats` | One native record, its candidate-column slice, shared context | `HaplotypeStats`, calculated for every cutoff together |
| `evaluation.compute_focal_ancestor_stats` | `focal_ancestors_path: pathlib.Path`, `ancestors_ts_path: pathlib.Path`, `match_file_path: pathlib.Path`, `config_path: pathlib.Path`, `ac_cutoff: list[int]` | Requested `pd.DataFrame`; loads shared inputs once, streams records, validates keys/groups, and returns deterministic rows |

Keep NPZ loading, record bookkeeping, and dataframe construction directly in
the public function. Keep path ordering/clipping in the per-haplotype function
with explicit intermediate variables. Separate helpers for NPZ loading, cutoff
validation, ordered paths, clipping, and row conversion are unnecessary.
The two dataclasses give clear multi-value contracts without a pipeline wrapper
or generic framework. Follow repository import, logger, pathlib, and docstring
conventions, and keep all tunable defaults in configuration.

## One match-file pass for all cutoffs

1. Load the inferred NPZ, reference TS, and native matching configuration.
   Build shared context, mapping each ancestor's integer count to its first
   eligible cutoff with `searchsorted(cutoffs, derived_ac, side="left")`.
   A result equal to the number of cutoffs contributes to none of them.
2. Build the sample-key-to-focal-row mapping and seen-row/result arrays.
   Stream one JSONL record at a time and select its candidate-column slice.
3. Index the precomputed ancestor bins for those candidates and build a small
   column-to-bin dictionary. Count candidates into first-cutoff buckets once.
4. Obtain `k` from native path length and `m` from mutation-list length; compute
   the scalar score from shared penalties. Sort the original path and clip it
   against evaluation intervals using a linear sweep.
5. Compute fragment base lengths and site counts using vectorised boundary
   searchsorted. Every parent contributes to denominators. Translate each
   parent through the shared reference mapping; eligible sample candidates
   contribute their fragments to the bucket of their first eligible cutoff.
6. Cumulatively sum the three bucket arrays to obtain covered bases, covered
   sites, and candidate counts for all cutoffs. Use float64 base buckets and
   int64 site/candidate buckets; `numpy.add.at` or a short loop preserves integer
   site counts without float-weight conversion.
7. After checking record completeness, create output rows in focal order and
   compute fractions with explicit zero-denominator handling.

The path is never revisited for individual cutoffs. With `a` ancestors, `h`
haplotypes, `f` cutoffs, `k` total candidate associations, `p_h` segments per
haplotype, `q` clipped fragments, and `s` sites, bin preparation costs
`O(a log f)`, candidate work `O(k)`, sorting costs `sum O(p_h log p_h)`, site
lookups `O(q log s)`, and required output/cumulative sums `O(h f)`. Clipping
is a linear segment/interval sweep. Memory consists of loaded NPZ/reference
arrays, the `O(h f)` output, and one current path. There is no `O(h a)` dense
membership matrix or `O(f p)` path traversal.

Use NumPy and serial streaming initially: the current 600-haplotype inferred
probe, including focal reconstruction, ran in about one second. Snakemake can
schedule independent datasets across available workers. If substantially larger
workloads make per-dataset evaluation long-running, follow AGENTS.md by using
bounded multiprocessing queues for independent record batches across allocated
CPU workers. Add a configured worker count only with that measured need; load
shared context once per worker and restore result order by focal row. Numba is
not needed for this initial scope.

## Workflow, migration, and documentation

Add `evaluation` to the Snakefile imports and add a rule named
`compute_focal_ancestor_stats`. Its paths explicitly select inferred products:

| Rule element | Value |
| --- | --- |
| `input.focal` | `{data_dir}/focal_ancestors/{name}_inferred_focal_ancestors.npz` |
| `input.reference` | `{data_dir}/ancestors/{name}_inferred_ancestors.trees` |
| `input.matches` | `{data_dir}/matches/{name}_inferred_samples_matches.jsonl` |
| `input.config` | `{data_dir}/configs/{name}_ancestor_inference.toml` |
| `output` | `{data_dir}/dataframes/{name}_inferred_focal_ancestor_stats.csv` |
| `params.ac_cutoff` | `config["ac_cutoff"]` |
| `log` | `{progress_dir}/compute_focal_ancestor_stats/{name}_inferred_compute_focal_ancestor_stats.log` |

The run block sets up logging, calls the dataframe-returning function with
pathlib paths, prepends dataset/panel identity, and writes the CSV. The cutoff
parameter makes changes to the cutoff list invalidate only statistics. Add
`params.hmm=config["hmm"]` to both native-TOML writer rules and pass it through
to the writer, so probability changes rebuild the affected matching products
and statistics together. Evaluation reads the same explicit probabilities
from its native config input.

Use an inferred-only `panels` collection for `rule all`, removing its current
extension with true panels and its explicit true Zarr/CSV targets. Keep optional
true rules and their input routing. No new rule receives a true statistics
output, and no truth dataset list is necessary to enumerate default targets.

The first migration needs the new sample annotation, explicit native matching
probabilities, and extended inferred NPZ. Because library code is intentionally
not a Snakemake input, regenerate from source with:

```sh
uv run snakemake --cores all --forcerun mask_singletons write_inference_config find_focal_ancestors compute_focal_ancestor_stats
```

The changed masked/config inputs may also rebuild inferred descendants through
the DAG. This is a one-time format/configuration migration; default probabilities
keep numerical matching unchanged. Existing true products are neither required
nor rewritten by this command. To refresh a true CSV later, explicitly target
it and force `extract_true_ancestors` using the refreshed count annotation.

After migration, cutoff-only changes rerun only statistics; evaluation-code
changes require forcing only `compute_focal_ancestor_stats`. Changes to focal
lookup require regenerating inferred focal NPZs and statistics. Matching/HMM
changes require rebuilding reference/sample TSs, match files, and statistics
together. Do not add library files as workflow inputs or extra dependencies.

Update README and matching_setup.md with inferred-only default targets,
`derived_ac` annotation/dataframe/NPZ meanings, output schema, likelihood
definition and reference node count, explicit HMM settings, and rebuild commands.
Retain true-stage documentation as an explicitly requested optional workflow.

## Internal verification and completion

Use short `uv run` scripts with in-memory fixtures and an independent per-cutoff
oracle. No permanent unit-test suite or new testing dependency is required.
The oracle may repeat tiny calculations; production uses shared ancestor bins
and one match-file pass.

1. Verify exact counts under both ancestral-state modes and missing calls.
   For three derived calls among ten called and twenty total haplotypes, assert
   `derived_ac=3`, observed `derived_af=0.3`, and inferred time `3/20`. Check
   annotations against direct genotype counts and preserve original calls.
2. For inferred multi-focal ancestors, verify identical carrier masks/counts
   across focal sites, one panel-level count, unchanged candidate NPZ rows and
   membership, and equality of first-focal-site eligibility with the direct
   per-cutoff oracle. Cover empty sets, missing/excluded focal calls, exact cutoff
   equality, and a count above the largest configured cutoff.
3. Explicitly request a small true dataframe and check that every row's new
   `derived_ac` matches observed calls, including flagged singleton-origin
   rows. All older fields, IDs, times, row order, and panel contents remain
   unchanged. Do not evaluate true coverage or choose a true-ancestor count.
4. Check a hand-calculated coverage fixture: reference sites `[10, 20, 30, 50]`,
   evaluation interval `[10, 51)`, and reverse-order native path with genomic
   segments `[0, 20)` from candidate A with count 3, `[20, 40)` from candidate B
   with count 5, and `[40, 100)` from a root. Denominators are 41 bases and
   four sites. Cutoff 3 covers ten bases/one site; cutoff 5 and above covers
   thirty bases/three sites. The path has two switches. Verify the site at 20
   belongs to B and root copying enters only the denominators.
5. Cover multiple disjoint evaluation intervals, omitted gaps, one segment
   spanning several intervals, shortened path ends, internal missing calls,
   a fragment containing no sites, and zero evaluated intersection. Clipping
   must not change switch count, mismatch count, or score.
6. Compare native `len(path)-1` against independently counted contiguous parent
   changes on all actual example records. Single-segment paths have zero
   switches; an empty synthetic path has zero denominators and count. Do not
   manufacture split same-parent paths and assume the native segment-count
   contract applies to that transformed representation.
7. For likelihood, compare the requested formula with a directly multiplied
   probability ratio on a tiny biallelic fixture. Use `mu=0.1`, `rho=0.2`,
   `n=5`, eight sites, two mismatches, and three switches; divide by the all-match,
   no-switch baseline. Check the zero-event score, negative default penalties,
   default tiny-mu numerical stability, and exact equality across cutoff rows.
   Assert that `n` includes roots and excludes appended sample nodes. Verify
   explicit configured probabilities reach both matching TOML and evaluation;
   reject multiple sample match groups and missing explicit probabilities.
8. Check metadata joins with shuffled node/column order and nonnumeric ancestor
   IDs. Shuffled JSONL completion order must give identical ordered output.
   Missing/duplicate/unknown keys, invalid candidate indices, and a stale NPZ
   must fail clearly. Invalid cutoff/probability settings must fail explicitly.
9. Run `uv run ruff check lib`, `uv run ruff format --check lib`, and
   `uv run snakemake --cores all --dry-run`. Use an isolated temporary manifest
   containing the existing zero-error, genotype-error, and no-truth n300 inputs.
   Expect three inferred statistics files, each with 3,000 rows. The default
   DAG must contain no true outputs or truth-dependent extraction jobs.
10. Compare every sample/cutoff metric with the independent oracle. Assert
    unique row keys, exact row counts, coverage/candidate monotonicity, bounded
    fractions, and fixed denominators/switches/mismatches/score across cutoffs.
    In the zero-error inferred example, each complete path evaluates 918,392
    bases and 6,120 sites, and total sample mismatches are zero.
11. Change only cutoffs, adding 2 and 20, and confirm only statistics rerun;
    shared cutoff rows must be unchanged. Separately change HMM probabilities
    and verify matching and scoring are regenerated consistently. Profile
    evaluation on its own to confirm streaming memory and one record read
    regardless of cutoff count.

Read-only checks performed for this revision confirmed all 793 multi-focal
inferred ancestors have identical observed focal counts, AFs, and carrier masks,
and that their AFs equal node times on this fully called dataset. All 600 native
paths satisfy the segment-count switch identity and use one sample match group.
Ancestor metadata joins bijectively to all 4,639 panel IDs. The checked sources
show `n=4,641`, `mu=1e-20`, `rho=0.01`, and no public switch-count accessor.

For those parameters, the mismatch log penalty is approximately -46.051702
and the switch log penalty -13.037807. Current sample scores range from
-691.003780 to zero; their sum is approximately -165,541.037585. A separate
hand-calculated probability-ratio fixture passed. These are observations about
the current matching products, not invariant expected values across future
tsinfer revisions.

The earlier read-only coverage oracle remains applicable to this inferred
dataset because the former per-carried-site minimum equals the single ancestor
count. A second read-only probe used the simplified panel-level cutoff bins
and matched all 3,000 sample/cutoff results against independently filtered
candidate sets, including both coverage numerators and candidate counts. Its
pooled fractions were:

| Cutoff | Fraction covered bases | Fraction covered sites |
| --- | ---: | ---: |
| 3 | 0.456260 | 0.457898 |
| 5 | 0.562427 | 0.564406 |
| 10 | 0.662139 | 0.664306 |
| 100 | 0.783609 | 0.785775 |
| 600 | 0.798774 | 0.800666 |

Implement in order: count annotation and dataframe fields; inferred NPZ scalar
counts; the small evaluation module; explicit HMM settings and inferred-only
workflow targets; documentation and independent verification. Completion means
all inferred datasets produce the specified sample-by-cutoff rows with correct
coverage and likelihood, default execution performs no true work, count fields
are available for both dataframe kinds, and cutoff edits reuse matching.

Future true-panel evaluation must define a count/eligibility policy for its
heterogeneous multi-focal ancestors before reactivating true statistics. Keep
that policy and any additional diagnostics outside this initial implementation.
