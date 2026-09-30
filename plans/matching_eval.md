# Matching evaluation implementation plan

## Scope and result

Add a diagnostic after focal lookup and full-panel sample matching. For each
dataset, panel kind, sample haplotype, and configured allele-count cutoff, report
how much of the existing copying path uses that haplotype's eligible focal
ancestors. Also report switches and mismatches on the existing path. Changing
the cutoff changes the classification of a parent, not the HMM result: no
restricted-panel matching or additional inference is required.

The implementation described here will produce
`{data_dir}/dataframes/{name}_{kind}_focal_ancestor_stats.csv`. Each panel's
dataframe has exactly `num_haplotypes * len(ac_cutoff)` rows. A sample means a
haplotype throughout this diagnostic; diploid individuals have separate rows
for `ploidy_index=0` and `ploidy_index=1` at every cutoff.

This planning task changes only this file. The implementation work below is
future work. It follows the existing contracts in
[matching_setup.md](matching_setup.md), [initial_mvp.md](initial_mvp.md), and
[../AGENTS.md](../AGENTS.md).

## Decisions needed for meaningful statistics

### Allele count versus allele frequency

Add the following required top-level setting to both `config.yaml` and
`config.yaml.example` when implementing:

```yaml
ac_cutoff: [3, 5, 10, 100, 600]
```

These are inclusive, cumulative allele-count cutoffs. A cutoff of 3 includes
eligible focal associations at observed doubleton and tripleton sites; 5
includes those plus four- and five-carrier sites. These are not disjoint bins.
Require a nonempty list of positive integers in strictly increasing order.
Reject booleans, duplicates, descending values, and noninteger values with a
`ValueError` naming `ac_cutoff`. Cutoffs above the dataset's haplotype count are
valid and include all otherwise eligible candidates. Keep defaults in YAML,
not Python constants.

The current true-ancestor CSV's `derived_af` is a proportion, calculated as
`derived_count / called_count`, not a count. Comparing it directly with 3 or 5
would include every focal site. Multiplying by the total number of haplotypes
would also be wrong at sites with missing calls.

Use exact observed derived counts on the shared panel site axis. In
`find_focal_ancestors`, the sample calls have already been read and polarised
into a boolean `derived` array of shape `(num_sites, num_samples, ploidy)`.
Compute `site_derived_ac = derived.sum(axis=(1, 2))` before clearing excluded
sites. This gives the count underlying `derived_af` without another genotype
read, floating-point rounding, or any dependency on truth availability. Keep
`derived_af` and the true CSV unchanged. An independent validation can check
that `derived_af * called_count` agrees with these counts, using the existing
sample-store annotation. Both ancestral-state modes must use allele strings,
as focal lookup already does.

### Multi-focal ancestors and the sample-specific threshold

An ancestor may have several focal sites. Their allele counts can differ,
particularly for true ancestors formed by grouping sites selecting the same
older truth node. Define eligibility on the focal associations carried by the
particular sample haplotype:

1. Retain the existing eligible focal relations. Inferred relations come from
   all non-padding panel focal positions. True relations come from CSV rows
   with `true_mutation_is_singleton=False`.
2. Retain relations whose focal derived allele the sample haplotype carries.
   Missing and excluded calls do not contribute a relation.
3. For each distinct candidate ancestor, retain the minimum `site_derived_ac`
   among those carried focal relations.
4. At cutoff `f`, the ancestor is a focal candidate exactly when this minimum
   count is at most `f`.

Equivalently, filter focal-site relations by count before forming the sample's
distinct candidate set. Do not take the minimum over all focal sites of an
ancestor before checking sample carriage. For example, if an ancestor has
focal counts 3 and 100, a sample carrying only the latter is eligible starting
at 100; a sample carrying both is eligible starting at 3. The parent is counted
once regardless of how many qualifying focal alleles are carried.

Eligibility applies to the parent across the evaluated portions of its copying
segments, not only at the qualifying focal site's position. This implements
the requested fraction copied from a subset of focal ancestors.

For a 600-haplotype dataset, cutoff 600 reproduces the existing complete focal
candidate sets. For larger datasets it need not include every candidate. True
singleton-origin rows remain excluded even if their observed allele count is
two or more due to recurrence or genotype errors.

### Evaluated bases and sites

Use the raw ancestor reference TS to supply both `sites_position` and its
top-level `sequence_intervals` metadata. Those intervals identify the regions
containing inference sites. They are sorted, disjoint, half-open intervals and
are preserved from the panel by the current matching code.

For sample haplotype `h`, define its evaluated region as the intersection of
the union of its matched path segments with the union of `sequence_intervals`.
Intersect each path segment with those intervals before counting bases or
sites. This also handles paths shortened by leading or trailing missing calls.
Uncovered regions do not enter either denominator. Disjoint evaluation
intervals do not cause intervening gaps to enter the denominator.

This clipping matters for the checked-in example: the TS's sequence length is
64,444,167, while the evaluation interval is `[81,353, 999,745)`, spanning
918,392 bases. Native paths can start at zero and end at the full contig length.
Using their untrimmed lengths would largely measure copying through a padded
region with no evaluated sites.

For a clipped fragment `[left, right)`, use:

```text
fragment_bp = right - left
left_site = searchsorted(sites_position, left, side="left")
right_site = searchsorted(sites_position, right, side="left")
fragment_sites = right_site - left_site
```

A site exactly at `left` belongs to the fragment; one exactly at `right` belongs
to the next fragment. Site fractions must count these sites explicitly: a
base fraction cannot be rescaled into a site fraction because site density
varies. Site denominators cover the reference inference sites reached by the
path, including internal sites with missing sample calls. This measures parent
selection; it does not claim that every counted site supplies an observed
genotype. Input-only sites absent from the reference are not evaluated.

Synthetic-root copying contributes to the denominators and never to the focal
numerators. An empty candidate set therefore gives zero fractions whenever
the corresponding denominator is positive. A zero denominator gives `NaN`
for that fraction and zero for its numerator, making absence of evaluated data
distinct from an evaluated path with no focal copying.

### Switches and mismatches

Count switches on the original complete matched path, once per haplotype.
Sort segments by genomic left coordinate: native JSONL paths are usually in
reverse genomic order. Count a switch at a boundary where two consecutive
segments touch and their parent node IDs differ. Adjacent segments with the
same parent add no switch; do not infer a switch across an uncovered gap.
Overlapping segments are an invalid copying path and must raise an error.

This is a count on the recorded path, so it can include a switch outside the
evaluation intervals. The base and site fractions use the clipped region.
Document that distinction explicitly. Do not derive switches from the clipped
fragments, whose extra boundaries can arise merely from region clipping.

Set `num_mismatches = len(record["mutations"])`. The local tsinfer matcher
emits exactly the nonmissing sample calls that disagree with the selected
parent haplotype within the matching range. These events include either
canonical allele state; do not count only derived-state 1 or use the raw TS's
total mutation count, which also contains ancestor mutations. Mismatches and
switches repeat unchanged for every cutoff of the same sample haplotype.

Frequency is a proxy for recent origin, not a direct age measurement. Focal
associations at recurrent sites describe allele sharing, as in existing focal
lookup. The diagnostic measures the selected copying path rather than a
posterior probability or the sample's true genealogical parent.

## Data contracts

### Extend the existing focal NPZ

Continue writing one compressed NPZ per dataset and panel kind, at the existing
path. Preserve the current row order and array meanings. Add one aligned array:

| Array | Shape and dtype | Meaning |
| --- | --- | --- |
| `sample_id` | `(h,)`, Unicode | Existing sample ID for each haplotype row |
| `ploidy_index` | `(h,)`, integer | Existing chromosome index |
| `ancestor_id` | `(a,)`, Unicode | Existing panel IDs in Zarr column order |
| `offsets` | `(h + 1,)`, int64 | Existing boundaries for candidate slices |
| `ancestor_index` | `(k,)`, int64 | Existing sorted distinct panel columns |
| `min_focal_derived_ac` | `(k,)`, int64 | Minimum count of a carried eligible focal relation for each aligned candidate |

For row `i`, slice both candidate arrays with
`offsets[i]:offsets[i + 1]`. Their entries correspond element for element.
Rows still follow sample-store order then increasing ploidy index. Equal
offsets preserve empty candidate sets. Save all eligible candidates, irrespective
of the configured cutoffs; modifying `ac_cutoff` must not require regenerating
this NPZ.

Construct each row using a dictionary from ancestor column to minimum count.
Sort its column keys once, then build both arrays in that order. This replaces
the current collect-and-unique step while preserving candidate membership and
ordering. Several relations for the same candidate update one dictionary
entry. No sample-by-ancestor dense matrix, per-cutoff candidate files, or
pickled object arrays are needed. Load using `allow_pickle=False`.

Existing NPZs have no count array and must be regenerated with
`--forcerun find_focal_ancestors`. The new loader must fail clearly when the
array is missing, with that rebuild instruction. Do not invent thresholds from
an ancestor-global minimum or add a legacy-format adapter.

### Join parents and sample records explicitly

Build the parent mapping once from the ancestor reference TS's node metadata:

```text
(source="ancestors", sample_id, ploidy_index=0)
    -> ancestor_id entry
    -> panel column index
```

Node IDs, panel column indices, and numeric suffixes of ancestor ID strings
are different namespaces. Use an `int64` node-to-column array indexed by TS
node ID; represent the reference's metadata-free synthetic roots with a
noncandidate sentinel. Every panel ancestor ID must join to exactly one
reference ancestor node. Missing, duplicate, or unexpected source metadata is
an error indicating inconsistent inputs. Roots must be identified from the
actual reference metadata, not by assuming two roots or subtracting a root
count. The current workflow disables path compression and has haploid panel
entries, so no additional node classes need a compatibility policy.

Use `(source, sample_id, ploidy_index)` to join each JSONL record to the focal
row. The current pipeline has one sample source named `samples`; prefix NPZ
row keys with that source. Never join by file line number, `group`, or
`haplotype_index`, since threaded matching can change completion order.

Require exactly one record per focal row. Reject duplicate records and unknown
keys when encountered, and missing records at end of file. Include the sample
key in errors. Reject out-of-range parent node IDs. Validate NPZ offset
boundaries, aligned array lengths, sorted distinct in-range candidate columns,
and positive candidate counts on load. These checks enforce joins and ragged
array invariants already used by this pipeline, without adding a generic file
validation framework.

### Returned dataframe and saved CSV

`compute_focal_ancestor_stats` returns the following columns, in this order:

| Column | Type | Meaning |
| --- | --- | --- |
| `source` | string | `samples` for this workflow |
| `sample_id` | string | Actual sample-store ID |
| `ploidy_index` | integer | Haplotype index within the sample |
| `ac_cutoff` | integer | Inclusive allele-count threshold |
| `evaluated_bp` | float64 | Total clipped path span, including root copying |
| `focal_bp` | float64 | Clipped span whose selected parent is an eligible focal candidate |
| `fraction_focal_bp` | float64 | `focal_bp / evaluated_bp`, or `NaN` for zero denominator |
| `evaluated_sites` | int64 | Reference sites covered by the clipped path |
| `focal_sites` | int64 | Such sites copied from eligible focal candidates |
| `fraction_focal_sites` | float64 | `focal_sites / evaluated_sites`, or `NaN` for zero denominator |
| `num_switches` | int64 | Parent changes on the original path |
| `num_mismatches` | int64 | Native mismatch events for this sample |
| `num_focal_ancestors` | int64 | Distinct eligible candidates at this cutoff, including candidates not selected on the path |

The numerators, denominators, and candidate count are useful diagnostic
variables beyond the four requested metrics. They make the fractions auditable
and distinguish no candidates from candidates that are never selected. They
require no additional matching or genotype reads.

Return rows in focal-NPZ row order, with cutoffs increasing within each row.
At the workflow boundary, prepend `dataset` and `panel_kind` columns using the
wildcards, then write CSV with `index=False`. Thus each saved file is
self-describing and can be concatenated across datasets and panels. Preserve
sample IDs as strings when reading it. Use standard pandas missing-value CSV
encoding for undefined fractions. Keep all file and column names lower case
with underscores.

## Functions and responsibilities

Keep focal construction and matching in `lib/matching.py`. Put the new
diagnostic in `lib/evaluation.py` to keep statistics independent of running the
HMM. Use module imports, one module logger, module-level imports, pathlib
paths, PEP 604 annotations, and linked docstrings in both modules.

| Function or dataclass | Inputs | Output and responsibility |
| --- | --- | --- |
| `matching._inferred_focal_dataframe` | `ancestors_path: pathlib.Path` | Existing two-column focal-relation dataframe; its contract remains unchanged |
| `matching.find_focal_ancestors` | Existing sample/panel/output paths, ancestral-state mapping, optional true CSV path | Writes the extended NPZ; computes exact site counts from already polarised calls and reduces carried relations to sample-specific candidate thresholds |
| `evaluation._validate_ac_cutoff` | `ac_cutoff: list[int]` | Sorted-as-supplied int64 array after enforcing the configuration contract; no implicit sorting or defaults |
| `evaluation.FocalCandidates` | Loaded NPZ arrays | Dataclass containing `sample_id`, `ploidy_index`, `ancestor_id`, `offsets`, `ancestor_index`, and `min_focal_derived_ac` |
| `evaluation._load_focal_candidates` | `focal_ancestors_path: pathlib.Path` | `FocalCandidates`, detached from the closed NPZ handle, with storage invariants checked |
| `evaluation._build_parent_columns` | Raw reference `ts: tskit.TreeSequence`, `ancestor_id: np.ndarray` | Node-to-panel-column array, validating the metadata join and distinguishing synthetic roots |
| `evaluation.PathArrays` | Numeric arrays `left`, `right`, `parent` | Dataclass for a genomic-order path or clipped fragments; arrays are aligned |
| `evaluation._ordered_path` | Native record `path: list[dict]`, reference sequence length and node count | `PathArrays` sorted by `left`, validating bounds, positive spans, parents, and nonoverlap |
| `evaluation._clip_path` | Ordered `PathArrays`, reference `sequence_intervals: np.ndarray` | `PathArrays` containing only nonempty intersections, using a two-pointer scan |
| `evaluation.HaplotypeStats` | Counts and cutoff vectors | Dataclass containing `evaluated_bp`, `evaluated_sites`, `num_switches`, `num_mismatches`, and arrays `focal_bp`, `focal_sites`, `num_focal_ancestors` of length `num_cutoffs` |
| `evaluation._compute_haplotype_stats` | One native record, candidate columns/counts for its focal row, validated cutoff array, site positions, intervals, parent mapping, reference sequence length | `HaplotypeStats`; traverses this path once to accumulate all cutoffs and path diagnostics |
| `evaluation.compute_focal_ancestor_stats` | `focal_ancestors_path: pathlib.Path`, `ancestors_ts_path: pathlib.Path`, `match_file_path: pathlib.Path`, `ac_cutoff: list[int]` | Requested `pd.DataFrame`; loads shared context once, streams records, verifies key completeness, and orders rows deterministically |

Expose `compute_focal_ancestor_stats` as a module function, matching the existing
functional library API. It returns data; writing the output belongs to the
Snakemake rule. No class wrapping the pipeline or new global configuration
object is required. Dataclasses replace multi-value tuple returns, not simple
single-array results.

The reference must have increasing site positions and valid, ordered,
nonoverlapping evaluation intervals within its sequence length, covering its
site axis. Validate this shared context once. The reference already provides
the correct site positions and ancestor metadata, so the evaluation rule needs
neither the raw sample TS nor the panel/sample Zarrs. Library code must not
rewrite either input TS or the JSONL.

## Single-pass accumulation

### Shared setup

1. Validate cutoffs and load the focal NPZ.
2. Load the raw ancestor TS once. Obtain `sites_position`, evaluation
   intervals, sequence length, and the parent-column lookup.
3. Build the sample-key-to-focal-row mapping once. Allocate a seen-row boolean
   array and result slots in focal-row order.
4. Stream the match JSONL. Keep only one decoded match record at a time; do
   not create a dataframe of all segments or load the complete JSONL.

### Per-haplotype calculation

1. Slice the focal candidate columns and their minimum counts using the row's
   offsets. For each count `c`, compute its first eligible cutoff index using
   `searchsorted(cutoffs, c, side="left")`. An index equal to the number of
   cutoffs means that candidate contributes to none of them.
2. Build a small dictionary from candidate column to first eligible cutoff
   index. This depends only on that sample's candidates. It avoids clearing a
   panel-sized array for each haplotype.
3. Normalise path order and count actual parent switches on the original path.
   Read the mismatch count once from the mutation list.
4. Clip the path against evaluation intervals with a two-pointer scan. A
   segment can produce several fragments when it spans disjoint intervals.
   Advance through segments and intervals monotonically, avoiding a nested
   comparison of every segment with every interval.
5. Compute fragment lengths and site counts. Apply vectorised `searchsorted`
   to the left and right boundary arrays. Sum all fragments for the two
   denominators, including fragments whose parents are roots or noncandidates.
6. Translate parent nodes through the shared node-to-column array and then
   the sample's candidate dictionary. Add each eligible fragment's length
   and site count to the bucket of its first eligible cutoff. The sample's
   candidates similarly supply a bucketed candidate count.
7. Take cumulative sums of the buckets to obtain all cutoff numerators and
   candidate counts. Use float64 buckets for bases and int64 buckets for
   sites/candidates. For integer additions use `numpy.add.at` or a simple
   loop; do not silently convert large site counts to float weights.
8. Store one `HaplotypeStats`. After every record has been seen exactly once,
   create output rows in focal order, calculate fractions with explicit
   zero-denominator handling, and construct the final dataframe.

One parent with several copied segments contributes each segment's span and
sites, but one candidate in `num_focal_ancestors`. Nothing is rescanned for
smaller cutoffs. There is no assumption that the largest configured cutoff
includes all focal ancestors.

Let `h` be haplotypes, `a` panel ancestors, `n` reference nodes, `s` sites,
`k` total candidate associations, `p` total path segments, `q` total clipped
fragments, and `f` cutoffs. Shared setup is linear in the loaded arrays and
metadata. Per-path sorting costs the sum of `p_h log(p_h)`. Candidate binning
costs `O(k log f)`, fragment site lookups cost `O(q log s)`, and cumulative
output costs `O(h f)`. Interval clipping uses a linear sweep per path.
Memory is `O(n + a + s + k + h f)` plus one current path and its fragments.
There is no `O(h a)` membership matrix or `O(f p)` path traversal.

Start with NumPy and a serial streaming evaluator. The read-only probe below
processed each complete current example panel, including reconstructing focal
counts from genotypes, in about one second. A separate Numba dependency or
process queue would add complexity without evidence of a long-running step
here. Snakemake can schedule independent dataset/panel evaluation jobs across
the allocated cores. If profiling larger real workloads shows a long-running
per-panel calculation, follow AGENTS.md by distributing independent record
batches through multiprocessing queues across the allocated CPU workers.
Load shared context once per worker, keep queues bounded, assemble results by
focal row, and propagate worker failures. Add a worker-count setting in YAML
only when implementing that measured need. Do not repeatedly pickle the full
candidate arrays or reference TS per record.

## Snakemake integration and rebuilds

Add `evaluation` to the existing `from lib import ...` import in `Snakefile`.
Add one rule named `compute_focal_ancestor_stats` with:

| Rule element | Value |
| --- | --- |
| `input.focal` | `{data_dir}/focal_ancestors/{name}_{kind}_focal_ancestors.npz` |
| `input.reference` | `{data_dir}/ancestors/{name}_{kind}_ancestors.trees` |
| `input.matches` | `{data_dir}/matches/{name}_{kind}_samples_matches.jsonl` |
| `output` | `{data_dir}/dataframes/{name}_{kind}_focal_ancestor_stats.csv` |
| `params.ac_cutoff` | `config["ac_cutoff"]` |
| `log` | `{progress_dir}/compute_focal_ancestor_stats/{name}_{kind}_compute_focal_ancestor_stats.log` |

The run block calls `utils.setup_log`, passes pathlib paths and the configured
cutoff list to `evaluation.compute_focal_ancestor_stats`, prepends the dataset
and panel-kind columns, and writes the returned dataframe. Log haplotype and
cutoff counts, output rows, and the destination. Use Snakemake's existing
directory/output handling; no additional CLI wrapper or rule for each cutoff
is necessary.

Extend `rule all` with one statistics CSV for each entry in the existing
`panels` collection. The true branch remains present only for datasets with
truth; the evaluation code itself is panel-independent. Keep the existing
`find_focal_ancestors` and `match_samples` branches independent until this rule
joins their products.

Putting the cutoff list in `params` records this configuration dependency, so
changing cutoffs reruns statistics without rebuilding focal candidates or
matching. Do not add library source files as Snakemake inputs, consistent with
the established pipeline. Because existing NPZs lack the new count array,
perform the first migration using:

```sh
uv run snakemake --cores all --forcerun find_focal_ancestors compute_focal_ancestor_stats
```

When only evaluation code changes later, force only
`compute_focal_ancestor_stats`. Focal lookup changes require regenerating its
NPZs and downstream statistics. Existing ancestor inference, truth extraction,
TOMLs, ancestor TSs, raw sample TSs, and match files can be reused because their
contracts are unchanged by this diagnostic.

Update README to describe the cutoff configuration, cumulative eligibility,
new NPZ array, CSV columns, evaluated-region definition, sample key, and force
commands. During implementation, amend matching_setup.md's focal-file schema
to link to this extension. Do not alter dependencies for the initial version.

## Internal verification

Use read-only `uv run` probes and small in-memory synthetic cases; no permanent
unit-test suite or pytest dependency is needed for this implementation. Keep
the verification oracle deliberately independent: for each cutoff, explicitly
filter that sample's focal relations and inspect its intervals/site positions.
The production calculation uses buckets and cumulative sums; the oracle may
repeat work on tiny cases to make correctness easy to see.

### Focal-threshold checks

| Case | Required assertion |
| --- | --- |
| Multi-focal counts 3 and 100 | A haplotype carrying only the count-100 allele has threshold 100; carrying only count-3 or both has threshold 3 |
| Several carried focal sites for one ancestor | One sorted candidate entry with its minimum carried count; no duplicate candidate or duplicated copied span |
| Site with 3 derived calls, 10 called haplotypes, 20 total haplotypes | Threshold is exactly 3 although `derived_af=0.3`; multiplying AF by 20 would incorrectly yield 6 |
| Reference allele is derived | Allele-string polarisation yields the same count and candidates as an equivalent ancestral-first encoding |
| Missing focal call and excluded focal site | Neither contributes a candidate, even if an ancestor has another qualifying focal site not carried by this sample |
| True singleton origin at an observed doubleton site | Its flagged CSV row contributes no ancestor; site remains on the reference axis |
| Empty candidate row | Equal offsets, empty aligned count slice, and all positive-denominator focal fractions equal zero |
| Cutoff 3, 5, 10 | Counts exactly equal to the cutoff are included; a count of 4 first appears at 5; a count above the maximum contributes nowhere |
| No truth dataset | Counts come from observed calls and inferred relations without reading any true CSV |

Compare the old and extended NPZ candidate IDs, row keys, offsets, and sorted
candidate arrays on both actual panels. They must be identical; only the new
aligned count array is added.

### Path and interval checks

Use a tiny hand-checkable fixture with positions `[10, 20, 30, 50]`, evaluation
interval `[10, 51)`, and a reverse-order native path whose genomic-order
segments are `[0, 20)` from ancestor A, `[20, 40)` from ancestor B, and
`[40, 100)` from a root. Candidate A has threshold 3 and B has threshold 5.

Expected denominators are 41 bases and 4 sites. At cutoff 3, focal totals are
10 bases and 1 site; at cutoff 5 and above they are 30 bases and 3 sites.
Fractions are therefore `10/41` and `1/4`, then `30/41` and `3/4`. The switch
count is 2. An arbitrary two-entry mismatch list gives 2 mismatches at every
cutoff. The site at 20 belongs to B, demonstrating half-open boundaries and
why base and site fractions differ.

Also check:

- Splitting A into adjacent same-parent segments leaves all metrics unchanged
  and adds no switch. Reusing A later in the path adds a switch and copied span,
  but does not add another candidate.
- Several disjoint evaluation intervals omit gap lengths and gap sites. A
  segment spanning several intervals is clipped into several fragments without
  adding switches to the original path.
- A switch entirely outside the evaluation intervals still appears in
  `num_switches`, as specified by the full-path definition.
- Leading and trailing uncovered regions are absent from denominators. Internal
  missing sample calls do not remove a reference site covered by a path.
- No evaluated intersection yields zero numerators/denominators and `NaN`
  fractions. A fragment with bases but no reference sites produces a defined
  base fraction and an undefined site fraction.
- A single-parent path has zero switches. Empty paths have zero switches and
  denominators; an empty file with expected sample rows is a missing-record
  error. Path overlap and out-of-range parents fail explicitly.
- A deliberately shuffled node-to-panel relationship, with nonnumeric IDs,
  yields the correct result through metadata. Root copying remains nonfocal.
- Shuffled JSONL records produce the same deterministic dataframe. Duplicate,
  unknown, and missing sample keys fail. Old NPZs fail with the focal rebuild
  instruction; malformed offsets or misaligned counts fail on load.
- Invalid cutoff lists fail before match-file streaming. A cutoff above the
  haplotype count is valid and saturates the complete candidate set.

For every evaluated sample and cutoff, assert bounds on numerators and
fractions, monotonic focal totals/candidate counts with increasing cutoff,
identical denominators and path diagnostics across cutoffs, unique output keys,
and the exact expected row count. Compare site counts against explicitly
enumerated reference positions rather than another searchsorted expression.

### Existing-example integration

1. Run `uv run ruff check lib` and `uv run ruff format --check lib`, then
   `uv run snakemake --cores all --dry-run`.
2. Use a temporary YAML manifest with separate output/progress directories and
   the existing zero-error n300 input, genotype-error n300 input, and a no-truth
   copy. Set the five requested cutoffs. Reuse existing compatible match
   products where available; run the pipeline to create missing prerequisites
   in the isolated integration directory. Do not modify source stores or the
   truth TS.
3. Expect 3,000 dataframe rows per panel for 600 haplotypes and five cutoffs.
   Expect five panels across the three datasets: inferred and true for the
   two truth datasets and inferred only for no truth. Confirm schema, string
   IDs, ordering, uniqueness, and aligned candidate counts.
4. Compare every row with a straightforward per-cutoff reference calculation.
   On these examples this is small enough to validate all haplotypes, rather
   than relying on aggregate monotonicity alone. For the zero-error panels,
   all complete paths evaluate 918,392 bases and 6,120 sites per haplotype.
5. Sum `num_mismatches` once per haplotype, not across repeated cutoff rows.
   The checked-in zero-error inferred matches contain 0 events; the true
   matches contain 6. Independently compare path switches and mutations with
   native JSONL and, for selected samples, raw sample TS edges and mutations.
6. Modify only `ac_cutoff`, for example adding 2 and 20, and dry-run/run the
   workflow. Only statistics outputs should be invalidated. Confirm that
   cutoffs shared between runs have identical values and the row counts change
   as expected. No-truth evaluation must request no true dependencies.
7. Time evaluation separately from matching and focal lookup. Confirm it reads
   each match record once and remains insensitive to the number of cutoffs
   except for candidate binning and the required output size. Compare peak
   memory with NPZ/reference/output sizes; no complete match-file dataframe
   should appear in memory. Use profiling evidence before considering Numba
   or multiprocessing.

### Read-only experiments performed while preparing this plan

Inspected the current local tsinfer implementation and existing default n300
products. The matcher emits mismatch events for nonmissing disagreements, and
the pipeline writes parent node IDs and native path order directly. Both
ancestor references have 6,120 sites, sequence length 64,444,167, and the
single evaluation interval `[81,353, 999,745)`. There are 793 multi-focal
inferred ancestors and 1,167 multi-focal true ancestors. Native paths are in
descending genomic order for 580 inferred and 582 true records; one-segment
paths do not require reordering.

A read-only prototype rebuilt the sample-specific minimum carried focal counts
from sample genotypes, reproduced all 600 existing candidate sets for each
panel, and checked the `derived_af * called_count` relationship. The input had
no missing calls. It then clipped paths and computed all requested cumulative
cutoffs. Rounded pooled fractions were:

| Panel | Cutoff | Fraction focal bases | Fraction focal sites |
| --- | ---: | ---: | ---: |
| inferred | 3 | 0.456260 | 0.457898 |
| inferred | 5 | 0.562427 | 0.564406 |
| inferred | 10 | 0.662139 | 0.664306 |
| inferred | 100 | 0.783609 | 0.785775 |
| inferred | 600 | 0.798774 | 0.800666 |
| true | 3 | 0.398478 | 0.400293 |
| true | 5 | 0.471093 | 0.473373 |
| true | 10 | 0.551720 | 0.554388 |
| true | 100 | 0.650127 | 0.653713 |
| true | 600 | 0.658269 | 0.661876 |

These are total focal bases/sites divided by total evaluated bases/sites,
pooling haplotypes. They are reference observations for the current matching
products and the specified semantics, not universal expected values for future
tsinfer revisions. The aggregate switch counts were 12,697 inferred and 15,736
true; mismatch counts were 0 and 6. Runtime including focal reconstruction was
about 0.99 seconds inferred and 0.81 seconds true. The prototype used a direct
per-cutoff calculation to provide an independent oracle; the implementation
will use the bucket accumulation described above.

A separate in-memory check verified the hand-calculated 41-base/four-site
fixture and its integer bucket accumulation. Both real references also passed
checks that their ancestor metadata maps bijectively to panel IDs, all panel
ancestors are haploid, and reference positions equal panel positions. No
production code, configuration, source data, or existing output was changed
by these experiments.

## Implementation sequence and completion criteria

1. Extend `matching.find_focal_ancestors` with exact site counts and minimum
   carried counts. Preserve existing focal relationships and candidate ordering.
2. Add `lib/evaluation.py` with the data contracts, metadata joins, clipping,
   and single-pass cumulative calculation described above. Verify the synthetic
   cases before integrating output writing.
3. Add the required YAML setting, the new rule, its cutoff parameter and logs,
   and the `rule all` targets. Keep diagnostic work downstream of matching.
4. Update README and the focal-storage section of matching_setup.md with the
   implemented contracts and rebuild instructions.
5. Run style checks, dry-run the DAG, rebuild focal files, and perform the
   independent example comparisons. Record actual verification results and
   timings in the implementation report.

The implementation is complete when every configured panel produces the
requested number of sample-by-cutoff rows; cutoff eligibility is correct for
multi-focal and missing-call cases; base/site fractions use the correct region
and half-open boundaries; switches and mismatches agree with native paths;
root and ID joins are correct; cutoff changes reuse matching products; and
the independent validation agrees with all output metrics.

Possible later diagnostics include the fraction copied from roots, the number
of distinct eligible focal parents actually selected, mismatches per evaluated
site, and direct ancestor-time stratification. They can answer different
questions but are not necessary to deliver this requested diagnostic.
