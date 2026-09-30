# Focal ancestor haplotype chunks

## Scope and deliverable

Implement the request in [haplotype_compare_prompt.md](haplotype_compare_prompt.md)
as a new inferred-only analysis branch. For each sample **haplotype**, compare
its sequence directly with its eligible inferred ancestor haplotypes, starting
at each carried focal site. Produce bounds for every mismatch budget from zero
through the configured maximum. Compute once at the largest allele-count cutoff;
smaller cutoffs are dataframe filters.

This document is the deliverable for the present planning task. The prompt
allows read-only experiments and explicitly says not to change other files.
The experiments below read existing stores and construct temporary arrays or an
in-memory dataframe. No production code, configuration, input store, or pipeline
output was changed, and no chunk CSV was written. The implementation and dataset
generation steps below remain to be executed in a subsequent implementation task.

The implementation adds `lib/haplotypes.py` and the Snakemake rule
`construct_focal_ancestor_chunks`. Each dataset gets one combined dataframe:

```text
{data_dir}/dataframes/{name}_inferred_focal_ancestor_chunks.csv
```

There are no ancestor TS, sample TS, match-file, HMM, or truth inputs to this
rule. Existing matching and evaluation remain independent branches. Do not add
coverage plots, HMM pruning, topology comparisons, or a true-panel version to
this change.

## Interpretation of the supervisor's proposal

The relevant source is the “Matching-engine performance” section of
[the 10 September meeting notes](</Users/duncan/exo/documents/meeting_notes/2026-09-10 (Meeting with Gil).md>),
particularly “Using rare variants as an index”, “Coverage and fallback to the
full HMM”, and “Ancestor spans and excluding obviously irrelevant ancestors”.

Gil's proposal is to use shared rare mutations as an index into recent
ancestors, extend their chunks until a mismatch criterion is reached, and later
measure coverage as increasingly common variants are admitted. Gaps could
eventually use the full ancestor set. The notes also mention using an ancestor's
full span as the simplest initial baseline. The present prompt specifically
requests direct haplotype comparison and trimming, so implement that refinement.
The output retains enough information to filter by focal AC and inspect the
resulting spans without involving matching.

### Mismatch budget

The two independent sweeps described in the prompt naturally define **a budget
per direction**. This plan uses that interpretation: a row with
`max_mismatches = k` tolerates at most `k` called mismatches to the left of its
focal site and at most `k` to the right. Its combined interval can therefore
contain up to `2k` mismatches. The focal genotype must match and uses neither
budget. Zero means a perfect match throughout the retained interval.

An optional clarification about this interpretation was requested during
planning; no answer had arrived when this plan was written. This is an explicit
planning assumption, not a confirmed preference. Before implementation, apply
any subsequent clarification to this contract. A budget of `k` across the entire
interval is a different specification: directional budgets would need to be
allocated between the two sides. The sweeps can support all allocations, but
there is no unique combined interval without choosing an allocation or a
selection criterion. Do not silently describe the independently extended
interval as containing at most `k` total mismatches.

There is also an off-by-one correction to the prose algorithm. To retain a
chunk with **up to** `k` mismatches, find and exclude the `(k + 1)`th mismatch.
Stopping after recording only `K` mismatches gives budgets `0..K-1`. For the
requested rows `0..K`, record up to `K + 1` mismatch positions per direction.
This still requires only one traversal in each direction.

### One seed per carried focal site

Use one seed for each `(sample_id, ploidy_index, ancestor_index,
focal_site_index)` association. Multi-focal ancestors contribute multiple seeds.
Their bounds may differ even for the same sample and ancestor, so selecting
only the first focal site would discard requested information.

The existing NPZ contains distinct ancestor candidates, not focal-site
associations. Expand each candidate using the panel's non-padding
`sample_focal_positions`. Inferred ancestors group focal sites with identical
sample genotype patterns. Consequently a sample carrying one of those focal
alleles carries all of them, and focal AC and AF are shared across that
ancestor's focal sites. This was verified on both datasets below. Keep the
individual focal sites in the result even when their output bounds coincide.

Do not change the existing NPZ schema or deduplicate output by ancestor alone.

## Existing data and input contracts

The relevant implementation is:

- [lib/matching.py](../lib/matching.py): `find_focal_ancestors` writes the ragged
  candidate NPZ; `_inferred_focal_dataframe` reconstructs focal relations.
- [lib/utils.py](../lib/utils.py): `add_singleton_mask` retains original calls
  and annotates observed derived AC, AF, and the singleton mask.
- [Snakefile](../Snakefile): the existing inferred-panel and focal-lookup rules.
- The editable sibling checkout's `tsinfer/vcz.py`: read-only `open_store`,
  canonical genotype encoding, ancestor Zarr metadata, and physical chunking.
- The sibling checkout's `tsinfer/ancestors.py`: focal-pattern grouping and
  inference intervals.

The new rule reads exactly these existing products:

| Input | Required information |
| --- | --- |
| `{name}_samples_masked.zarr` | `sample_id`, original `call_genotype`, positions, alleles, ancestral-state annotation, derived AC/AF |
| `{name}_inferred_ancestors.zarr` | Canonical haplotypes, positions, panel IDs, focal positions, ancestor spans, `sequence_intervals` |
| `{name}_inferred_focal_ancestors.npz` | Haplotype IDs/ploidy, ragged candidate columns, panel IDs, panel-aligned `derived_ac` |

Open Zarr stores read-only with `tsinfer.vcz.open_store`. Use the installed Zarr
v3 APIs, including explicit orthogonal selection through `.oindex` where needed.
An `ancestor_index` is a Zarr column, and `ancestor_id` is its `sample_id` string.
Neither is a tree-sequence node ID.

### Align sites and polarise sample calls

Let `sites_position` be the inferred panel's `variant_position` array. This is
the complete comparison axis; excluded singletons and other sample-only sites
do not participate. The masked sample store still contains these additional
sites, so its row number is not an inference-site index.

1. Load both position vectors once. Compute sample rows with
   `numpy.searchsorted(sample_positions, sites_position)`.
2. Require every panel position to have an exact sample-store match. Report an
   absent position explicitly; do not intersect the arrays and silently lose
   sites.
3. Select those rows from sample calls and allele/annotation arrays. Validate
   the NPZ's panel IDs and haplotype identity/order against these stores once.
4. Under the existing biallelic contract, map sample calls to the panel's
   canonical codes: ancestral `0`, derived `1`, missing `-1`. When REF is
   ancestral, raw calls already have this polarity; otherwise swap called
   `0` and `1`. Preserve missing calls. Resolve the configured ancestral state
   by allele strings, including `{is_reference: true}`.
5. Check that the panel's allele order agrees with that polarity. Do not compare
   raw sample allele indices with canonical ancestor calls merely because both
   currently contain zeros and ones.

The current examples have ancestral REF everywhere and no missing aligned
sample calls. The recoding remains necessary for the ancestral-state modes
already supported by this repository. Do not instantiate an HMM haplotype
reader, whose cache is designed for a different access pattern.

### Eligibility and annotations

Select candidate columns from each NPZ slice, then retain only columns with
`derived_ac <= max_ac_cutoff`. The cutoff is inclusive, as in existing
evaluation. Doubletons are present naturally; sample singletons were excluded
when the panel was built.

Read `focal_ac` from the NPZ's panel-aligned counts and `focal_af` from
`variant_match_eval_derived_af` at each focal position. Check consistency with
the sample count annotation when preparing focal metadata. AF is derived count
divided by called haplotypes, as currently defined; do not infer it from an
ancestor time or divide AC by an assumed fully called sample count.

An ancestor's times are irrelevant to chunk extension. Missing/excluded focal
calls do not create seeds. A selected inferred seed must have matching derived
sample and ancestor calls at its focal site; inconsistent inputs should fail
clearly rather than producing an empty or invalid anchored interval.

## Bounds and terminal conditions

All site bounds use zero-based **inferred-panel indices** and half-open
intervals `[left_site_index, right_site_index)`. The focal site belongs to that
interval. Both genotype comparison and output coordinates use this same axis.

Derive each ancestor's supported site interval from its stored half-open
`[sample_start_position, sample_end_position)` with `searchsorted`. Respect the
containing panel `sequence_intervals` interval as well. An inferred ancestor
does not extend beyond its generated support, and a comparison must not bridge
an excluded inference-interval gap.

For a focal index `f`, let the called mismatches encountered to the right be
`r_1, r_2, ...`, in increasing order. Let the left mismatches be
`l_1, l_2, ...`, in decreasing order. Then:

```text
right[k] = r_(k+1), if that mismatch exists before the right terminal boundary
           otherwise the right terminal boundary

left[k]  = l_(k+1) + 1, if that mismatch exists before the left terminal boundary
           otherwise the left terminal boundary
```

The stopping mismatch is excluded. Previously encountered mismatches remain
inside the larger-budget intervals. All unfilled budgets receive the terminal
boundary when an ancestor finishes, rather than sentinel or missing bounds.

Treat an internal missing ancestor call or missing sample call as a terminal
barrier. It provides no evidence for a perfect match, and two missing calls
must not count as a match. On the right, a barrier at index `i` ends the
interval at `i`; on the left, it begins at `i + 1`. Missing calls outside an
ancestor's support are simply outside support. The actual datasets were
checked and have no internal ancestor missing calls.

For example, with support `[0, 10)`, focal index `4`, left mismatch indices
`3, 1`, and right mismatch indices `6, 8`:

| `max_mismatches` | Left index | Right index | Interval |
| ---: | ---: | ---: | --- |
| 0 | 4 | 6 | `[4, 6)` |
| 1 | 2 | 8 | `[2, 8)` |
| 2 | 0 | 10 | `[0, 10)` |

The middle interval has one mismatch on each side. This example should appear
in the function documentation so the meaning of the budget remains explicit.

### Base-pair coordinates

Use absolute positions from `sites_position`, not a rescaled `0..1 Mb` or
`0..10 Mb` axis. The sample contig length is 64,444,167 in both existing stores,
but the inferred comparison intervals occupy only the simulated region.
Never extend chunks to the full contig length merely because it is stored.

Use the following explicit breakpoint convention:

```text
left_position  = sites_position[left_site_index]
right_position = sites_position[right_site_index], if right_site_index < num_sites
                 otherwise the end of the containing inference interval
```

Clip both positions to the ancestor's stored BP support and its containing
inference interval. When the right site bound reaches that interval's end,
use the interval's BP end, including for an internal interval followed by a
gap. When it reaches the ancestor's supported end, clip to that ancestor's
`sample_end_position`; this can precede the next panel site's position.
The existing ancestor end positions are the final supported site's position
plus one.

This defines inference-site cells beginning at their site positions. It is a
consistent BP convention for subsequent coverage calculations, rather than a
claim that variation-free gaps locate biological recombination breakpoints.
For a left mismatch at `i`, the retained BP interval begins at the next site's
position, not at `sites_position[i]`. Store focal position separately as
`sites_position[focal_site_index]`.

## Storage-aware streaming and sweeps

### Why the loop order matters here

Both existing ancestor stores have `call_genotype.chunks = (50000, 100, 1)` and
no shards. Both datasets have fewer than 50,000 inference sites. A tiny site
slice consequently decompresses a full-height ancestor-column chunk. Repeating
such reads for every site window and every sample would hide considerable
repeated I/O behind an apparently streaming algorithm.

Use the physical ancestor-column chunks as work units. Read one physical
column block once, reuse it for the relevant sample haplotypes, and release it
when that block is complete. Within that block, perform the requested
left-to-right and right-to-left event sweeps for each haplotype. Partitioning by
ancestor columns does not change a seed's comparisons or bounds: each ancestor
belongs to exactly one block, and each seed is processed exactly once in each
direction. This adapts the storage loop order while preserving the specified
sweep algorithm.

The reader can stream forward site chunks and then read them in reverse for
the same column block if an input has several physical variant chunks. Derive
that layout from the array's chunk metadata and carry sweep state between site
blocks. With the current stores, each direction sees a single full-height
block; retain it for both sweeps, with no second Zarr read.

Do not create a rechunked ancestor store, copy the entire genotype matrix,
build a sample-by-ancestor mismatch matrix, or keep individual active ancestor
haplotypes in memory. Active state consists only of numeric seed indices and
mismatch counters into the currently loaded genotype block.

### Aligned sample data

For the two requested datasets, the full aligned sample calls occupy only
3,672,000 bytes and 8,845,000 bytes respectively. Load and polarise these once
per worker, then reuse them across ancestor blocks. This deliberately keeps
the small sample side resident while streaming the large ancestor side. The
memory cost is explicit; the plan does not claim that all sample genotypes are
streamed site by site.

Both source stores also have just one sample-axis physical chunk. There is no
benefit to repeatedly decoding that same sample chunk for individual
haplotypes. For substantially larger future sample panels, the resident sample
matrix would become the memory limit; processing physical sample-column batches
would then be a separate scaling extension. It is unnecessary for these two
datasets and should not introduce a cache framework into this implementation.

### Directional sweep kernel

Keep Zarr, pandas, strings, and process management outside Numba. The Numba
kernel receives numeric genotype blocks, seed focal indices, local ancestor
columns, support boundaries, and numeric state arrays.

1. For a haplotype and ancestor block, expand its eligible candidates into
   seeds. Assign deterministic global seed indices before scheduling work.
2. Sort activation events by focal site once. Reuse the reverse event order for
   the backward sweep; do not copy/reverse genotype matrices.
3. Initialise each seed's output bounds to its support boundaries. Maintain an
   array of active seed indices and one mismatch count per seed.
4. Sweep sites in the requested direction, activating seeds at their focal
   site. Activation before comparison is safe because the focal calls must
   match. Compare other active seeds at that site normally.
5. On a called mismatch, write the appropriate bound for the current count and
   increment it. After recording `K + 1` mismatches, remove that seed from the
   active array by replacing its slot with the last active entry.
6. On a support edge, inference-interval edge, or missing-call barrier, fill
   the remaining budgets with that terminal bound and remove the seed.
7. If there are no active seeds, jump to the next activation event. Stop when
   no events or active seeds remain. Skip column blocks with no eligible seeds.
8. Reuse the same kernel with direction `+1` or `-1`. The main directional
   differences are event order, site increment, and the left-side `+1` when
   excluding a mismatch or barrier.

When crossing physical site-block boundaries, retain active indices, event
cursor, and counters. A block edge is an I/O boundary, not a terminal boundary.
Use straightforward arrays and explicit intermediate variables. There is no
need for a class per ancestor or per active seed.

### Multiprocessing and result assembly

Use multiprocessing across independent ancestor-column blocks, distributing
tasks and returning numeric results through queues. A
`concurrent.futures.ProcessPoolExecutor` with a multiprocessing `spawn` context
provides these queues and normal worker exception propagation without custom
queue/error-handling infrastructure. Import it as
`import concurrent.futures as cf`, following repository guidance.

Request the available Snakemake core allocation for the new rule, pass its
allocated `threads` to the library, and start at most that many processes or
the number of eligible physical blocks, whichever is smaller. With `--cores
all`, use all available cores when there are enough tasks. Numba kernels should
be serial inside each process, so they do not add another parallel worker pool.

Initialise each worker once: open its own read-only stores, prepare the small
aligned sample array and shared metadata, and initialise/compile the numeric
kernel. Tasks contain physical block indices, not genotype arrays or pandas
objects. Results contain global seed indices and the two numeric bound arrays.
Represent a multi-value result with a small dataclass. Use a dataclass for
shared numeric sweep state if needed; do not introduce a generic scheduler or
an extensible class hierarchy.

Keep submissions and completed results bounded by the worker count: replace a
finished task with the next physical block. The parent fills preallocated
global bound arrays by seed index, which preserves output order independently
of worker completion order. Ensure processes are joined and failures propagate
before any CSV is written. A one-worker call follows the same block algorithm
directly, which is useful for the first verification run and avoids spawn
overhead on this small workload.

The prototype's sweeps are very short. Record actual process startup/JIT costs
and end-to-end time during implementation; the serial probe timings below are
not multiprocessing speedup claims. Avoid adding a guessed workload threshold
or another scheduling mode.

### Complexity

Let `S` be inference sites, `A` ancestor columns, `H` sample haplotypes, `C` the
physical ancestor-column width, `F` selected carried focal-site associations,
and `T` the number of comparisons performed while seeds are active.

The genotype data loaded across all ancestor blocks is at most `S * A` calls
once for each necessary directional read, shared across samples. For the
current full-height chunks it is read once total, since the same block serves
both directions. Blocks containing no eligible seeds can be skipped. The core
work is `O(T + F * (K + 1))` plus event sorting and the site advances while
there is active state. It does not scan inactive gaps or repeat work for each
frequency cutoff.

Per-worker genotype memory is `O(S * H + V * C)`, where `V` is the loaded
physical site-block height, plus metadata and active/bound arrays for that
column block. The retained global bounds require `O(F * (K + 1))`; the final
dataframe has that same row count. The full `S * A` ancestor matrix is never
resident. Increasing `K` also increases active lifetimes and output memory.

## Dataframe contract

The rule prepends dataset identity to the library result. Emit these columns in
the following order:

| Column | Meaning |
| --- | --- |
| `dataset` | Configured dataset name |
| `panel_kind` | `inferred` |
| `source` | `samples` |
| `sample_id` | Original sample-store ID, kept separate from ploidy |
| `ploidy_index` | Haplotype index within that sample |
| `ancestor_id` | Panel `sample_id` string |
| `ancestor_index` | Panel column index |
| `focal_site_index` | Focal index on the inferred-panel axis |
| `focal_position` | Absolute focal BP position |
| `focal_ac` | Observed derived count for that focal ancestor |
| `focal_af` | Observed derived frequency among called haplotypes |
| `max_mismatches` | Directional mismatch budget `k`, ranging from `0` to configured `K` |
| `left_site_index` | Inclusive retained site bound |
| `right_site_index` | Exclusive retained site bound |
| `left_position` | Inclusive absolute BP bound under the defined site-cell convention |
| `right_position` | Exclusive absolute BP bound, clipped to supported inference sequence |

Use the requested `max_mismatches` name for the dataframe budget column.
Use integer indices/counts/positions and float64 AF. Preserve existing integer
position arrays where possible; site indices and output bounds can use int64
consistently with the NPZ. Use categorical columns for repeated strings while
assembling the in-memory dataframe; CSV exports their ordinary string values.
Do not build millions of Python record dictionaries.

The unique row key is `(dataset, panel_kind, source, sample_id, ploidy_index,
ancestor_index, focal_site_index, max_mismatches)`. Order rows by NPZ haplotype
order, increasing ancestor column, increasing focal index, then increasing
budget. There are exactly `F * (K + 1)` rows after the maximum-cutoff filter.
No-candidate haplotypes contribute no rows, and an entirely empty selection
returns the defined empty schema. A chunk table cannot independently count
haplotypes with no seeds; later coverage analysis must take its denominator
from the sample/NPZ identities.

Every output row must satisfy:

```text
0 <= left_site_index <= focal_site_index < right_site_index <= num_sites
left_position <= focal_position < right_position
ancestor_start_position <= left_position < right_position <= ancestor_end_position
focal_ac <= max_ac_cutoff
```

With increasing budgets, left bounds never increase and right bounds never
decrease. For a lower AC cutoff `c`, select `focal_ac <= c`; no haplotype
comparison is required. AF can similarly support later continuous filtering.
There is no cutoff column and no duplicated output per cutoff.

## Function and workflow integration

Put the public function and its tightly scoped helpers in `lib/haplotypes.py`:

```python
construct_focal_ancestor_chunks(
    samples_path: pathlib.Path,
    ancestors_path: pathlib.Path,
    focal_ancestors_path: pathlib.Path,
    ancestral_state: dict,
    max_ac_cutoff: int,
    max_mismatches: int,
    threads: int,
) -> pd.DataFrame
```

Its responsibility is preparing validated metadata and aligned calls, building
seed indices, scheduling physical blocks, collecting bounds, and returning the
dataframe. Keep numeric sweeps and BP conversion in separate, documented
helpers. Link their docstrings to the public function and state the half-open
and directional-budget contracts explicitly. Keep one module-level logger.

Add `numba` as a direct project dependency even though it is currently installed
through tsinfer. Refresh the lockfile with `uv`; do not pin an unnecessary
version in this plan. Add one analysis configuration section to
`config.yaml.example` and the implementation-run manifest:

```yaml
haplotype_compare:
  max_mismatches: 2
```

Two is the initial experimental maximum suggested by the meeting discussion,
not a scientifically final setting. Zero is also valid. Reject negative or
non-integer maxima and nonpositive/non-integer maximum AC values. Do not add
module constants for experimental parameters, worker counts, or chunk sizes.
The physical block dimensions come from the store.

Import the new module alongside existing library modules in the Snakefile and
add the rule:

| Rule field | Value |
| --- | --- |
| Name | `construct_focal_ancestor_chunks` |
| `input.samples` | `{data_dir}/samples/{name}_samples_masked.zarr` |
| `input.ancestors` | `{data_dir}/ancestors/{name}_inferred_ancestors.zarr` |
| `input.focal` | `{data_dir}/focal_ancestors/{name}_inferred_focal_ancestors.npz` |
| `output` | `{data_dir}/dataframes/{name}_inferred_focal_ancestor_chunks.csv` |
| `params.max_ac_cutoff` | `max(config["ac_cutoff"])` |
| `params.max_mismatches` | `config["haplotype_compare"]["max_mismatches"]` |
| `params.ancestral_state` | Dataset's existing ancestral-state specification |
| `threads` | Available workflow core allocation, with actual allocated value passed to the function |
| `log` | `{progress_dir}/construct_focal_ancestor_chunks/{name}_inferred_construct_focal_ancestor_chunks.log` |

Follow the existing `run` blocks: initialise logging, call with pathlib paths,
prepend `panel_kind` and `dataset`, and write `to_csv(..., index=False)` once
the complete dataframe is available. Add the inferred chunk CSV to `rule all`
without removing existing products. Explicitly targeting just the new CSV must
produce a DAG with ancestor inference and focal lookup dependencies, and no
ancestor/sample matching or truth branch.

Use only the largest AC cutoff as the rule parameter. Editing smaller cutoffs
while keeping that maximum leaves chunks reusable. Changing the maximum or
`max_mismatches` reruns this analysis only; it does not require HMM matching or
ancestor regeneration. After changing `lib/haplotypes.py`, force this rule,
following the repository's existing convention that library files are not
Snakemake inputs.

For fixed upstream stores, HMM settings cannot affect the chunk dataframe.
There is one existing dependency caveat: `write_inference_config` combines
ancestor-inference and HMM settings in one TOML and tracks `params.hmm`. An HMM
configuration edit can therefore invalidate that TOML and trigger upstream
ancestor regeneration, followed by rebuilding dependent chunk outputs. The new
rule itself neither reads HMM parameters nor executes matching. Separating the
existing TOML's invalidation responsibilities is outside this change; do not
promise that the entire current DAG ignores HMM-only configuration edits.

Document the new target, configuration, schema, budget semantics, and filtering
in README during implementation. No changes are required in existing matching,
evaluation, or true-ancestor extraction functions.

## Read-only evidence obtained during planning

Probes used the project's existing environment with `uv run --no-sync --frozen
python`, bytecode writing disabled, read-only stores, and in-memory Numba kernels
without a disk compilation cache. Installed versions were tsinfer
`0.5.2.dev317`, Zarr `3.4.0`, and Numba `0.67.0`. These observations describe the
current local data, rather than hard-coded regression expectations for future
tsinfer versions.

| Measurement | n300, 1 Mb | n100, 10 Mb |
| --- | ---: | ---: |
| Sample individuals / haplotypes | 300 / 600 | 100 / 200 |
| Original sample sites | 9,208 | 68,151 |
| Inferred sites | 6,120 | 44,225 |
| Inferred ancestor columns | 4,639 | 26,393 |
| Distinct sample–ancestor candidate associations | 325,711 | 925,918 |
| Carried focal-site seeds | 393,585 | 1,404,159 |
| Multi-focal ancestors | 793 | 7,238 |
| Largest number of focal sites in one ancestor | 22 | 111 |
| Observed focal AC range | 2–599 | 2–199 |
| Aligned sample-call bytes | 3,672,000 | 8,845,000 |
| Largest loaded ancestor-block bytes | 612,000 | 4,422,500 |
| Output rows for budgets 0, 1, 2 at full cutoff | 1,180,755 | 4,212,477 |

The inferred intervals are `[81353, 999745)` for n300 and
`[81208, 9999801)` for n100. All panel positions aligned exactly to sample rows.
All multi-focal ancestors in both datasets had identical focal AC, AF, and
sample carrier masks. Every ancestor column in both panels was checked for
internal missing calls; none were found. An initial check of 300 columns per
dataset also confirmed no called genotypes outside stored ancestor support.

The independent oracle locates **all** mismatch indices within each selected
sample–ancestor supported span, then uses rank lookups around each focal site.
It does not use the active-sweep algorithm. This deliberately slower method is
suitable for validation, without becoming the production implementation.

Results:

1. The two-direction Numba prototype matched every oracle bound for all 600
   n300 haplotypes, all 325,711 sample–ancestor pairs, all 393,585 seeds, and all
   budgets `0..2`. It also checked anchored and nested site intervals. Its
   full serial probe took about 2.62 seconds: 0.14 seconds of ancestor-block
   reading, 0.50 seconds of sweeps including initial JIT compilation, and 1.80
   seconds of oracle comparison; the remainder was setup/assembly within the
   measured loop.
2. An initial n100 probe sampled 20 evenly spaced haplotypes, compared 92,788
   sample–ancestor pairs and 140,614 seeds, and matched all oracle bounds for
   budgets `0..2`. It read all physical ancestor blocks once and checked every
   column for internal missing calls.
3. A subsequent full n100 probe compared all 200 haplotypes, all 1,404,159
   seeds, and all 4,212,477 budget rows against the oracle. It checked BP
   clipping and focal containment, then constructed the complete proposed
   dataframe in memory. The comparison/oracle loop took about 6.66 seconds.
   No CSV was written. The dataframe's measured deep memory usage was
   381,626,051 bytes, about 364 MiB, using categorical ID columns and the
   prototype's integer widths. Its two int64 bound matrices occupied
   67,399,632 bytes. Total process peak memory was not measured and is larger
   than the dataframe alone; multiprocessing startup, CSV serialization, and
   production dtype choices were not benchmarked.
4. Small in-memory checks exercised mismatch exclusion and internal missing
   barriers: for a focal at index 2, a left mismatch at 0, a right mismatch at
   3, and a right missing call at 5, the budget-`0..2` bounds were left
   `[1, 0, 0]` and right `[3, 5, 5]`. A separate biallelic polarity check
   confirmed reversed ancestral/REF order and preservation of missing calls.

For the complete n100 in-memory result, smaller AC thresholds selected these
row counts without any further comparison:

| Inclusive focal AC cutoff | Rows across budgets 0, 1, 2 |
| ---: | ---: |
| 2 | 49,536 |
| 3 | 91,359 |
| 5 | 165,027 |
| 10 | 292,290 |
| 20 | 507,120 |
| 100 | 2,108,724 |
| 200 | 4,212,477 |

The result size is practical for the requested initial `K = 2`. At `K = 20`,
the same full-cutoff seed counts imply 8,265,285 n300 rows and 29,487,339 n100
rows; output memory will grow accordingly. Measure larger settings when they
are actually requested. Keep this initial implementation as the requested
single dataframe rather than adding an unrequested partitioned output format.

## Implementation and verification sequence

Do not add a test suite for this change. The prompt requests dataset testing,
which can use direct read-only/manual probes and integration runs. If a formal
suite is separately requested later, organise it in pytest classes with
fixtures as specified by AGENTS.md.

1. Add the direct Numba dependency and configuration setting. Implement metadata
   alignment, focal-seed expansion, and exact coordinate contracts in
   `lib/haplotypes.py`.
2. Implement the reusable directional kernel and block reader. Start with the
   direct one-worker path. Verify the worked boundary example, `K = 0`, exactly
   `K` versus `K + 1` mismatches, no mismatches, support ending before a budget
   is exhausted, internal/sample missing barriers, and empty candidate sets.
3. Exercise multiple focal sites on one ancestor, two seeds activating at the
   same site, first/last supported focal sites, a final partial ancestor-column
   block, and two inference intervals separated by a gap. Repeat a small
   in-memory case over several physical site-block layouts, checking that
   active state survives block boundaries in both directions. Check reversed
   sample allele order and both existing ancestral-state modes.
4. Compare the production one-worker output on the existing n300 1 Mb panel
   with the independent mismatch-rank oracle for every haplotype/seed/budget.
   Initially use full cutoff 600 so the measured counts above apply. Also
   check a smaller maximum cutoff: it must equal filtering the full-cutoff
   result, including every bound and annotation.
5. Add the bounded process/queue scheduling. Compare one-worker and allocated
   multi-worker results exactly in canonical order. Measure end-to-end
   runtime and peak memory, including process startup and output assembly.
   Verify a worker failure reaches the caller and prevents a partial CSV.
6. Add the new rule, default target, and README documentation. Run
   `uv run ruff check lib` and `uv run ruff format --check lib`. Dry-run the new
   explicit target and confirm there are no `.trees`, match JSONL, or true
   products in its dependency chain.
7. Create an implementation-run manifest selecting the existing n300 input,
   `ac_cutoff: [3, 5, 10, 100, 600]`, and the new comparison configuration.
   Use an isolated data/progress directory for any inputs that require
   rebuilding; do not overwrite the upstream simulation VCZ. Alternatively,
   point at the already available n300 products with a manifest using
   `data_dir: data/match_eval`. Finalise the n300 dataframe first and reread
   the CSV to verify schema, identity keys, annotations, counts, interval
   containment, and budget nesting.
8. Once n300 passes, target the n100 10 Mb dataframe using its existing inferred
   products and maximum cutoff 200. Compare the complete production output
   against the oracle, as in the read-only planning probe. Expect 4,212,477
   rows for the current data at `K = 2`. Measure CSV file size and actual peak
   memory as part of this final run.
9. Check invalidation semantics: changing smaller cutoffs while retaining the
   maximum must leave this target reusable; changing the maximum or `K` must
   select only this analysis branch for rerun. Compare shared rows after
   increasing `K`, and compare a decreased maximum against filtered existing
   rows. Confirm chunk values are independent of HMM settings and match files
   for fixed upstream stores, while accounting for the existing combined-TOML
   dependency caveat described above.

The n100 manifest currently lives in ignored `config.yaml` and already points
at `data/match_eval`. The tracked example manifest describes n300. Add the new
configuration section during implementation before using either; do not rely
on a Python fallback constant.

After the new rule exists, representative implementation commands are:

```sh
# A separate n300 manifest includes the new setting and the full cutoff 600.
uv run snakemake --cores all --configfile /private/tmp/haplotype_compare_n300.yaml --dry-run data/match_eval/dataframes/out_of_africa_n300_1mbp_inferred_focal_ancestor_chunks.csv
uv run snakemake --cores all --configfile /private/tmp/haplotype_compare_n300.yaml data/match_eval/dataframes/out_of_africa_n300_1mbp_inferred_focal_ancestor_chunks.csv

# The existing n100 manifest uses full cutoff 200, after adding the new setting.
uv run snakemake --cores all --dry-run data/match_eval/dataframes/out_of_africa_n100_10mbp_inferred_focal_ancestor_chunks.csv
uv run snakemake --cores all data/match_eval/dataframes/out_of_africa_n100_10mbp_inferred_focal_ancestor_chunks.csv
```

Completion of the future implementation means both actual CSVs exist, their
production bounds agree with independent direct comparisons, multi-focal and
missing-call semantics are checked, multiprocessing preserves deterministic
results, and no TS or HMM participates in constructing the chunks.
