# Stitched matching from focal coverage masks

The whole-genome reduced pass and process orchestration below are superseded by
[fast_stitched_matching.md](fast_stitched_matching.md). The pipeline now matches
focal and gap components independently and shares inputs across threads.

## Approach

For each haplotype and AC cutoff, simplify the ancestor reference to its focal
ancestors and run one reduced-reference HMM over all inference sites. For each
requested mismatch budget k, retain that match in focal-covered regions and
overwrite gaps with independent full-reference HMM matches. Assemble a raw,
unsimplified sample TS alongside the existing full-HMM baseline.

The reduced pass uses gap observations too, so focal pieces share whole-genome
context. It is reused across all k values. Only gaps create region HMM calls.
This avoids repeated focal traversals with the current engine; full-reference
gap traversal costs remain.

Implement in `lib/haplotypes.py`, `lib/matching.py`, the Snakefile and current
documentation/configuration. Start with `method="hmm"`. Greedy matching,
boundary conditioning, engine optimisation and analysis are later work.
No tsinfer changes or dependency audits are part of this plan.

## Configuration and outputs

Replace the nested scalar setting with these top-level example values:

```yaml
ac_cutoff: [3, 5, 10]
max_mismatches: [0, 1]
focal_methods: [hmm]
```

Remove `haplotype_compare`. Validate the numeric lists once in the Snakefile,
following the existing AC validation: nonempty, strictly increasing integers,
excluding booleans; AC must be positive and k nonnegative. Initially accept only
`hmm` in the method list. Pass only each list's maximum to
`make_ancestor_intervals`; retain its current archive schema and columns 0
through the maximum k. The budget is per direction from the focal seed and
defines coverage, not an HMM mismatch limit.

Under `data_dir`, produce:

```text
focal_masks/{name}_inferred_ac{ac}_k{k}_focal_mask.npz
matches/{name}_inferred_ac{ac}_k{k}_{method}_stitched_samples_raw.trees
matches/{name}_inferred_ac{ac}_k{k}_{method}_stitched_samples_matches.jsonl
```

Expand masks over every AC/k combination. Expand stitched outputs over AC/k
and method, and include them in `rule all` with the existing baseline products.
Update the README, example config and local config for the new settings.

## 1. make_focal_mask

Add `haplotypes.make_focal_mask(intervals_path, output_path, ac_cutoff,
max_mismatches)` and its Snakemake rule. The interval NPZ and selected scalar
AC/k are its only inputs.

For each haplotype's association slice, select `focal_ac <= ac_cutoff` and
column k of the bounds. Sort by left endpoint, merge overlapping or touching
intervals and emit the covered union plus its complement as ordered runs.
`is_gap=True` means zero coverage; `False` means covered. Runs partition the
inference-site axis. Preserve the roster, including haplotypes with no eligible
ancestors, which have one all-gap run.

Save with `np.savez_compressed`, using numeric/Unicode arrays:

| Fields | Meaning |
| --- | --- |
| `sample_id`, `ploidy_index` | Original haplotype roster |
| `offsets` | Haplotype boundaries in the flattened run arrays |
| `left_site_index`, `right_site_index` | Half-open run bounds |
| `is_gap` | Boolean value of each run |
| `num_sites` | Inference-site axis length |
| `ac_cutoff`, `max_mismatches`, `focal_choice` | Selected parameters and seed provenance |

Trust the interval writer's bounds and roster. The DAG supplies budgets/cutoffs
within the generated maxima; do not repeat their validation in this function.

## 2. simplify_ancestors_ts

Build a dictionary once per worker joining focal NPZ `ancestor_id` values to
original reference nodes through metadata `(source, sample_id, ploidy_index)`.
Use source `ancestors` and ploidy 0. Candidate indices are panel columns, not TS
node IDs; resolve them through this dictionary.

For the current haplotype, select its candidate columns through the NPZ offsets
and retain `derived_ac <= ac_cutoff`. With
`retained_nodes = [0, 1, *focal_nodes]`, simplify in one call:

```python
reduced_ts, old_to_new = ancestors_ts.simplify(retained_nodes, map_nodes=True, filter_sites=False)
```

Nodes 0 and 1 are the raw reference's ultimate and virtual roots. Their order
must be retained. Invert the non-null entries of `old_to_new` to map every
reduced copying parent back to its original node ID. Return the reduced TS and
inverse map in a dataclass. Use the guarantees of `simplify` rather than checking
roots, sites and maps again.

Retained internal ancestors remain HMM copying states. Coverage comes solely
from the saved mask, with no parent-membership or allele-agreement acceptance
filter. Keep the reduced TS transient and reuse it across k values. Skip the
reduced pass when the haplotype has no focal coverage.

## 3. Coarse scheduling and input reuse

Add one `match_stitched_samples` Snakemake job per dataset/AC/method, producing
all k outputs. Its inputs are the original ancestors TS, sample VCZ, inference
TOML, focal NPZ and the masks for that cutoff. Use output expansion over k with
name/AC/method wildcards. Reserve `workflow.cores` through `threads` and pass the
granted allocation to the runner. Simplification and region matching are Python
helpers, without per-haplotype/per-region rules, files or job manifests.

The coordinator:

1. Reuses the sample-only configuration from `match_samples`; factor its
   construction into a shared helper. Call
   `tsinfer.pipeline._build_jobs_with_metadata` for native `MatchJob` objects.
   These already supply the sample identity and individual/population IDs.
2. Uses `tsinfer.vcz.HaplotypeReader` with the sample source, reference positions
   and ancestral alleles, and a schedule of one read per haplotype. Reuse its
   allele mapper/counts and existing cache setting. Close it in `finally`.
3. Streams one task per observed haplotype to `multiprocessing.Pool`, with
   `chunksize=1` and workers limited by the allocation and number of haplotypes.
   Each task processes all k values. Use the same worker function directly
   when only one worker is requested.

Each worker initializer loads the reference/masks/focal NPZ and creates its
full-reference indexes and matcher once. Each haplotype task builds its reduced
reference/matcher once, performs one whole-genome reduced match and processes
its gaps sequentially. Return native results per k; order them by native
`haplotype_index` in the coordinator.

Trust products from the same DAG and reuse the native builders/readers. Do not
add roster comparisons, individual/population-table audits, genotype validators
or bespoke empty-dataset handling. Each worker owns its mutable matcher state.

## 4. Match and stitch using per-haplotype arrays

Use one shared helper around
`AncestorMatcher.find_path(haplotype, start, end, match_out)` for reduced and gap
matches. Pass the full observed array and absolute site indices. Construct
indexes with `vestigial_root=False`, native allele counts and the configured
sample HMM probabilities. Leave native `weight_by_n` behaviour unchanged.

The helper converts returned genomic path bounds to site parents and copies
the requested slice of `match_out` before reuse. Return the two arrays in a
small dataclass. Materialise them before another matcher call.

For the reduced pass, use `[0, num_sites)` and remap its site parents to original
node IDs. Keep its parent/copied-allele arrays in memory for this haplotype.
For each k, copy those arrays and overwrite the gap slices with full-reference
results. If no reduced pass was needed, allocate arrays and fill the all-gap
domain directly. No segment-clipping algorithm, region-result classes or
separate focal-extraction function is needed. Only final native results leave
the worker; these arrays are transient, not cohort-sized storage.

Use the baseline's first-called to last-called range as the stitched output
domain, or the entire axis for an all-missing haplotype. Clip mask runs to that
domain and match only True runs. Split gap calls at inference-interval boundaries
from reference metadata; focal matching remains one whole-genome pass. Missing
calls are neutral observations. Gap calls have no method parameter and start
independently, without padding or endpoint conditioning.

Keep a `method` parameter on the focal/haplotype matching entry point, initially
`hmm`. A later greedy method can populate the focal arrays from intervals while
using the same gap loop, output patterns and assembly. Do not build its dispatch
or method-specific caching now.

## 5. merge_paths and native assembly

For each completed haplotype/k, encode changes in the finished site-parent array
as native `PathSegment` objects. This merges adjacent equal parents automatically,
including across seams. Use site positions for internal boundaries, BP 0 when
the domain starts at site 0, and `sequence_length` when it ends at the final site;
otherwise its outer bounds are `positions[start]` and `positions[end]`.

Derive native mutations once from the finished copied alleles within that domain:
called observations differing from copied alleles produce a `Mutation` with the
observed canonical allele code. This automatically excludes discarded reduced
gap mismatches and mutations at missing calls. Return a native `MatchResult`.

For each k/method output, use one `tsinfer.matching.extend_ts` call with the
original ancestor reference, `(job, result)` pairs and the reader's allele mapper.
It supplies sample nodes, individual/population assignments, edges and mutation
parents. Dump its raw, unsimplified TS and write native-shaped JSONL in haplotype
order, adding AC/k/method to the native record fields. The original reference
node IDs are preserved and each haplotype adds one node. Reuse native structures
and assembly rather than reproducing their table-building or checks.

## Implementation and focused verification

Implement configuration/masks first, then the shared matching helpers and
haplotype runner, then the coarse rule and native assembly. Use `uv run` for
tooling. Do not add tests or run a full cohort automatically.

Use a small smoke run with temporary outputs to verify only the new behaviour:

- The example produces six masks, and interval generation uses maxima 10 and 1.
  One mask agrees with direct interval-union coverage, including an all-gap row.
- A single-interval all-gap haplotype reproduces the native full-HMM result.
  For focal coverage, only one reduced call is made across k values and parents
  agree wherever their focal masks overlap.
- The saved TS/JSONL describe the same stitched paths and reproduce called sample
  alleles, including genuine seam switches and mutations in retained focal pieces.
- A dry run shows mask jobs per AC/k and stitched jobs per dataset/AC/method,
  without per-region jobs or a dependency on the baseline sample matches.

Report those results and any limitations. Do not add identity-audit frameworks,
performance benchmarks, comparison CSVs or further configuration switches.
