# Matching setup implementation plan

## Scope and source review

Extend [initial_mvp.md](initial_mvp.md) with ancestor matching, per-haplotype
focal-ancestor lookup, and sample matching. Produce independent results for
inferred and true ancestor panels; datasets with `ts_path: null` have only the
inferred branch. Matching uses the full panel and does not consume focal-ancestor
results. Candidate restriction, copying-overlap metrics, ARG comparisons,
post-processing, and real-dataset validation remain out of scope. No unit tests.

This plan follows `AGENTS.md`, the existing Snakefile and library modules, the
11:57 “Design matching evaluation pipeline” thread in
`~/exo/this_week.org`, and the discussion beginning at “General pruning/indexing
idea” in `~/exo/documents/meeting_notes/2026-09-10 (Meeting with Gil).md`.
The focal lookup implements the proposed allele index; it does not test whether
a candidate provides a good copying path.

The local tsinfer commit reviewed is
`a7043084c74ed491bf3be8cd8670638a21285f4e`, “Support matching against ancestors_ts”.
Relevant implementation references are
[pipeline.py](../../tsinfer/tsinfer/pipeline.py),
[config.py](../../tsinfer/tsinfer/config.py),
[matching.py](../../tsinfer/tsinfer/matching.py), and
[the split-matching tests](../../tsinfer/tests/test_pipeline.py).
The new reference handoff supersedes the journal's earlier checkpoint workaround.

A read-only split-matching probe on the existing zero-error inferred panel
completed: the ancestor TS had 4,641 nodes and 6,120 sites; the raw sample TS
had 5,241 nodes and all 600 sample haplotypes. Both retained sequence length
64,444,167. This checks the local API handoff, not the rules still to implement.

Before adding matching, replace the mutation-level true panel with a site-level
dataframe. Retain one selected derived mutation per inference site and
discard the other mutation records at recurrent sites. Reuse the existing
selection of the mutation covering the most present-day derived carriers.
The dataframe retains the original mutation count for identifying recurrent
sites, but no longer represents alternative or back-mutation ancestors.
The approved matching policy retains zero-time selections as flagged CSV rows
and writes one panel column per unique selected positive-time truth node.
The detailed extraction contract below supersedes the mutation-level contract
in `initial_mvp.md`.

## Configuration and output paths

Keep the current dataset manifest, singleton annotation, inference settings,
and ancestor output paths. Add this setting to `config.yaml.example` and the
executor's local configuration:

```yaml
matching_threads: 1
```

Use `matching_threads` in both matching rules' `threads` declarations and pass
Snakemake's allocated `threads` to tsinfer. Run with `--cores all` so independent
panel jobs can execute in parallel. Use Snakemake's job scheduling and tsinfer's
existing worker implementation; do not add another executor around these calls.
Leave HMM recombination and mismatch fields unset to use tsinfer defaults.
Explicitly write `path_compression = false` in the native match configuration.
Do not add post-process or augment-sites sections.

For `{kind}` equal to `inferred` or `true`, use:

```text
{data_dir}/configs/{name}_ancestor_inference.toml       # inferred, existing
{data_dir}/configs/{name}_true_ancestors.toml           # true only
{data_dir}/ancestors/{name}_{kind}_ancestors.zarr/      # existing panel
{data_dir}/ancestors/{name}_{kind}_ancestors.trees
{data_dir}/focal_ancestors/{name}_{kind}_focal_ancestors.npz
{data_dir}/matches/{name}_{kind}_samples_raw.trees
{data_dir}/matches/{name}_{kind}_samples_matches.jsonl
{progress_dir}/{rule}/{name}_{kind}_{rule}.log
```

The existing inferred TOML's match output is already its ancestor Zarr path
with the suffix changed to `.trees`. Preserve that convention. Neither stage
uses a checkpoint work directory, which avoids stale jobs and accidental
resume. Each sample-matching invocation writes its own JSONL output from scratch.

## Step contracts

Keep the original four rules, updating `extract_true_ancestors` as described
below. Add the following rules, with direct function
calls inside `run` blocks, following the existing workflow style:

| Rule | Inputs | Outputs | Function and location |
| --- | --- | --- | --- |
| `write_true_ancestors_config` | Masked sample Zarr; true ancestor Zarr | `{name}_true_ancestors.toml` | Reuse `utils.write_inference_config` in `lib/utils.py` |
| `match_ancestors` | Panel Zarr; masked sample Zarr; inferred or true TOML | `{name}_{kind}_ancestors.trees` | `matching.match_ancestors(config_path, output_path, threads)` in `lib/matching.py` |
| `find_focal_ancestors` | Masked sample Zarr; panel Zarr; true CSV for the true branch only | `{name}_{kind}_focal_ancestors.npz` | `matching.find_focal_ancestors(samples_path, ancestors_path, output_path, ancestral_state, true_dataframe_path=None)` in `lib/matching.py` |
| `match_samples` | Ancestor TS; masked sample Zarr; panel TOML | `{name}_{kind}_samples_raw.trees`; `{name}_{kind}_samples_matches.jsonl` | `matching.match_samples(config_path, ancestors_ts_path, output_path, match_file_path, threads)` in `lib/matching.py` |

Pass the dataset's `ancestral_state` mapping to the focal function. Declare
stores as `directory(...)` only where they are outputs. The panel and sample
stores are explicit matching-rule inputs even though their paths also appear
inside TOML: this makes the data dependencies visible to Snakemake.

Use `{name}` and `{kind}` wildcards, restricting `kind` to `inferred|true`.
Use a small Snakefile input function to select the existing inferred TOML or
the true TOML. For the focal rule, a named dataframe input is an empty list for
inferred panels and a one-element list containing the existing true CSV for
true panels. Pass `None` through to the function for the inferred branch.
Do not derive dataset names by splitting filenames on underscores.

Build the requested panel collection from every dataset's inferred panel plus
the true panels belonging to `simulated_names`. This is a list of dataset/kind
records, not a new panel registry or parameter grid. `rule all` collects:

- Existing inferred ancestor Zarrs and optional true Zarrs and true CSVs.
- Ancestor TSs for every requested panel.
- Focal-ancestor NPZ files for every requested panel.
- Raw sample TSs and sample match JSONL files for every requested panel.

There must be no true targets for a dataset without truth. Do not generate a
cross product of all dataset names with both kinds.

## One selected mutation per site and one panel ancestor per older node

Update `ancestors.extract_true_ancestors` in `lib/ancestors.py` before building
the matching rules. Keep its existing inputs and output paths. The inferred
panel continues to define the shared inference-site axis; recurrent truth
sites retain their position on that axis, but contribute only one selected
event instead of one record per original mutation.

For each inference site:

1. Fetch the corresponding truth site and record `num_mutations` from the
   original truth TS.
2. If there is one mutation, select `site.mutations[0]` directly. Do not loop
   over mutations or decode present-day carriers for this ordinary case.
3. If the site is recurrent, call the existing `primary_mutation` helper.
   Decode present-day derived carriers, track them in the tree, and select the
   derived-state mutation whose node covers the most carriers. Preserve its
   existing deterministic tie-breaking by mutation order. Only this case
   needs a loop through the site's mutations.
4. Append one record for the selected mutation's node and move to the next
   site. Do not append records or retain details for unselected mutations.
   If the helper finds no derived carriers, fail with the site identified:
   there is no valid selected ancestor for the one-row-per-site contract.

The true CSV has exactly one row per inference site, with these columns:

```text
inference_site_id, true_site_id, focal_position, inferred_ancestor_id,
true_ancestor_id, true_mutation_id, true_mutation_is_singleton, true_node_id, true_node_time,
inferred_node_time, derived_af, num_mutations
```

Drop `is_primary_for_site` and `focal_allele`. Every retained row is the selected
derived event for its site. Keep `true_mutation_id` for that selected event,
and retain the original `num_mutations`, including values greater than one.
Sort by decreasing `true_node_time`, then `inference_site_id`.
Set `true_mutation_is_singleton = (true_node_time == 0)`. These terminal
selections retain their CSV rows and original truth IDs but have a blank
`true_ancestor_id` and no panel column. The flag applies to a selected origin;
it can be true at a recurrent site with more than one present-day derived carrier.
`num_mutations` distinguishes recurrent origins from ordinary truth singletons.

For the other rows, assign one panel `sample_id` per unique `true_node_id`, in
order of first occurrence in the sorted records. Rows selecting the same node
share `true_ancestor_id` and contribute multiple focal positions to that column.

Reuse the existing chunked decoding and native ancestor-Zarr writer. Each
eligible node supplies one haplotype column, all associated focal positions,
and its span. Decode the selected node's final genotype on the shared site axis;
do not rewrite its alleles to manufacture a nonrecurrent haplotype. Preserve
missing calls.

For now, inferred ancestors use their inferred `sample_time` values and true
ancestors use their original `true_node_time` values as `sample_time`. Samples
remain at time zero. Do not introduce an offset, retime nodes, or substitute
frequency-derived ages for true ancestors. Check the selected true panel's
time ordering during example validation. The zero-time selections are recorded
in the CSV and left to sample-stage mutation placement; their sites are never
filtered from the shared inference axis.

Selecting one event per site removes alternative mutation-origin records,
but is distinct from deleting recurrent sites or changing decoded haplotypes.
Confirm in the manual matching checks that the resulting reference satisfies
the current matcher's one-mutation-per-site restriction. If it does not,
report the offending sites; do not assume the dataframe row count proves that
the matched TS is nonrecurrent. Recurrence is allowed in the final raw sample TS,
which is not used as a reference for another matching group.

## Native TOML and the ancestor/sample boundary

Update `utils.write_inference_config` to explicitly disable path compression.
The optional `write_true_ancestors_config` rule calls the same writer with the
true panel path. It only writes configuration; never run ancestor inference
using that TOML, since that would overwrite the extracted true panel.

Both TOMLs contain the same masked sample source, including
`exclude = "INFO/match_eval_singleton_mask == 1"`, the same ancestral-state
annotation, and one `[[ancestors]]` section pointing to the selected panel.
Keep the names `ancestors` and `samples` for both branches. The TOML parser
automatically creates the ancestor source with `sample_time = "sample_time"`;
do not also declare an explicit source with that name.

Both sources must remain in `[match.sources]` during ancestor matching:

```toml
[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.samples]
node_flags = 1
create_individuals = true
```

Including samples causes `pipeline.match` to obtain the complete contig length
from the sample VCZ before it processes any matching groups. An ancestor-only
configuration instead falls back to the final sequence-interval endpoint.
For the stored n300 example those values are 64,444,167 and 999,745 respectively;
the sample VCZ's declared contig length is authoritative for this workflow,
even though the simulation covers approximately one megabase.

In `lib/matching.py`, `match_ancestors` does the following:

1. Load the TOML with `tsinfer.config.Config.from_toml`.
2. Obtain the job list from `tsinfer.pipeline.compute_groups_json` and decode
   it with `json.loads`.
3. Find the minimum group among jobs whose source is `samples`. Require every
   ancestor job to have a smaller group. Raise a clear `ValueError` if the
   sources cannot be separated. This enforces the required ordering of
   ancestors before contemporary samples without changing their times.
4. Call `tsinfer.pipeline.match(cfg, group_stop=first_sample_group,
   num_threads=threads)`. `group_stop` is exclusive. The routine's warning that
   a partial run without a workdir cannot be resumed is harmless here: the next
   step uses the saved reference TS, not checkpoint resume.
5. Dump the returned TS directly to the rule's output. Keep its root nodes,
   sites, node metadata, and pre-created sample individual rows intact.

## Focal lookup and storage

Treat a sample here as a haplotype, identified by `(sample_id, ploidy_index)`.
For diploid input, the two chromosomes receive separate candidate sets. Take
the panel's `variant_position` as the matching site axis. A raw input genotype
code of 1 is not necessarily derived: polarise using the configured ancestral
state before selecting derived calls. Missing calls contribute no candidates.

Use small dataframes only for the focal relation:

- For inferred panels, `_inferred_focal_dataframe(ancestors_path)` in
  `lib/matching.py` builds an in-memory dataframe with one row per non-padding
  entry of `sample_focal_positions`. Its columns are `focal_position` and
  `inferred_ancestor_id` (the actual panel `sample_id`).
  Padding is the native writer's negative sentinel. A multi-focal ancestor
  contributes multiple rows. No separate inferred CSV or new workflow rule is
  necessary.
- For true panels, read the site-level true CSV, with ancestor IDs as strings,
  and skip rows with `true_mutation_is_singleton=True`. Use `true_ancestor_id`
  to locate the selected panel column for each eligible focal position; multiple
  sites can share that column. Every eligible row is a derived focal association;
  no allele or primary-event filtering remains. These are focal associations, not claims
  about the sample's actual mutation origin at a recurrent site.

Build a position-to-ancestor-column lookup once. For each sample haplotype,
find its derived positions on the panel axis, collect the lookup entries,
and use `numpy.unique` to produce sorted, distinct ancestor indices. Excluded
sample sites and sites with no focal record contribute nothing. Multiple
carried focal sites belonging to one ancestor still produce one candidate.

For this MVP, read the sample genotype matrix on the panel axis and polarise
it once, matching the initial pipeline's in-memory approach. Use
`numpy.searchsorted` on the ordered sample positions to select the panel rows.
Read allele strings and ancestral-state values at the same rows. Keep missing
calls missing, and identify derived calls as nonmissing calls whose allele
differs from the configured ancestral allele. Do not compare raw codes from
sample and ancestor stores, whose allele orders can differ.

Write one `numpy.savez_compressed` file per panel, using ordinary numeric and
Unicode arrays that load with `allow_pickle=False`:

| Array | Shape/type | Meaning |
| --- | --- | --- |
| `sample_id` | `(h,)`, Unicode | Sample ID for each haplotype row |
| `ploidy_index` | `(h,)`, integer | Chromosome index for that row |
| `ancestor_id` | `(a,)`, Unicode | All panel `sample_id` values in Zarr column order |
| `offsets` | `(h + 1,)`, int64 | Boundaries of each row's candidate slice |
| `ancestor_index` | `(k,)`, int64 | Concatenated candidate column indices |

Rows follow sample-store order, then increasing ploidy index. Row `i` uses
`ancestor_index[offsets[i]:offsets[i + 1]]`. Equal offsets represent an empty
set. Start offsets at zero and finish at `k`. This compact ragged representation
avoids pickled object arrays, a dense sample-by-ancestor matrix, and one file
per haplotype. IDs are store-scoped; these indices are never TS node IDs.

## Sample matching and match-file contents

`matching.match_samples` loads the panel TOML, then constructs a sample-only
`tsinfer.config.Config` with:

- Only the original `samples` source, preserving its name, filters, selection,
  order, time policy, and ancestral-state annotation.
- `ancestors=[]` and only the original sample `MatchSourceConfig`.
- `reference_ts` pointing to the saved raw ancestor TS, the raw sample output
  path, and `path_compression=False` in `MatchConfig`.
- The original individual metadata settings, if present, and no post-process
  or augment-sites configuration.

Use `dataclasses.replace` for these small changes rather than mutating the
ancestor-stage configuration. Call
`tsinfer.pipeline.match(sample_cfg, num_threads=threads,
match_file=match_file_path)` and dump its returned TS. Do not call
`pipeline.run`, which performs additional inference stages. Do not simplify,
compress, augment, or otherwise transform either TS.

The reference supplies positions, ancestral alleles, and sequence length.
The sample-only call recreates job individual IDs in the same order as the
ancestor stage's pre-created individuals. Preserving the source name and
sample ordering is essential to that correspondence.

Retain tsinfer's JSONL directly, without a conversion or aggregation rule.
It already provides one file per panel and one record per matched haplotype,
including all current fields:

```text
group, haplotype_index, source, sample_id, ploidy_index, time,
path: [{left, right, parent}, ...],
mutations: [{position, derived_state}, ...]
```

Coordinates are absolute genomic positions and path intervals are half-open.
Mutation derived states are canonical allele indices, whereas TS mutation
states are allele strings. JSONL completion order can vary with threading;
join records by `(source, sample_id, ploidy_index)`, not line number. The
sample-only invocation has its own haplotype indices and group numbering.
Parent IDs refer to the preserved reference node IDs in the raw sample TS.
Join them to panel IDs using node metadata `source`, `sample_id`, and
`ploidy_index`, never by subtracting a root count.

Neither the sample-matching rule nor its function takes the focal NPZ as an
input. The two products meet only in a later analysis step.

## Implementation order and manual checks

1. Revise `extract_true_ancestors` to select one mutation per site, update its
   CSV schema and docstrings, and preserve original node times. Then add
   `lib/matching.py` with the four functions described above, module-level
   imports, a single module logger, and clear docstrings linking the storage
   and ID contracts. Keep miscellaneous configuration and logging helpers in
   `lib/utils.py`. Add its currently missing `import tsinfer`, which is needed
   by `setup_log`.
2. Add the matching thread setting, explicit native path
   compression setting, and the four rules. Extend `rule all` using the panel
   collection. Log each rule through `utils.setup_log`.
3. Update README output paths and explain the split, site-level true dataframe,
   original node times, focal-file layout, and native match records. Update
   `initial_mvp.md` to replace its superseded mutation-level extraction contract
   and schema.
   Add no tests, analysis rules, or additional dependencies.
4. Check syntax/style with `uv run ruff check lib` and
   `uv run ruff format --check lib`. Dry-run the DAG with
   `uv run snakemake --cores all --dry-run`.
5. Use a temporary YAML manifest with output and progress directories separate
   from the existing MVP outputs. Include the same three n300 inputs used in
   `initial_mvp.md`: zero error, genotype error, and the zero-error input with
   `ts_path: null`. Run using
   `uv run snakemake --cores all --configfile <example_manifest.yaml>`.
   Do not validate real datasets.
6. Check the regenerated true CSVs and Zarrs: exactly one row per inference
   site and one panel column per unique selected positive-time node, one selected
   derived event per row, original mutation counts retained, and the
   `true_mutation_is_singleton` flag. Flagged rows have blank ancestor IDs and no
   panel association; all their site positions remain. Verify that ordinary sites
   use their sole mutation and recurrent sites use the existing carrier-count
   selection. Check `sample_time` against the CSV's original `true_node_time`
   and confirm that the selected ancestors precede sample groups without
   retiming. Check the focal files: every diploid
   dataset has 600 haplotype rows; offsets delimit valid sorted, distinct panel
   indices; empty candidate sets remain present. For a few haplotypes, compare
   candidate sets directly with derived calls and dataframe focal rows, covering
   multi-focal inferred ancestors and the single selected true ancestor at
   recurrent sites, excluding flagged singleton origins. Confirm the no-truth
   branch requests no true outputs. On these examples the true CSVs have 3 and
   54 flagged rows, and the true panels have 3,535 and 3,494 columns respectively.
7. For branches that match successfully, verify that ancestor TSs have no
   `samples` source nodes, ancestor node flags are zero, and raw sample TSs
   contain all 600 sample haplotypes. Both TSs must have the sample VCZ's full
   sequence length and the panel's site positions. Match records must cover
   each sample/haplotype key exactly once, and their copying segments and
   mutations must agree with the corresponding raw TS nodes. Preserve the
   pre-created individual rows and ancestor metadata.
8. On the zero-error inferred example, compare the split result with a single
   uninterrupted `pipeline.match` invocation using the same panel, sample
   source, and settings. Compare tables ignoring provenance. This is a manual
   integration check, not a new test suite.
9. Report actual results for both simulated branches. Check that the true
   panels match successfully with original times and their references contain
   at most one mutation per site. Verify sample-stage mutation placement at
   flagged positions: 6 mutations at 3 recurrent sites for zero error and
   110 mutations at 54 recurrent sites for genotype error. Recurrence in the
   final sample TS is valid. No additional site filtering is applied.

Because library code is intentionally not a Snakemake input, force affected
rules after edits. In particular, regenerate TOMLs after changing their writer
and force `extract_true_ancestors` after changing its selection and schema;
old true stores can contain duplicated nodes and zero-time columns, while old
CSVs lack the singleton-origin flag and shared ancestor IDs.
Subsequent ancestor TS,
raw sample TS, and match JSONL products must be rebuilt together when their
matching settings change.
