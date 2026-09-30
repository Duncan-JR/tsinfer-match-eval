# Initial MVP: inferred and true ancestor panels

## Scope

Read existing single-contig, phased sample VCZs. Produce an inferred ancestor
VCZ for every dataset and, where `ts_path` is supplied, a true ancestor VCZ and
site-level CSV. Reuse `../tsinfer-anc-eval` inputs and the editable local
`../tsinfer` code. Matching is extended in [matching_setup.md](matching_setup.md);
ARG comparisons remain out of scope. Assume correct polarisation and binary,
biallelic simulations. Do not add a unit-test suite.

The initial implementation has two small modules: `lib/utils.py` for the copied
sample store and TOML; `lib/ancestors.py` for inference and truth extraction.
The Snakefile calls those functions in `run` blocks and uses explicit rules and
output names, following `tsinfer-anc-eval`. Run the workflow with
`uv run snakemake --cores all`.
No rule invokes `uv run` internally or lists source files as inputs.
After a code edit, use Snakemake's `--forcerun` for affected rules.

## Configuration and outputs

Use `config.yaml.example` as the manifest template. It specifies `data_dir` and
`progress_dir` separately, datasets with `name`, `zarr_path`, `ts_path`, and
`ancestral_state`, plus `ancestor_chunk_size` and `inference_threads`. Set
`ts_path: null` for a dataset without truth. The ancestral allele comes from a
VCZ field or the REF allele, as specified by the manifest; never derive it from
the truth TS. The input Zarr and truth TS stay read-only.

Output paths use `{name}` followed by a suffix under the configured data folder:

```text
{data_dir}/samples/{name}_samples_masked.zarr/
{data_dir}/configs/{name}_ancestor_inference.toml
{data_dir}/ancestors/{name}_inferred_ancestors.zarr/
{data_dir}/ancestors/{name}_true_ancestors.zarr/  # with truth only
{data_dir}/dataframes/{name}_true_ancestors.csv   # with truth only
{progress_dir}/{rule}/{name}_{rule}.log
```

## Four rules

1. `mask_singletons` copies the sample store with `shutil.copytree` and reads
   `call_genotype` once as a NumPy array. Count current derived calls and called
   haplotypes, treating `-1` as missing. For biallelic input, genotype code 1
   is derived when REF is ancestral, and code 0 otherwise. Write
   `variant_match_eval_singleton_mask = (derived_count == 1)` and
   `variant_match_eval_derived_af = derived_count / called_count` to the copy.
   Use zero when the called count is zero. Keep genotype calls unchanged.
   This loads the genotype array into memory; no chunk reader is needed for
   this MVP. Refresh Zarr metadata after adding the arrays. Existing anc-eval singleton
   and frequency annotations can be stale after genotype errors.
2. `write_inference_config` writes native tsinfer TOML with one sample source,
   the configured ancestral state, one inferred ancestor output, and the match
   sections for later use. Its source has
   `exclude = "INFO/match_eval_singleton_mask == 1"`. vcztools resolves this
   expression to the new variant array. With one source, excluded positions
   contribute no genotypes, and tsinfer removes all-missing sites during its
   normal site-selection pass. The original genotypes need no masking step.
3. `infer_ancestors` calls the current tsinfer library on the generated TOML
   with eight-bit genotype encoding. Its output `variant_position` is the
   authoritative analysis site list.
4. `extract_true_ancestors` maps those positions to the original truth TS and
   selects one derived-mutation ancestor per site, retaining recurrent sites
   on the shared axis. A private TS copy flags internal nodes as samples for
   genotype decoding. Decode ancestor columns in simple sequential batches of
   `ancestor_chunk_size`; the tskit matrix covers all truth sites for nodes in
   one batch, then select the inferred sites. The data are written with
   tsinfer's native ancestor Zarr setup/finalisation helpers.

Snakemake's `all` target requests all inferred panels and only the true panels
and CSVs whose datasets have a non-null `ts_path`. Logs are separate YAML
configured paths. The output Zarr uses Zarr-Python 3 APIs, while tsinfer's
current native writer uses storage format 2.

## True records and metadata

Select one representative mutation per inference site. For a site with one
mutation, select it. For a recurrent site, decode current derived carriers on
the original TS and choose the derived-state mutation whose node covers the
most tracked carriers, breaking ties by mutation order. If no derived carriers
remain, raise an error identifying the site. Ordinary sites use their sole
mutation directly, without decoding carriers. This selection occurs before
internal nodes are flagged as samples.

Emit exactly one CSV row for each inference site, retaining only the selected
mutation. Sort rows by decreasing `true_node_time`, then `inference_site_id`.
The true panel has one haplotype column per unique positive-time selected truth
node. Multiple sites choosing the same node have separate CSV rows sharing
`true_ancestor_id`; their focal positions belong to the same panel column.
Panel columns follow the nodes' first occurrence in the sorted CSV.
Write the original node's *final* genotype state at each inference site; never
force the focal genotype to match a transient mutation state.

The CSV columns are `inference_site_id`, `true_site_id`, `focal_position`,
`inferred_ancestor_id`, `true_ancestor_id`, `true_mutation_id`, `true_mutation_is_singleton`,
`true_node_id`, `true_node_time`, `inferred_node_time`, `derived_af`,
and `num_mutations`. There are no `focal_allele` or `is_primary_for_site` columns.
`num_mutations` retains the original truth site's count, including recurrence.
`derived_af` describes observed derived-allele frequency in the sample VCZ.
True and inferred times use different scales.

`true_mutation_is_singleton` is `True` for a selected mutation whose node has
time zero. Keep its CSV row and site position but leave `true_ancestor_id`
blank and omit its node from the panel. The flag identifies a terminal origin,
including at recurrent sites where separate singleton mutations jointly form
an observed doubleton. `num_mutations` preserves that distinction. Focal lookup
uses only rows whose flag is `False`.

The true store reuses inferred positions, ancestral-first allele strings,
contig information, and sequence intervals. It writes `call_genotype` with
shape `(num_inference_sites, num_true_ancestors, 1)`, `sample_time` from the
original node, a start/end span from its first/last nonmissing inference site,
and all associated focal positions. It retains internal missing calls. The
matcher creates its own ultimate and virtual root nodes, so no synthetic root
haplotypes belong in this store. Original node times are unchanged. Zero-time
selections remain in the CSV; sample matching places their mutations in the
final TS. Split matching requires ancestor groups strictly before contemporary
sample groups and a reference with at most one mutation per site. Recurrence
is allowed in the final raw sample TS.

## Which ID links a matched node to a Zarr haplotype?

VCZ calls the ancestor dimension `samples`. A sample-column index is simply
the zero-based index along that dimension: column 17's haplotype is
`call_genotype[:, 17, 0]`. The final zero selects the sole haplotype of this
haploid ancestor entry. tsinfer labels its ancestor columns with strings such
as `"a17"`; `a` is just the label prefix. It is not an allele, node, or second
index.

Both nonmissing `inferred_ancestor_id` and `true_ancestor_id` in the CSV mean the **actual
`sample_id` string** in their respective Zarr stores. There are no separate
`inferred_sample_id` and `true_sample_id` columns. During matching, tsinfer
adds node metadata including `source`, `sample_id`, and `ploidy_index`. To find
a matched node's haplotype, use `source` to select the store, find its
`sample_id` in that store, and read the matching genotype column. Node IDs and
column indices have no fixed arithmetic relationship; matching groups reorder
ancestors and add root/path-compression nodes. IDs such as `a17` are scoped to
one store. The CSV's `true_node_id` points to the original simulation TS node
whose haplotype was decoded, not a node created by matching.

## Completed checks

The original mutation-level n300 zero-error run produced 6,120 inference sites,
4,639 inferred ancestors, and 6,157 true records. The genotype-error run produced
6,063, 5,443, and 6,100 respectively, and the no-truth branch omitted truth outputs.
The fresh mask disagreed with the inherited one at 365 genotype-error sites.
The simplified implementation yielded identical inferred/true arrays and CSVs
to the original implementation for both simulated inputs. The site-level
extraction above supersedes that historical schema and keeps one site record
per inference site, with shared older-node ancestor columns. Lint and formatting
are checked with
`uv run ruff check lib` and `uv run ruff format --check lib`.
