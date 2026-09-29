# tsinfer matching evaluation

A small Snakemake pipeline that builds inferred and true ancestor panels from
existing sample VCZs. It reads simulations from `../tsinfer-anc-eval` and uses
an editable `../tsinfer` checkout. The workflow calls functions in
`lib/utils.py` and `lib/ancestors.py` directly.

## Run

From this repository, with both sibling repositories available:

```sh
cp config.yaml.example config.yaml
uv sync
uv run snakemake --cores all
```

After changing code in `lib`, rerun affected outputs explicitly because these
files are intentionally not Snakemake inputs:

```sh
uv run snakemake --cores all --forcerun mask_singletons
```

Edit `config.yaml` to choose datasets and output folders. Paths in the YAML are
relative to this working directory; `~` is expanded for input paths. Each
`datasets` entry gives a `name`, `zarr_path`, `ts_path`, and `ancestral_state`.
Use `ts_path: null` for an inference-only run. Ancestral state may be
`{field: variant_ancestral_state}` or `{is_reference: true}`. The input is one
phased, single-contig VCZ per dataset, with all samples included.

The `data_dir` setting contains category folders. For dataset `{name}`:

```text
{data_dir}/samples/{name}_samples_masked.zarr/
{data_dir}/configs/{name}_ancestor_inference.toml
{data_dir}/ancestors/{name}_inferred_ancestors.zarr/
{data_dir}/ancestors/{name}_true_ancestors.zarr/  # truth TS only
{data_dir}/dataframes/{name}_true_ancestors.csv   # truth TS only
```

Rule logs go under the separately configured `progress_dir`. The sample store
is a regular copy of the input. Its genotypes are unchanged; the name
`_samples_masked.zarr` means that the copy contains the fresh
`variant_match_eval_singleton_mask` annotation. The generated native tsinfer
TOML excludes that mask. The annotation is computed from current genotype
calls, so it remains correct after anc-eval has added genotype errors. A second
annotation, `variant_match_eval_derived_af`, records derived allele count divided
by called haplotypes. Missing calls contribute to neither count. Creating these
annotations reads the genotype array into memory, which keeps this MVP simple
and suits the example datasets.

The inferred panel's positions define the true panel's site axis. For each
retained site, the true panel contains one haplotype per original truth mutation,
including recurrence and back mutations. The CSV records its source TS IDs,
inferred ancestor association, times, frequency, and primary-event flag.
Mutation-node genotypes are decoded from a private flags-only TS copy in
chunks of `ancestor_chunk_size`. No matching command runs in this MVP.

In the CSV, `inferred_ancestor_id` and `true_ancestor_id` are **strings** from
the corresponding Zarr store's `sample_id` array. For instance, `"a17"` may be
at column 17, whose haplotype is `call_genotype[:, 17, 0]`. These IDs are scoped
to their store. Later matching writes `source`, `sample_id`, and `ploidy_index`
to node metadata, which supplies the node-to-Zarr join. A matched tree-sequence
node ID is unrelated to its Zarr column index. `true_node_id` refers to the
original simulation TS. The matcher creates its own root nodes; the ancestor
Zarr stores have no synthetic root columns.

## Checked examples

The zero-error, genotype-error, and `ts_path: null` n300 examples completed.
The new inferred and true panels and CSVs matched the previous implementation
exactly. The error input's fresh singleton mask differed from its inherited
annotation at 365 sites. The normal example produced 6,120 inference sites,
4,639 inferred ancestors, and 6,157 true mutation records. The error example
produced 6,063 sites, 5,443 inferred ancestors, and 6,100 true records. The
no-truth branch produced only the inferred panel. `uv run ruff check lib` and
`uv run ruff format --check lib` are the project code checks. No unit-test suite
is included.

See [plans/initial_mvp.md](plans/initial_mvp.md) for the extraction and ID
contracts. The source VCZs and truth TS were read without modification.
