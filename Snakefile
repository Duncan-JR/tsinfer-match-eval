import pathlib

from lib import ancestors, evaluation, haplotypes, matching, utils


configfile: "config.yaml"

data_dir = pathlib.Path(config["data_dir"]).expanduser()
progress_dir = pathlib.Path(config["progress_dir"]).expanduser()
datasets = {dataset["name"]: dataset for dataset in config["datasets"]}
names = list(datasets)
panels = [{"name": name, "kind": "inferred"} for name in names]


def panel_config(wildcards):
    suffix = "ancestor_inference" if wildcards.kind == "inferred" else "true_ancestors"
    return data_dir / "configs" / f"{wildcards.name}_{suffix}.toml"


def focal_dataframe(wildcards):
    if wildcards.kind == "inferred":
        return []
    return [data_dir / "dataframes" / f"{wildcards.name}_true_ancestors.csv"]


def metadata_input(wildcards):
    dataset = datasets[wildcards.name]
    source = dataset.get("metadata_source")
    if source is None:
        return []
    if source == "ts":
        field = "ts_path"
    elif source == "csv":
        field = "csv_path"
    else:
        raise ValueError(f"Unknown metadata_source: {source}")
    if dataset.get(field) is None:
        raise ValueError(f"metadata_source: {source} requires {field}")
    return [pathlib.Path(dataset[field]).expanduser()]


def metadata_params(wildcards):
    dataset = datasets[wildcards.name]
    fields = (
        "metadata_source", "ts_path", "csv_path",
        "zarr_id_field", "csv_id_field", "pop_field", "samples",
    )
    return {field: dataset.get(field) for field in fields}


def enrich_populations(dataframe, samples_path, dataset):
    metadata = utils.read_population_metadata(samples_path, dataset)
    if metadata is None:
        return dataframe
    return dataframe.merge(
        metadata, on=["source", "sample_id", "ploidy_index"],
        how="left", validate="many_to_one", sort=False,
    )


wildcard_constraints:
    kind="inferred|true",


rule all:
    input:
        expand(data_dir / "ancestors" / "{name}_inferred_ancestors.zarr", name=names),
        [
            data_dir / "ancestors" / "{name}_{kind}_ancestors.trees".format(**panel)
            for panel in panels
        ],
        [
            data_dir / "focal_ancestors" / "{name}_{kind}_focal_ancestors.npz".format(**panel)
            for panel in panels
        ],
        [
            data_dir / "matches" / "{name}_{kind}_samples_raw.trees".format(**panel)
            for panel in panels
        ],
        [
            data_dir / "matches" / "{name}_{kind}_samples_matches.jsonl".format(**panel)
            for panel in panels
        ],
        expand(
            data_dir / "dataframes" / "{name}_inferred_focal_ancestor_stats.csv",
            name=names,
        ),
        expand(
            data_dir / "haplotype_intervals" / "{name}_inferred_focal_ancestor_intervals.npz",
            name=names,
        ),


rule annotate_derived_counts:
    input:
        lambda wildcards: pathlib.Path(datasets[wildcards.name]["zarr_path"]).expanduser(),
    output:
        directory(data_dir / "samples" / "{name}_samples_masked.zarr"),
    log:
        progress_dir / "annotate_derived_counts" / "{name}_annotate_derived_counts.log",
    params:
        samples=lambda wildcards: datasets[wildcards.name].get("samples"),
        ancestral_state=lambda wildcards: datasets[wildcards.name]["ancestral_state"],
    run:
        utils.setup_log(pathlib.Path(log[0]))
        utils.annotate_derived_counts(
            pathlib.Path(input[0]), pathlib.Path(output[0]),
            params.ancestral_state, params.samples,
        )


rule write_inference_config:
    input:
        data_dir / "samples" / "{name}_samples_masked.zarr",
    output:
        data_dir / "configs" / "{name}_ancestor_inference.toml",
    log:
        progress_dir / "write_inference_config" / "{name}_write_inference_config.log",
    params:
        hmm=config["hmm"],
        ancestral_state=lambda wildcards: datasets[wildcards.name]["ancestral_state"],
        include=lambda wildcards: datasets[wildcards.name].get("include"),
        exclude=lambda wildcards: datasets[wildcards.name].get("exclude"),
        samples=lambda wildcards: datasets[wildcards.name].get("samples"),
    run:
        utils.setup_log(pathlib.Path(log[0]))
        ancestor_path = data_dir / "ancestors" / f"{wildcards.name}_inferred_ancestors.zarr"
        utils.write_inference_config(
            pathlib.Path(input[0]), ancestor_path, pathlib.Path(output[0]),
            params.ancestral_state,
            params.hmm,
            params.include, params.exclude, params.samples,
        )


rule infer_ancestors:
    input:
        data_dir / "configs" / "{name}_ancestor_inference.toml",
    output:
        directory(data_dir / "ancestors" / "{name}_inferred_ancestors.zarr"),
    log:
        progress_dir / "infer_ancestors" / "{name}_infer_ancestors.log",
    threads: config["inference_threads"]
    run:
        utils.setup_log(pathlib.Path(log[0]))
        ancestors.infer_ancestors(pathlib.Path(input[0]), threads)


rule extract_true_ancestors:
    input:
        inferred=data_dir / "ancestors" / "{name}_inferred_ancestors.zarr",
        samples=data_dir / "samples" / "{name}_samples_masked.zarr",
        truth=lambda wildcards: pathlib.Path(datasets[wildcards.name]["ts_path"]).expanduser(),
    output:
        panel=directory(data_dir / "ancestors" / "{name}_true_ancestors.zarr"),
        dataframe=data_dir / "dataframes" / "{name}_true_ancestors.csv",
    log:
        progress_dir / "extract_true_ancestors" / "{name}_extract_true_ancestors.log",
    run:
        utils.setup_log(pathlib.Path(log[0]))
        ancestors.extract_true_ancestors(
            pathlib.Path(input.inferred),
            pathlib.Path(input.samples),
            pathlib.Path(input.truth),
            pathlib.Path(output.panel),
            pathlib.Path(output.dataframe),
            config["ancestor_chunk_size"],
        )


rule write_true_ancestors_config:
    input:
        samples=data_dir / "samples" / "{name}_samples_masked.zarr",
        ancestors=data_dir / "ancestors" / "{name}_true_ancestors.zarr",
    output:
        data_dir / "configs" / "{name}_true_ancestors.toml",
    log:
        progress_dir / "write_true_ancestors_config" / "{name}_true_write_true_ancestors_config.log",
    params:
        hmm=config["hmm"],
        ancestral_state=lambda wildcards: datasets[wildcards.name]["ancestral_state"],
        include=lambda wildcards: datasets[wildcards.name].get("include"),
        exclude=lambda wildcards: datasets[wildcards.name].get("exclude"),
        samples=lambda wildcards: datasets[wildcards.name].get("samples"),
    run:
        utils.setup_log(pathlib.Path(log[0]))
        utils.write_inference_config(
            pathlib.Path(input.samples),
            pathlib.Path(input.ancestors),
            pathlib.Path(output[0]),
            params.ancestral_state,
            params.hmm,
            params.include, params.exclude, params.samples,
        )


rule match_ancestors:
    input:
        ancestors=data_dir / "ancestors" / "{name}_{kind}_ancestors.zarr",
        samples=data_dir / "samples" / "{name}_samples_masked.zarr",
        config=panel_config,
    output:
        data_dir / "ancestors" / "{name}_{kind}_ancestors.trees",
    log:
        progress_dir / "match_ancestors" / "{name}_{kind}_match_ancestors.log",
    params:
        cache_size=lambda wildcards: datasets[wildcards.name].get("matching_cache_size"),
    threads: config["matching_threads"]
    run:
        utils.setup_log(pathlib.Path(log[0]))
        matching.match_ancestors(
            pathlib.Path(input.config), pathlib.Path(output[0]), threads, params.cache_size
        )


rule find_focal_ancestors:
    input:
        samples=data_dir / "samples" / "{name}_samples_masked.zarr",
        ancestors=data_dir / "ancestors" / "{name}_{kind}_ancestors.zarr",
        dataframe=focal_dataframe,
    output:
        data_dir / "focal_ancestors" / "{name}_{kind}_focal_ancestors.npz",
    log:
        progress_dir / "find_focal_ancestors" / "{name}_{kind}_find_focal_ancestors.log",
    params:
        samples=lambda wildcards: datasets[wildcards.name].get("samples"),
    run:
        utils.setup_log(pathlib.Path(log[0]))
        dataframe_path = None
        if wildcards.kind == "true":
            dataframe_path = pathlib.Path(input.dataframe[0])
        matching.find_focal_ancestors(
            pathlib.Path(input.samples),
            pathlib.Path(input.ancestors),
            pathlib.Path(output[0]),
            datasets[wildcards.name]["ancestral_state"],
            dataframe_path,
            params.samples,
        )


rule construct_focal_ancestor_intervals:
    input:
        samples=data_dir / "samples" / "{name}_samples_masked.zarr",
        ancestors=data_dir / "ancestors" / "{name}_inferred_ancestors.zarr",
        focal=data_dir / "focal_ancestors" / "{name}_inferred_focal_ancestors.npz",
    output:
        data_dir / "haplotype_intervals" / "{name}_inferred_focal_ancestor_intervals.npz",
    params:
        samples=lambda wildcards: datasets[wildcards.name].get("samples"),
        max_ac_cutoff=max(config["ac_cutoff"]),
        max_mismatches=config["haplotype_compare"]["max_mismatches"],
        ancestral_state=lambda wildcards: datasets[wildcards.name]["ancestral_state"],
    threads: workflow.cores
    log:
        progress_dir / "construct_focal_ancestor_intervals" / "{name}_inferred_construct_focal_ancestor_intervals.log",
    run:
        utils.setup_log(pathlib.Path(log[0]))
        haplotypes.construct_focal_ancestor_intervals(
            pathlib.Path(input.samples),
            pathlib.Path(input.ancestors),
            pathlib.Path(input.focal),
            pathlib.Path(output[0]),
            params.ancestral_state,
            params.max_ac_cutoff,
            params.max_mismatches,
            threads,
            params.samples,
        )

rule match_samples:
    input:
        reference=data_dir / "ancestors" / "{name}_{kind}_ancestors.trees",
        ancestors=data_dir / "ancestors" / "{name}_{kind}_ancestors.zarr",
        samples=data_dir / "samples" / "{name}_samples_masked.zarr",
        config=panel_config,
    output:
        trees=data_dir / "matches" / "{name}_{kind}_samples_raw.trees",
        matches=data_dir / "matches" / "{name}_{kind}_samples_matches.jsonl",
    log:
        progress_dir / "match_samples" / "{name}_{kind}_match_samples.log",
    params:
        cache_size=lambda wildcards: datasets[wildcards.name].get("matching_cache_size"),
    threads: config["matching_threads"]
    run:
        utils.setup_log(pathlib.Path(log[0]))
        matching.match_samples(
            pathlib.Path(input.config),
            pathlib.Path(input.reference),
            pathlib.Path(output.trees),
            pathlib.Path(output.matches),
            threads,
            params.cache_size,
        )


rule compute_focal_ancestor_stats:
    input:
        samples=data_dir / "samples" / "{name}_samples_masked.zarr",
        metadata=metadata_input,
        focal=data_dir / "focal_ancestors" / "{name}_inferred_focal_ancestors.npz",
        reference=data_dir / "ancestors" / "{name}_inferred_ancestors.trees",
        matches=data_dir / "matches" / "{name}_inferred_samples_matches.jsonl",
        config=data_dir / "configs" / "{name}_ancestor_inference.toml",
    output:
        data_dir / "dataframes" / "{name}_inferred_focal_ancestor_stats.csv",
    params:
        metadata=metadata_params,
        samples=lambda wildcards: datasets[wildcards.name].get("samples"),
        ac_cutoff=config["ac_cutoff"],
    log:
        progress_dir / "compute_focal_ancestor_stats" / "{name}_inferred_compute_focal_ancestor_stats.log",
    run:
        utils.setup_log(pathlib.Path(log[0]))
        dataframe = evaluation.compute_focal_ancestor_stats(
            pathlib.Path(input.focal),
            pathlib.Path(input.reference),
            pathlib.Path(input.matches),
            pathlib.Path(input.config),
            params.ac_cutoff,
        )
        dataframe.insert(0, "panel_kind", "inferred")
        dataframe.insert(0, "dataset", wildcards.name)
        dataframe = enrich_populations(
            dataframe, pathlib.Path(input.samples), datasets[wildcards.name]
        )
        dataframe.to_csv(output[0], index=False)
