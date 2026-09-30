import pathlib

from lib import ancestors, evaluation, matching, utils


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


rule mask_singletons:
    input:
        lambda wildcards: pathlib.Path(datasets[wildcards.name]["zarr_path"]).expanduser(),
    output:
        directory(data_dir / "samples" / "{name}_samples_masked.zarr"),
    log:
        progress_dir / "mask_singletons" / "{name}_mask_singletons.log",
    run:
        utils.setup_log(pathlib.Path(log[0]))
        dataset = datasets[wildcards.name]
        utils.add_singleton_mask(
            pathlib.Path(input[0]), pathlib.Path(output[0]), dataset["ancestral_state"]
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
    run:
        utils.setup_log(pathlib.Path(log[0]))
        dataset = datasets[wildcards.name]
        ancestor_path = data_dir / "ancestors" / f"{wildcards.name}_inferred_ancestors.zarr"
        utils.write_inference_config(
            pathlib.Path(input[0]), ancestor_path, pathlib.Path(output[0]),
            dataset["ancestral_state"],
            params.hmm,
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
    run:
        utils.setup_log(pathlib.Path(log[0]))
        dataset = datasets[wildcards.name]
        utils.write_inference_config(
            pathlib.Path(input.samples),
            pathlib.Path(input.ancestors),
            pathlib.Path(output[0]),
            dataset["ancestral_state"],
            params.hmm,
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
    threads: config["matching_threads"]
    run:
        utils.setup_log(pathlib.Path(log[0]))
        matching.match_ancestors(pathlib.Path(input.config), pathlib.Path(output[0]), threads)


rule find_focal_ancestors:
    input:
        samples=data_dir / "samples" / "{name}_samples_masked.zarr",
        ancestors=data_dir / "ancestors" / "{name}_{kind}_ancestors.zarr",
        dataframe=focal_dataframe,
    output:
        data_dir / "focal_ancestors" / "{name}_{kind}_focal_ancestors.npz",
    log:
        progress_dir / "find_focal_ancestors" / "{name}_{kind}_find_focal_ancestors.log",
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
    threads: config["matching_threads"]
    run:
        utils.setup_log(pathlib.Path(log[0]))
        matching.match_samples(
            pathlib.Path(input.config),
            pathlib.Path(input.reference),
            pathlib.Path(output.trees),
            pathlib.Path(output.matches),
            threads,
        )


rule compute_focal_ancestor_stats:
    input:
        focal=data_dir / "focal_ancestors" / "{name}_inferred_focal_ancestors.npz",
        reference=data_dir / "ancestors" / "{name}_inferred_ancestors.trees",
        matches=data_dir / "matches" / "{name}_inferred_samples_matches.jsonl",
        config=data_dir / "configs" / "{name}_ancestor_inference.toml",
    output:
        data_dir / "dataframes" / "{name}_inferred_focal_ancestor_stats.csv",
    params:
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
        dataframe.to_csv(output[0], index=False)
