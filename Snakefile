from pathlib import Path

from lib import ancestors, utils


configfile: "config.yaml"

data_dir = Path(config["data_dir"]).expanduser()
progress_dir = Path(config["progress_dir"]).expanduser()
datasets = {dataset["name"]: dataset for dataset in config["datasets"]}
names = list(datasets)
simulated_names = [name for name in names if datasets[name]["ts_path"] is not None]


rule all:
    input:
        expand(data_dir / "ancestors" / "{name}_inferred_ancestors.zarr", name=names),
        expand(data_dir / "ancestors" / "{name}_true_ancestors.zarr", name=simulated_names),
        expand(data_dir / "dataframes" / "{name}_true_ancestors.csv", name=simulated_names),


rule mask_singletons:
    input:
        lambda wildcards: Path(datasets[wildcards.name]["zarr_path"]).expanduser(),
    output:
        directory(data_dir / "samples" / "{name}_samples_masked.zarr"),
    log:
        progress_dir / "mask_singletons" / "{name}_mask_singletons.log",
    run:
        utils.setup_log(Path(log[0]))
        dataset = datasets[wildcards.name]
        utils.add_singleton_mask(
            Path(input[0]), Path(output[0]), dataset["ancestral_state"]
        )


rule write_inference_config:
    input:
        data_dir / "samples" / "{name}_samples_masked.zarr",
    output:
        data_dir / "configs" / "{name}_ancestor_inference.toml",
    log:
        progress_dir / "write_inference_config" / "{name}_write_inference_config.log",
    run:
        utils.setup_log(Path(log[0]))
        dataset = datasets[wildcards.name]
        ancestor_path = data_dir / "ancestors" / f"{wildcards.name}_inferred_ancestors.zarr"
        utils.write_inference_config(
            Path(input[0]), ancestor_path, Path(output[0]), dataset["ancestral_state"]
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
        utils.setup_log(Path(log[0]))
        ancestors.infer_ancestors(Path(input[0]), threads)


rule extract_true_ancestors:
    input:
        inferred=data_dir / "ancestors" / "{name}_inferred_ancestors.zarr",
        samples=data_dir / "samples" / "{name}_samples_masked.zarr",
        truth=lambda wildcards: Path(datasets[wildcards.name]["ts_path"]).expanduser(),
    output:
        panel=directory(data_dir / "ancestors" / "{name}_true_ancestors.zarr"),
        dataframe=data_dir / "dataframes" / "{name}_true_ancestors.csv",
    log:
        progress_dir / "extract_true_ancestors" / "{name}_extract_true_ancestors.log",
    run:
        utils.setup_log(Path(log[0]))
        ancestors.extract_true_ancestors(
            Path(input.inferred),
            Path(input.samples),
            Path(input.truth),
            Path(output.panel),
            Path(output.dataframe),
            config["ancestor_chunk_size"],
        )
