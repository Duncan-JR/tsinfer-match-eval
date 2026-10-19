import csv
import pathlib

from lib import ancestors, benchmark, matching, utils


configfile: "config.yaml"

utils.validate_config(config)
focal_choice = config.get("focal_choice", "left")
ac_cutoff = config["ac_cutoff"]
min_focal_region_size = config["min_focal_region_size"]
helpers = config["helper_threads"]
utils.configure_helpers(helpers)

data_dir = pathlib.Path(config["data_dir"]).expanduser()
progress_dir = pathlib.Path(config["progress_dir"]).expanduser()
datasets = {dataset["name"]: dataset for dataset in config["datasets"]}
for dataset in datasets.values():
    dataset["samples"] = None
    sample_list = dataset.get("sample_list")
    if sample_list is not None:
        with pathlib.Path(sample_list).expanduser().open(newline="") as sample_file:
            sample_ids = [sample_id for row in csv.reader(sample_file) for sample_id in row]
        dataset["samples"] = ",".join(sample_ids)
names = list(datasets)


def panel_config(wildcards):
    suffix = "ancestor_inference" if wildcards.kind == "inferred" else "true_ancestors"
    return data_dir / "configs" / f"{wildcards.name}_{suffix}.toml"


wildcard_constraints:
    kind="inferred|true",


rule all:
    input:
        expand(data_dir / "matches" / "{name}_inferred_samples_raw.trees", name=names),
        expand(data_dir / "matches" / "{name}_inferred_greedy_samples_raw.trees", name=names),
        (expand(data_dir / "matches" / "{name}_inferred_samples_matches.jsonl", name=names)
         if config["path_output"] else []),
        (expand(data_dir / "matches" / "{name}_inferred_greedy_samples_matches.jsonl", name=names)
         if config["path_output"] else []),


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


rule make_inference_config:
    input:
        data_dir / "samples" / "{name}_samples_masked.zarr",
    output:
        data_dir / "configs" / "{name}_ancestor_inference.toml",
    log:
        progress_dir / "make_inference_config" / "{name}_make_inference_config.log",
    params:
        hmm=config["hmm"],
        ancestral_state=lambda wildcards: datasets[wildcards.name]["ancestral_state"],
        include=lambda wildcards: datasets[wildcards.name].get("include"),
        exclude=lambda wildcards: datasets[wildcards.name].get("exclude"),
        samples=lambda wildcards: datasets[wildcards.name].get("samples"),
    run:
        utils.setup_log(pathlib.Path(log[0]))
        ancestor_path = data_dir / "ancestors" / f"{wildcards.name}_inferred_ancestors.zarr"
        utils.make_inference_config(
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


rule make_true_ancestors_config:
    input:
        samples=data_dir / "samples" / "{name}_samples_masked.zarr",
        ancestors=data_dir / "ancestors" / "{name}_true_ancestors.zarr",
    output:
        data_dir / "configs" / "{name}_true_ancestors.toml",
    log:
        progress_dir / "make_true_ancestors_config" / "{name}_true_make_true_ancestors_config.log",
    params:
        hmm=config["hmm"],
        ancestral_state=lambda wildcards: datasets[wildcards.name]["ancestral_state"],
        include=lambda wildcards: datasets[wildcards.name].get("include"),
        exclude=lambda wildcards: datasets[wildcards.name].get("exclude"),
        samples=lambda wildcards: datasets[wildcards.name].get("samples"),
    run:
        utils.setup_log(pathlib.Path(log[0]))
        utils.make_inference_config(
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


rule match_stitched_samples:
    input:
        reference=data_dir / "ancestors" / "{name}_inferred_ancestors.trees",
        ancestors=data_dir / "ancestors" / "{name}_inferred_ancestors.zarr",
        samples=data_dir / "samples" / "{name}_samples_masked.zarr",
        config=data_dir / "configs" / "{name}_ancestor_inference.toml",
    output:
        trees=data_dir / "matches" / "{name}_inferred_greedy_samples_raw.trees",
        matches=([data_dir / "matches" / "{name}_inferred_greedy_samples_matches.jsonl"]
                 if config["path_output"] else []),
    params:
        ac_cutoff=ac_cutoff,
        min_focal_region_size=min_focal_region_size,
        focal_choice=focal_choice,
        workers=config["matching_threads"],
        sample_prefetch_chunks=config["sample_prefetch_chunks"],
        max_pending_haplotypes=config["max_pending_haplotypes"],
        temp_dir=config["temp_dir"],
        helper_threads=helpers,
    threads: utils.cpu_allocation(config["matching_threads"], helpers)
    log:
        progress_dir / "match_stitched_samples" / "{name}_inferred_match_stitched_samples.log",
    run:
        utils.setup_log(pathlib.Path(log[0]))
        required = utils.cpu_allocation(params.workers, params.helper_threads)
        if threads < required:
            raise ValueError(f"Greedy matching needs {required} CPUs including helpers; got {threads}")
        matching.match_stitched_samples(
            pathlib.Path(input.config), pathlib.Path(input.reference), pathlib.Path(output.trees),
            ac_cutoff=params.ac_cutoff, min_focal_region_size=params.min_focal_region_size,
            focal_choice=params.focal_choice, workers=params.workers,
            sample_prefetch_chunks=params.sample_prefetch_chunks,
            max_pending_haplotypes=params.max_pending_haplotypes,
            temp_dir=None if params.temp_dir is None else pathlib.Path(params.temp_dir).expanduser(),
            helper_threads=params.helper_threads,
            match_file_path=pathlib.Path(output.matches[0]) if output.matches else None,
        )


rule match_samples:
    input:
        reference=data_dir / "ancestors" / "{name}_{kind}_ancestors.trees",
        ancestors=data_dir / "ancestors" / "{name}_{kind}_ancestors.zarr",
        samples=data_dir / "samples" / "{name}_samples_masked.zarr",
        config=panel_config,
    output:
        trees=data_dir / "matches" / "{name}_{kind}_samples_raw.trees",
        matches=([data_dir / "matches" / "{name}_{kind}_samples_matches.jsonl"]
                 if config["path_output"] else []),
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
            pathlib.Path(output.matches[0]) if output.matches else None,
            threads,
            params.cache_size,
        )


if "performance" in config:
    performance_dir = benchmark.output_directory(config)
    performance_products = ["manifest.json", "runs.csv", "stages.csv", "complete.json"]
    if config["performance"]["quality_checks"]:
        performance_products.append("quality.json")
    performance_names = config["performance"]["datasets"]

    rule performance:
        input:
            expand(performance_dir / "{product}", product=performance_products),

    rule performance_suite:
        input:
            configs=expand(data_dir / "configs" / "{name}_ancestor_inference.toml", name=performance_names),
            samples=expand(data_dir / "samples" / "{name}_samples_masked.zarr", name=performance_names),
            panels=expand(data_dir / "ancestors" / "{name}_inferred_ancestors.zarr", name=performance_names),
            references=expand(data_dir / "ancestors" / "{name}_inferred_ancestors.trees", name=performance_names),
        output:
            protected(expand(performance_dir / "{product}", product=performance_products)),
        threads: benchmark.required_allocation(config)
        log:
            progress_dir / "performance" / f"{config['performance']['experiment_id']}.log",
        run:
            utils.setup_log(pathlib.Path(log[0]))
            benchmark.run_suite(config, threads, [pathlib.Path(path) for path in workflow.configfiles])
else:
    rule performance:
        run:
            raise ValueError("Add the performance section from config.yaml.example")
