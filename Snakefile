import csv
import pathlib
import re

from lib import ancestors, benchmark, diagnostics, matching, utils


configfile: "config.yaml"

utils.validate_config(config)
focal_choice = config.get("focal_choice", "left")
af_cutoffs = {value for dataset in config["datasets"] for value in utils.parameter_values(dataset["af_cutoff"])}
min_region_sizes_kb = {value for dataset in config["datasets"] for value in utils.parameter_values(dataset["min_region_size_kb"])}
helpers = config["helper_threads"]
utils.configure_helpers(helpers)

data_dir = pathlib.Path(config["data_dir"]).expanduser()
progress_dir = pathlib.Path(config["progress_dir"]).expanduser()
datasets = {dataset["name"]: dict(dataset) for dataset in config["datasets"]}
for dataset in datasets.values():
    dataset["samples"] = None
    sample_list = dataset.get("sample_list")
    if sample_list is not None:
        with pathlib.Path(sample_list).expanduser().open(newline="") as sample_file:
            sample_ids = [sample_id for row in csv.reader(sample_file) for sample_id in row]
        dataset["samples"] = ",".join(sample_ids)
names = list(datasets)
parameter_dir = pathlib.Path("af_{af_cutoff}") / "min_region_kb_{min_region_size_kb}"
greedy_dir = data_dir / "matches" / parameter_dir
greedy_log_dir = progress_dir / "match_stitched_samples" / parameter_dir


def panel_config(wildcards):
    suffix = "ancestor_inference" if wildcards.kind == "inferred" else "true_ancestors"
    return data_dir / "configs" / f"{wildcards.name}_{suffix}.toml"


def greedy_input(wildcards, path):
    utils.validate_parameter_pair(
        datasets[wildcards.name], float(wildcards.af_cutoff),
        float(wildcards.min_region_size_kb),
    )
    return path.format(name=wildcards.name)


wildcard_constraints:
    kind="inferred|true",
    af_cutoff="|".join(re.escape(str(value)) for value in af_cutoffs),
    min_region_size_kb="|".join(re.escape(str(value)) for value in min_region_sizes_kb),


inventory = diagnostics.inventory(config)
full_outputs = inventory.full
stitched_outputs = [run.stitched for run in inventory.runs]
performance_outputs = (
    benchmark.final_products(config) if config.get("performance") is not None else []
)


rule all:
    input:
        full_outputs,
        stitched_outputs,


rule diagnostics:
    input:
        inventory.archive,


rule make_diagnostics:
    input:
        matches=lambda wildcards: diagnostics.matching_inputs(config, inventory),
        samples=expand(data_dir / "samples" / "{name}_samples_masked.zarr", name=names),
        panels=expand(data_dir / "ancestors" / "{name}_inferred_ancestors.zarr", name=names),
        references=expand(data_dir / "ancestors" / "{name}_inferred_ancestors.trees", name=names),
        configs=expand(data_dir / "configs" / "{name}_ancestor_inference.toml", name=names),
        population=lambda wildcards: diagnostics.population_sources(config),
    output:
        archive=inventory.archive,
        masks=[run.mask for run in inventory.runs],
        tables=inventory.tables,
        sites=inventory.sites,
        snapshot=inventory.directory / "config.yaml",
        manifest=inventory.directory / "manifest.json",
    params:
        effective_config=config,
    threads: config["inference_threads"]
    log:
        progress_dir / "diagnostics" / f"{config['config_name']}.log",
    run:
        utils.setup_log(pathlib.Path(log[0]))
        diagnostics.run(params.effective_config, threads)


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
        reference=lambda wildcards: greedy_input(wildcards, str(data_dir / "ancestors" / "{name}_inferred_ancestors.trees")),
        ancestors=data_dir / "ancestors" / "{name}_inferred_ancestors.zarr",
        samples=data_dir / "samples" / "{name}_samples_masked.zarr",
        config=data_dir / "configs" / "{name}_ancestor_inference.toml",
    output:
        trees=greedy_dir / "{name}_inferred_greedy_samples_raw.trees",
    params:
        af_cutoff=lambda wildcards: float(wildcards.af_cutoff),
        min_region_size_kb=lambda wildcards: float(wildcards.min_region_size_kb),
        focal_choice=focal_choice,
        workers=config["matching_threads"],
        sample_prefetch_chunks=config["sample_prefetch_chunks"],
        sample_reader_threads=config.get("sample_reader_threads", 1),
        max_pending_haplotypes=config["max_pending_haplotypes"],
        temp_dir=config["temp_dir"],
        helper_threads=helpers,
    threads: utils.cpu_allocation(config["matching_threads"], helpers, config.get("sample_reader_threads", 1))
    log:
        greedy_log_dir / "{name}_inferred_match_stitched_samples.log",
    run:
        utils.setup_log(pathlib.Path(log[0]))
        required = utils.cpu_allocation(params.workers, params.helper_threads, params.sample_reader_threads)
        if threads < required:
            raise ValueError(f"Greedy matching needs {required} CPUs including helpers; got {threads}")
        matching.match_stitched_samples(
            pathlib.Path(input.config), pathlib.Path(input.reference), pathlib.Path(output.trees),
            af_cutoff=params.af_cutoff, min_region_size_kb=params.min_region_size_kb,
            focal_choice=params.focal_choice, workers=params.workers,
            sample_prefetch_chunks=params.sample_prefetch_chunks,
            sample_reader_threads=params.sample_reader_threads,
            max_pending_haplotypes=params.max_pending_haplotypes,
            temp_dir=None if params.temp_dir is None else pathlib.Path(params.temp_dir).expanduser(),
            helper_threads=params.helper_threads,
        )


rule match_samples:
    input:
        reference=data_dir / "ancestors" / "{name}_{kind}_ancestors.trees",
        ancestors=data_dir / "ancestors" / "{name}_{kind}_ancestors.zarr",
        samples=data_dir / "samples" / "{name}_samples_masked.zarr",
        config=panel_config,
    output:
        trees=data_dir / "matches" / "{name}_{kind}_samples_raw.trees",
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
            threads,
            params.cache_size,
        )


if config.get("performance") is not None:
    performance_names = config["performance"]["datasets"]

    rule performance:
        input:
            performance_outputs,

    rule performance_suite:
        input:
            configs=expand(data_dir / "configs" / "{name}_ancestor_inference.toml", name=performance_names),
            samples=expand(data_dir / "samples" / "{name}_samples_masked.zarr", name=performance_names),
            panels=expand(data_dir / "ancestors" / "{name}_inferred_ancestors.zarr", name=performance_names),
            references=expand(data_dir / "ancestors" / "{name}_inferred_ancestors.trees", name=performance_names),
        output:
            protected(performance_outputs),
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
