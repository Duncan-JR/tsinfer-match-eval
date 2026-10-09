import csv
import pathlib

from lib import ancestors, haplotypes, matching, utils


configfile: "config.yaml"

focal_choice = config.get("focal_choice", "left")
if focal_choice not in ("left", "right"):
    raise ValueError("focal_choice must be exactly 'left' or 'right'")
ac_cutoff = config["ac_cutoff"]
if not isinstance(ac_cutoff, list) or len(ac_cutoff) == 0:
    raise ValueError("ac_cutoff must be a nonempty list")
if any(isinstance(value, bool) or not isinstance(value, int) for value in ac_cutoff):
    raise ValueError("ac_cutoff values must be integers, not booleans")
if any(value <= 0 for value in ac_cutoff) or any(
    right <= left for left, right in zip(ac_cutoff, ac_cutoff[1:])
):
    raise ValueError("ac_cutoff must contain strictly increasing positive integers")
max_ac_cutoff = max(ac_cutoff)
max_mismatches = config["max_mismatches"]
if not isinstance(max_mismatches, list) or len(max_mismatches) == 0:
    raise ValueError("max_mismatches must be a nonempty list")
if any(isinstance(value, bool) or not isinstance(value, int) for value in max_mismatches):
    raise ValueError("max_mismatches values must be integers, not booleans")
if any(value < 0 for value in max_mismatches) or any(
    right <= left for left, right in zip(max_mismatches, max_mismatches[1:])
):
    raise ValueError("max_mismatches must contain strictly increasing nonnegative integers")
max_mismatch_budget = max(max_mismatches)
focal_methods = config["focal_methods"]
if not isinstance(focal_methods, list) or focal_methods != ["hmm"]:
    raise ValueError("focal_methods must be the list ['hmm']")

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
    ac="|".join(str(value) for value in ac_cutoff),
    k="|".join(str(value) for value in max_mismatches),
    method="|".join(focal_methods),


rule all:
    input:
        expand(data_dir / "ancestors" / "{name}_inferred_ancestors.zarr", name=names),
        expand(data_dir / "ancestors" / "{name}_inferred_ancestors.trees", name=names),
        expand(data_dir / "focal_ancestors" / "{name}_inferred_focal_ancestors.npz", name=names),
        expand(data_dir / "matches" / "{name}_inferred_samples_raw.trees", name=names),
        expand(data_dir / "matches" / "{name}_inferred_samples_matches.jsonl", name=names),
        expand(
            data_dir / "haplotype_intervals" / "{name}_inferred_focal_ancestor_intervals.npz",
            name=names,
        ),
        expand(
            data_dir / "focal_masks" / "{name}_inferred_ac{ac}_k{k}_focal_mask.npz",
            name=names, ac=ac_cutoff, k=max_mismatches,
        ),
        expand(
            data_dir / "matches" / "{name}_inferred_ac{ac}_k{k}_{method}_stitched_samples_raw.trees",
            name=names, ac=ac_cutoff, k=max_mismatches, method=focal_methods,
        ),
        expand(
            data_dir / "matches" / "{name}_inferred_ac{ac}_k{k}_{method}_stitched_samples_matches.jsonl",
            name=names, ac=ac_cutoff, k=max_mismatches, method=focal_methods,
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


rule find_focal_ancestors:
    input:
        samples=data_dir / "samples" / "{name}_samples_masked.zarr",
        ancestors=data_dir / "ancestors" / "{name}_inferred_ancestors.zarr",
    output:
        data_dir / "focal_ancestors" / "{name}_inferred_focal_ancestors.npz",
    log:
        progress_dir / "find_focal_ancestors" / "{name}_inferred_find_focal_ancestors.log",
    params:
        samples=lambda wildcards: datasets[wildcards.name].get("samples"),
        focal_choice=focal_choice,
    run:
        utils.setup_log(pathlib.Path(log[0]))
        haplotypes.find_focal_ancestors(
            pathlib.Path(input.samples),
            pathlib.Path(input.ancestors),
            pathlib.Path(output[0]),
            params.focal_choice,
            params.samples,
        )


rule make_ancestor_intervals:
    input:
        samples=data_dir / "samples" / "{name}_samples_masked.zarr",
        ancestors=data_dir / "ancestors" / "{name}_inferred_ancestors.zarr",
        focal=data_dir / "focal_ancestors" / "{name}_inferred_focal_ancestors.npz",
    output:
        data_dir / "haplotype_intervals" / "{name}_inferred_focal_ancestor_intervals.npz",
    params:
        samples=lambda wildcards: datasets[wildcards.name].get("samples"),
        max_ac_cutoff=max_ac_cutoff,
        max_mismatches=max_mismatch_budget,
        focal_choice=focal_choice,
    threads: workflow.cores
    log:
        progress_dir / "make_ancestor_intervals" / "{name}_inferred_make_ancestor_intervals.log",
    run:
        utils.setup_log(pathlib.Path(log[0]))
        haplotypes.make_ancestor_intervals(
            pathlib.Path(input.samples),
            pathlib.Path(input.ancestors),
            pathlib.Path(input.focal),
            pathlib.Path(output[0]),
            params.max_ac_cutoff,
            params.max_mismatches,
            threads,
            params.focal_choice,
            params.samples,
        )


rule make_focal_mask:
    input:
        data_dir / "haplotype_intervals" / "{name}_inferred_focal_ancestor_intervals.npz",
    output:
        data_dir / "focal_masks" / "{name}_inferred_ac{ac}_k{k}_focal_mask.npz",
    params:
        ac_cutoff=lambda wildcards: int(wildcards.ac),
        max_mismatches=lambda wildcards: int(wildcards.k),
    log:
        progress_dir / "make_focal_mask" / "{name}_inferred_ac{ac}_k{k}_make_focal_mask.log",
    run:
        utils.setup_log(pathlib.Path(log[0]))
        haplotypes.make_focal_mask(
            pathlib.Path(input[0]), pathlib.Path(output[0]),
            params.ac_cutoff, params.max_mismatches,
        )


rule match_stitched_samples:
    input:
        reference=data_dir / "ancestors" / "{name}_inferred_ancestors.trees",
        samples=data_dir / "samples" / "{name}_samples_masked.zarr",
        config=data_dir / "configs" / "{name}_ancestor_inference.toml",
        focal=data_dir / "focal_ancestors" / "{name}_inferred_focal_ancestors.npz",
        masks=expand(
            data_dir / "focal_masks" / "{{name}}_inferred_ac{{ac}}_k{k}_focal_mask.npz",
            k=max_mismatches,
        ),
    output:
        trees=expand(
            data_dir / "matches" / "{{name}}_inferred_ac{{ac}}_k{k}_{{method}}_stitched_samples_raw.trees",
            k=max_mismatches,
        ),
        matches=expand(
            data_dir / "matches" / "{{name}}_inferred_ac{{ac}}_k{k}_{{method}}_stitched_samples_matches.jsonl",
            k=max_mismatches,
        ),
    params:
        ac_cutoff=lambda wildcards: int(wildcards.ac),
        max_mismatches=max_mismatches,
        cache_size=lambda wildcards: datasets[wildcards.name].get("matching_cache_size"),
    threads: workflow.cores
    log:
        progress_dir / "match_stitched_samples" / "{name}_inferred_ac{ac}_{method}_match_stitched_samples.log",
    run:
        utils.setup_log(pathlib.Path(log[0]))
        matching.match_stitched_samples(
            pathlib.Path(input.config), pathlib.Path(input.reference),
            pathlib.Path(input.focal),
            [pathlib.Path(path) for path in input.masks],
            [pathlib.Path(path) for path in output.trees],
            [pathlib.Path(path) for path in output.matches],
            params.ac_cutoff, params.max_mismatches, wildcards.method,
            threads, params.cache_size,
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
