"""Sequential, isolated sample-matching experiments (``uv run -m lib.benchmark``).

The launcher samples the trial process from outside its GIL. Resident trials
retain canonical compact results until one continuous matching window ends;
from-reference trials use the deployed, bounded-batch production leaves.
"""

import argparse
import csv
import dataclasses
import gc
import hashlib
import importlib.metadata
import itertools
import json
import logging
import os
import pathlib
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback

import numcodecs
import numpy as np
import psutil
import tsinfer
import tskit
import yaml
import zarr

from . import haplotypes, matching, performance, utils

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Inputs:
    """Common prepared products; focal products are resident-only inputs."""

    config: pathlib.Path
    reference: pathlib.Path
    samples: pathlib.Path
    ancestors: pathlib.Path
    focal: pathlib.Path
    intervals: pathlib.Path
    mask: pathlib.Path


@dataclasses.dataclass
class Run:
    """Stable runs.csv schema; missing measurements stay empty on failure."""

    run_id: str
    dataset: str
    ac_cutoff: int
    method: str
    mode: str
    repeat: int
    order: int
    requested_workers: int
    actual_workers: int | None = None
    haplotype_batch_size: int | None = None
    status: str = "failed"
    wall_s: float | None = None
    user_s: float | None = None
    system_s: float | None = None
    window_peak_rss_bytes: int | None = None
    trial_peak_rss_bytes: int | None = None
    matched_haplotypes: int | None = None
    domain_sites: int | None = None
    focal_sites: int | None = None
    gap_sites: int | None = None
    focal_calls: int | None = None
    gap_calls: int | None = None
    full_calls: int | None = None
    reductions: int | None = None
    reduced_nodes: int | None = None
    reduced_edges: int | None = None
    output_trees: str = ""
    output_matches: str = ""
    log_path: str = ""
    error: str = ""


@dataclasses.dataclass
class ResidentReader:
    """Native-encoded, contiguous int8 rows indexed by native haplotype ID."""

    haplotypes: dict[int, np.ndarray]
    allele_mapper: tsinfer.vcz.AlleleMapper
    num_alleles: np.ndarray

    def read_haplotype(self, job):
        return self.haplotypes[job.haplotype_index]


@dataclasses.dataclass
class TrialMeasurement:
    """Measured total and the effective matching concurrency of one trial."""

    window: performance.Window
    actual_workers: int


def prepared_inputs(config, dataset, max_ac_cutoff=None):
    root = pathlib.Path(config["data_dir"]).expanduser().resolve()
    name = dataset["name"]
    cutoff = config["ac_cutoff"][0]
    if max_ac_cutoff is None:
        max_ac_cutoff = max(config["ac_cutoff"])
    budget = config["max_mismatches"][0]
    return Inputs(
        root / "configs" / f"{name}_ancestor_inference.toml",
        root / "ancestors" / f"{name}_inferred_ancestors.trees",
        root / "samples" / f"{name}_samples_masked.zarr",
        root / "ancestors" / f"{name}_inferred_ancestors.zarr",
        root / "focal_ancestors"
        / f"{name}_inferred_max_ac{max_ac_cutoff}_focal_ancestors.npz",
        root / "haplotype_intervals"
        / f"{name}_inferred_max_ac{max_ac_cutoff}_focal_ancestor_intervals.npz",
        root / "focal_masks" / f"{name}_inferred_ac{cutoff}_k{budget}_focal_mask.npz",
    )


def output_directory(config):
    return pathlib.Path("data/performance") / config["performance"]["experiment_id"]


def required_allocation(config):
    """Conservative CPU reservation, independent of CPUs visible on the host.

    Reserve match workers, readers, codec pools, the Zarr helper pool, and
    additional BLAS/Numba helpers. Matching tasks can load synchronously on a
    cache miss, so reserve their extra codec threads too. Async I/O concurrency
    is recorded and capped separately; it is not a count of CPU threads.
    """
    settings = config["performance"]
    helpers = settings["helper_threads"]
    readers = settings["read_workers"]
    codec_workers = readers * helpers["blosc_threads"]
    codec_workers += max(settings["worker_counts"]) * (helpers["blosc_threads"] - 1)
    other_helpers = helpers["zarr_threading_max_workers"]
    other_helpers += helpers["blas_threads"] - 1
    other_helpers += helpers["numba_threads"] - 1
    return max(settings["worker_counts"]) + readers + codec_workers + other_helpers


def validate_ac_cutoffs(cutoffs):
    """Require distinct positive integer cutoffs, preserving configured order."""
    if not isinstance(cutoffs, list) or len(cutoffs) == 0:
        raise ValueError("ac_cutoff must be a nonempty list")
    if any(type(value) is not int or value <= 0 for value in cutoffs):
        raise ValueError("ac_cutoff values must be positive integers, not booleans")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("ac_cutoff must not contain duplicates")


def validate_config(config, allocation):
    validate_ac_cutoffs(config["ac_cutoff"])
    settings = config["performance"]
    experiment = settings["experiment_id"]
    if not experiment or any(
        c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in experiment
    ):
        raise ValueError(
            "performance.experiment_id must use lowercase underscores/numbers"
        )
    if config["max_mismatches"] != [0]:
        raise ValueError("Performance experiments require max_mismatches: [0]")
    if not settings["methods"] or not set(settings["methods"]) <= {
        "full_hmm",
        "reduced_hmm",
        "greedy",
    }:
        raise ValueError("performance.methods: choose full_hmm, reduced_hmm, greedy")
    if not settings["modes"] or not set(settings["modes"]) <= {
        "resident_match",
        "from_reference",
    }:
        raise ValueError("performance.modes: choose resident_match, from_reference")
    for name in ("methods", "modes", "worker_counts", "datasets"):
        if len(set(settings[name])) != len(settings[name]):
            raise ValueError(f"performance.{name} must not contain duplicates")
    names = {dataset["name"] for dataset in config["datasets"]}
    if not settings["datasets"] or not set(settings["datasets"]) <= names:
        raise ValueError("performance.datasets must select configured datasets")
    focal_methods = {"reduced_hmm": "hmm", "greedy": "greedy"}
    for method in settings["methods"]:
        if (
            method in focal_methods
            and focal_methods[method] not in config["focal_methods"]
        ):
            raise ValueError(f"Enable {focal_methods[method]} in focal_methods")
    integers = [
        settings["repeats"],
        settings["read_workers"],
        config["haplotype_batch_size"],
    ]
    integers.extend(settings["worker_counts"])
    integers.extend(settings["helper_threads"].values())
    if not settings["worker_counts"] or any(
        type(value) is not int or value < 1 for value in integers
    ):
        raise ValueError(
            "Worker counts, repeats, batch size and helper limits "
            "must be positive integers"
        )
    for name in (
        "rss_sample_interval_s",
        "trial_timeout_s",
        "resident_max_matrix_mib",
        "resident_max_rss_mib",
    ):
        if settings[name] <= 0:
            raise ValueError(f"performance.{name} must be positive")
    if required_allocation(config) > allocation:
        raise ValueError(
            f"Performance suite needs {required_allocation(config)} CPUs "
            f"including readers/helpers; scheduler granted {allocation}. "
            "Reduce configured concurrency or increase --cores."
        )
    if max(settings["worker_counts"]) > config["haplotype_batch_size"]:
        logger.warning("Batch size limits actual stitched matching concurrency")


def _sample_jobs(cfg, reference):
    positions = reference.sites_position.astype(np.int32)
    jobs, _, _ = tsinfer.pipeline._build_jobs_with_metadata(cfg, positions)
    jobs.sort(key=lambda job: job.haplotype_index)
    if (
        not jobs
        or len({job.group for job in jobs}) != 1
        or any(job.time != 0 for job in jobs)
    ):
        raise ValueError("Benchmarks require one nonempty present-day sample group")
    return jobs


def _native_reader(cfg, reference, jobs, cache_size, workers, read_workers):
    positions = reference.sites_position.astype(np.int32)
    alleles = np.asarray([site.ancestral_state for site in reference.sites()])
    schedule = [(job.source, job.sample_id, job.ploidy_index) for job in jobs]
    return tsinfer.vcz.HaplotypeReader(
        cfg.sources,
        positions,
        alleles,
        cache_size_mb=cache_size,
        schedule=schedule,
        num_threads=workers,
        read_workers=read_workers,
    )


def _resident_reader(cfg, reference, jobs, cache_size, workers, settings):
    matrix_bytes = len(jobs) * reference.num_sites
    if matrix_bytes > settings["resident_max_matrix_mib"] * 1024**2:
        raise ValueError(
            f"Resident int8 matrix needs {matrix_bytes} bytes; "
            "exceeds resident_max_matrix_mib. "
            "Use from_reference for scale runs."
        )
    reader = _native_reader(
        cfg, reference, jobs, cache_size, workers, settings["read_workers"]
    )
    try:
        rows = {}
        for job in jobs:
            haplotype = reader.read_haplotype(job)
            rows[job.haplotype_index] = np.array(
                haplotype, dtype=np.int8, order="C", copy=True
            )
        resident = ResidentReader(rows, reader.allele_mapper, reader.get_num_alleles())
    finally:
        reader.shutdown()
    del reader
    gc.collect()
    return resident


def _warm_resident(method):
    """Warm only JIT code called inside resident matching, with production layouts.

    Native Matcher/_match_region contain no Numba calls. Native job grouping
    already occurs outside the measured window. Only greedy needs a signature.
    """
    if method == "greedy":
        left = np.array([0], dtype=np.int64)
        right = np.array([1], dtype=np.int64)
        nodes = np.array([0], dtype=np.int32)
        out_left = np.empty(1, dtype=np.int64)
        out_right = np.empty(1, dtype=np.int64)
        out_parent = np.empty(1, dtype=np.int32)
        matching._greedy_cover(
            left, right, nodes, left, right, out_left, out_right, out_parent
        )


def _write_full_results(path, paired_results):
    with path.open("w") as output:
        for job, result in paired_results:
            record = {
                "group": job.group,
                "haplotype_index": job.haplotype_index,
                "source": job.source,
                "sample_id": job.sample_id,
                "ploidy_index": job.ploidy_index,
                "time": job.time,
                "path": [dataclasses.asdict(segment) for segment in result.path],
                "mutations": [
                    dataclasses.asdict(mutation) for mutation in result.mutations
                ],
            }
            output.write(json.dumps(record) + "\n")


def _resident_trial(config, dataset, paths, method, workers, folder, collector):
    settings = config["performance"]
    output = folder / "samples_raw.trees"
    matches = folder / "samples_matches.jsonl"
    cfg = matching._sample_config(paths.config, paths.reference, output)
    reference = tskit.load(paths.reference)
    jobs = _sample_jobs(cfg, reference)
    workers = min(workers, len(jobs))
    if method != "full_hmm":
        workers = min(workers, config["haplotype_batch_size"])
    reader = _resident_reader(
        cfg, reference, jobs, dataset.get("matching_cache_size"), workers, settings
    )
    positions = reference.sites_position.astype(np.int32)
    if method == "full_hmm":
        matcher = tsinfer.matching.Matcher(
            reference,
            positions,
            num_alleles=reader.num_alleles,
            source_parameters=cfg.match.sources,
        )
    else:
        native_method = "hmm" if method == "reduced_hmm" else "greedy"
        inputs = matching._prepare_stitched_inputs(
            reference,
            positions,
            paths.focal if native_method == "hmm" else None,
            [paths.mask],
            config["ac_cutoff"][0],
            reader.num_alleles,
            cfg.match.sources["samples"],
            intervals_path=paths.intervals if native_method == "greedy" else None,
        )
    _warm_resident(method)
    gc.collect()
    with performance.Window() as window:
        if method == "full_hmm":
            with performance.measure(collector, "matching"):
                results = list(matcher.match(jobs, reader, num_threads=workers))
        else:
            batches = matching.iter_stitched_batches(
                jobs,
                reader,
                inputs,
                native_method,
                workers,
                config["haplotype_batch_size"],
                collector,
            )
            retained_batches = list(batches)
    # Everything below is deliberately outside the resident matching window.
    if method == "full_hmm":
        results.sort(key=lambda pair: pair[0].haplotype_index)
        _write_full_results(matches, results)
        ts = tsinfer.matching.extend_ts(
            reference, paired_results=results, allele_mapper=reader.allele_mapper
        )
        ts.dump(output)
    else:
        matching.assemble_stitched_batches(
            retained_batches,
            reference,
            reader.allele_mapper,
            config["max_mismatches"],
            [output],
            [matches],
            config["ac_cutoff"][0],
            native_method,
        )
    return TrialMeasurement(window, workers)


def _focal_preparation(config, cfg, paths, folder, workers):
    selection = cfg.sources["samples"].samples
    choice = config.get("focal_choice", "left")
    cutoff = config["ac_cutoff"][0]
    budget = config["max_mismatches"][0]
    focal = folder / f"max_ac{cutoff}_focal_ancestors.npz"
    intervals = folder / f"max_ac{cutoff}_ancestor_intervals.npz"
    mask = folder / f"ac{cutoff}_focal_mask.npz"
    haplotypes.find_focal_ancestors(
        paths.samples, paths.ancestors, focal, choice, selection, max_ac_cutoff=cutoff
    )
    haplotypes.make_ancestor_intervals(
        paths.samples,
        paths.ancestors,
        focal,
        intervals,
        cutoff,
        budget,
        workers,
        choice,
        selection,
    )
    haplotypes.make_focal_mask(intervals, mask, cutoff, budget)
    return dataclasses.replace(paths, focal=focal, intervals=intervals, mask=mask)


def _from_reference_trial(config, dataset, paths, method, workers, folder, collector):
    output = folder / "samples_raw.trees"
    matches = folder / "samples_matches.jsonl"
    # No measured signatures are warmed in this mode.
    with performance.Window() as window:
        if method == "full_hmm":
            with performance.measure(collector, "full_pipeline"):
                matching.match_samples(
                    paths.config,
                    paths.reference,
                    output,
                    matches,
                    workers,
                    dataset.get("matching_cache_size"),
                    config["performance"]["read_workers"],
                )
        else:
            cfg = matching._sample_config(paths.config, paths.reference, output)
            with performance.measure(collector, "focal_preparation"):
                local = _focal_preparation(config, cfg, paths, folder, workers)
            native_method = "hmm" if method == "reduced_hmm" else "greedy"
            matching.match_stitched_samples(
                paths.config,
                paths.reference,
                local.focal if native_method == "hmm" else None,
                [local.mask],
                [output],
                [matches],
                config["ac_cutoff"][0],
                config["max_mismatches"],
                native_method,
                workers,
                dataset.get("matching_cache_size"),
                intervals_path=local.intervals if native_method == "greedy" else None,
                haplotype_batch_size=config["haplotype_batch_size"],
                read_workers=config["performance"]["read_workers"],
                collector=collector,
            )
    records = _records(matches)
    actual = min(workers, len(records))
    if method != "full_hmm":
        actual = min(actual, config["haplotype_batch_size"])
    return TrialMeasurement(window, actual)


def _records(path):
    with path.open() as source:
        records = [json.loads(line) for line in source]
    records.sort(key=lambda record: record["haplotype_index"])
    return records


def _configure_helpers(settings):
    helpers = settings["helper_threads"]
    numcodecs.blosc.set_nthreads(helpers["blosc_threads"])
    zarr.config.set(
        {
            "async.concurrency": helpers["zarr_async_concurrency"],
            "threading.max_workers": helpers["zarr_threading_max_workers"],
        }
    )


def trial(spec_path):
    """Subprocess entry point. Write measurements only after the window ends."""
    spec = json.loads(spec_path.read_text())
    folder = spec_path.parent
    utils.setup_log(folder / "native.log")
    config = spec["config"]
    _configure_helpers(config["performance"])
    collector = performance.Collector(stages=config["performance"]["collect_stages"])
    try:
        validate_config(config, spec["allocation"])
        run = spec["run"]
        if len(config["ac_cutoff"]) != 1 or run["ac_cutoff"] != config["ac_cutoff"][0]:
            raise ValueError("Trial must select its configured single AC cutoff")
        selections = {
            "requested_workers": "worker_counts",
            "method": "methods",
            "mode": "modes",
            "dataset": "datasets",
        }
        for field, setting in selections.items():
            if run[field] not in config["performance"][setting]:
                raise ValueError(f"Trial {field} is outside configured {setting}")
        dataset = next(
            item for item in config["datasets"] if item["name"] == spec["run"]["dataset"]
        )
        paths = prepared_inputs(config, dataset, spec["max_ac_cutoff"])
        function = (
            _resident_trial if run["mode"] == "resident_match" else _from_reference_trial
        )
        measurement = function(
            config,
            dataset,
            paths,
            run["method"],
            run["requested_workers"],
            folder,
            collector,
        )
        if run["method"] == "full_hmm":
            positions = tskit.load(paths.reference).sites_position
            records = _records(folder / "samples_matches.jsonl")
            domain_sites = 0
            for record in records:
                left = min(segment["left"] for segment in record["path"])
                right = max(segment["right"] for segment in record["path"])
                domain_sites += int(
                    np.searchsorted(positions, right) - np.searchsorted(positions, left)
                )
            collector.count(
                matched_haplotypes=len(records),
                domain_sites=domain_sites,
                full_calls=len(records),
            )
        else:
            for name in (
                "focal_sites",
                "gap_sites",
                "focal_calls",
                "gap_calls",
                "reductions",
                "reduced_nodes",
                "reduced_edges",
            ):
                collector.workload.setdefault(name, 0)
        result = {
            "status": "ok",
            "window": dataclasses.asdict(measurement.window),
            "actual_workers": measurement.actual_workers,
            "workload": collector.workload,
            "stages": [
                dataclasses.asdict(record) for record in collector.records.values()
            ],
        }
    except Exception:
        result = {"status": "failed", "error": traceback.format_exc()}
        logger.exception("Trial failed")
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result["status"] == "ok"


def _helper_environment(settings):
    env = os.environ.copy()
    helpers = settings["helper_threads"]
    for name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[name] = str(helpers["blas_threads"])
    env["NUMBA_NUM_THREADS"] = str(helpers["numba_threads"])
    return env


def _launch(spec_path, settings):
    """Sample one child process, terminate failures, and always return a row."""
    samples = []
    error = ""
    spec = json.loads(spec_path.read_text())
    resident = spec["run"]["mode"] == "resident_match"
    resident_limit_bytes = settings["resident_max_rss_mib"] * 1024**2
    resident_error = (
        "Resident RSS exceeds resident_max_rss_mib; use from_reference for scale runs"
    )
    started = time.perf_counter()
    with (spec_path.parent / "trial.log").open("w") as log:
        # sys.executable is the active uv environment, with no second resolver.
        with subprocess.Popen(
            [sys.executable, "-m", "lib.benchmark", "trial", str(spec_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=_helper_environment(settings),
        ) as child:
            process = psutil.Process(child.pid)
            while child.poll() is None:
                try:
                    rss = process.memory_info().rss
                    samples.append((time.perf_counter(), rss))
                except psutil.NoSuchProcess:
                    break
                elapsed = time.perf_counter() - started
                if elapsed > settings["trial_timeout_s"]:
                    error = f"Trial exceeded timeout ({settings['trial_timeout_s']} s)"
                if resident and rss > resident_limit_bytes:
                    error = resident_error
                if error:
                    child.kill()
                    break
                # Wait in the launcher: child progress never depends on its GIL.
                try:
                    child.wait(timeout=settings["rss_sample_interval_s"])
                except subprocess.TimeoutExpired:
                    pass
            child.wait()
    with (spec_path.parent / "rss_samples.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["timestamp", "rss_bytes"])
        writer.writerows(samples)
    result_path = spec_path.parent / "result.json"
    if error:
        result = {"status": "failed", "error": error}
    elif result_path.exists():
        try:
            result = json.loads(result_path.read_text())
        except json.JSONDecodeError as exception:
            result = {
                "status": "failed",
                "error": (
                    f"Invalid trial result after exit code {child.returncode}: "
                    f"{exception}; see trial.log/native.log"
                ),
            }
        if child.returncode != 0 and result["status"] == "ok":
            result = {
                "status": "failed",
                "error": f"Trial exited with code {child.returncode}",
            }
    else:
        result = {
            "status": "failed",
            "error": (
                f"Trial exited with code {child.returncode} without measurements; "
                "see trial.log/native.log"
            ),
        }
    rss_values = [rss for _, rss in samples]
    if result["status"] == "ok":
        window = result["window"]
        boundary_values = [window["start_rss_bytes"], window["end_rss_bytes"]]
        window_values = [
            rss
            for timestamp, rss in samples
            if window["start_timestamp"] <= timestamp <= window["end_timestamp"]
        ]
        result["window_peak_rss_bytes"] = max(boundary_values + window_values)
        rss_values.extend(boundary_values)
    result["trial_peak_rss_bytes"] = max(rss_values, default=None)
    if (
        result["status"] == "ok"
        and resident
        and result["trial_peak_rss_bytes"] > resident_limit_bytes
    ):
        # A short trial can finish between launcher samples; its boundary RSS
        # still participates in the configured resident limit.
        result["status"] = "failed"
        result["error"] = resident_error
    return result


def _revision(root):
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    )
    return {
        "path": str(root),
        "commit": commit,
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def _cohort_manifest(config, dataset, folder, max_ac_cutoff):
    paths = prepared_inputs(config, dataset, max_ac_cutoff)
    reference = tskit.load(paths.reference)
    cfg = matching._sample_config(paths.config, paths.reference, folder / "unused.trees")
    jobs = _sample_jobs(cfg, reference)
    identities = [[job.source, job.sample_id, job.ploidy_index] for job in jobs]
    encoded = json.dumps(identities, separators=(",", ":")).encode()
    roster = folder / f"{dataset['name']}_haplotypes.json"
    roster.write_bytes(encoded + b"\n")
    shutil.copy2(paths.config, folder / f"{dataset['name']}_ancestor_inference.toml")
    sample_list = dataset.get("sample_list")
    if sample_list is not None:
        shutil.copy2(
            pathlib.Path(sample_list).expanduser(),
            folder / f"{dataset['name']}_samples.csv",
        )
    cache_size = dataset.get("matching_cache_size")
    if cache_size is None:
        cache_size = tsinfer.config.DEFAULT_CACHE_SIZE
    return {
        "inputs": {
            name: str(value) for name, value in dataclasses.asdict(paths).items()
        },
        "requested_include": dataset.get("include"),
        "requested_exclude": dataset.get("exclude"),
        "native_include": cfg.sources["samples"].include,
        "native_exclude": cfg.sources["samples"].exclude,
        "inference_intervals": reference.metadata["sequence_intervals"],
        "reference_sites": reference.num_sites,
        "reference_nodes": reference.num_nodes,
        "reference_edges": reference.num_edges,
        "reference_sequence_length": reference.sequence_length,
        "matched_haplotypes": len(jobs),
        "cohort_sha256": hashlib.sha256(encoded).hexdigest(),
        "haplotype_roster": str(roster),
        "matching_cache_size_mib": dataset.get("matching_cache_size"),
        "resolved_matching_cache_size_mib": cache_size,
    }


def _write_manifest(config, allocation, folder, config_paths, max_ac_cutoff):
    versions = {}
    source_hashes = {}
    for path in [pathlib.Path("Snakefile"), *pathlib.Path("lib").glob("*.py")]:
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in (
        "numpy",
        "numba",
        "numcodecs",
        "psutil",
        "tsinfer",
        "tskit",
        "zarr",
        "snakemake",
        "pyyaml",
    ):
        versions[name] = importlib.metadata.version(name)
    selected = set(config["performance"]["datasets"])
    datasets = {}
    for dataset in config["datasets"]:
        if dataset["name"] in selected:
            datasets[dataset["name"]] = _cohort_manifest(
                config, dataset, folder, max_ac_cutoff
            )
    manifest = {
        "configuration": config,
        "resident_input_max_ac_cutoff": max_ac_cutoff,
        "source_sha256": source_hashes,
        "repositories": {
            "tsinfer_match_eval": _revision(pathlib.Path.cwd()),
            "tsinfer": _revision(pathlib.Path(tsinfer.__file__).resolve().parent.parent),
        },
        "python": sys.version,
        "executable": sys.executable,
        "packages": versions,
        "machine": platform.uname()._asdict(),
        "platform": platform.platform(),
        "cpu": {
            "processor": platform.processor(),
            "logical_count": psutil.cpu_count(),
            "physical_count": psutil.cpu_count(logical=False),
        },
        "memory_bytes": psutil.virtual_memory().total,
        "scheduler_allocation": allocation,
        "required_allocation": required_allocation(config),
        "allocation_policy": (
            "matching + readers + reader/matching codec reserve + zarr helpers "
            "+ extra blas/numba helpers"
        ),
        "datasets": datasets,
        "haplotype_batch_size": config["haplotype_batch_size"],
        "workload_call_count_policy": (
            "focal_calls counts reduced HMM regional calls or whole-haplotype "
            "greedy sweeps; gap_calls counts full-HMM regional calls; full_calls "
            "counts native whole-haplotype full-HMM calls"
        ),
        "topology_reuse": {
            "full_hmm": False,
            "reduced_hmm": True,
            "greedy_gap_hmm": True,
        },
        "result_retention": {
            "resident_match": (
                "all native-encoded haplotypes and completed compact results "
                "retained until matching ends; not production bounded memory"
            ),
            "from_reference": (
                "stitched results/tasks bounded by haplotype_batch_size; final "
                "tables, input archives and reader cache still cohort/reference "
                "sized; full HMM uses native retention"
            ),
        },
        "compilation_policy": {
            "resident_match": (
                "warm only greedy run-emitting signature on tiny valid arrays; "
                "fresh matching worker state"
            ),
            "from_reference": "first use compilation included; no warm-up",
        },
        "rss_policy": (
            "launcher samples child RSS plus measured-window boundary samples; "
            "sampled maxima can miss short spikes; trial peak includes "
            "preparation, assembly and measurement serialization"
        ),
        "filesystem_cache_policy": "fresh process does not imply cold filesystem cache",
        "config_sources": [str(path.resolve()) for path in config_paths],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    for index, config_path in enumerate(config_paths):
        shutil.copy2(config_path, folder / f"source_config_{index}.yaml")
    shutil.copy2("uv.lock", folder / "uv.lock")


def run_suite(config, allocation, config_paths=None):
    """Finish a complete suite at each configured cutoff before the next pass.

    Each pass is an ordinary single-cutoff experiment. Resident trials share
    maximum-cutoff preparation; from-reference trials prepare their own cutoff
    inside timing. Publish completion only after every trial and check succeeds.
    """
    validate_config(config, allocation)
    _configure_helpers(config["performance"])
    if config_paths is None:
        config_paths = []
    folder = output_directory(config).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    pass_folders = {folder / f"ac{cutoff}" for cutoff in config["ac_cutoff"]}
    # Snakemake creates output parent directories before entering the rule.
    for entry in folder.iterdir():
        if entry not in pass_folders or not entry.is_dir() or any(entry.iterdir()):
            raise ValueError(
                "Experiment directory is not empty; choose a new experiment_id"
            )
    max_ac_cutoff = max(config["ac_cutoff"])
    succeeded = []
    for cutoff in config["ac_cutoff"]:
        pass_config = {**config, "ac_cutoff": [cutoff]}
        pass_folder = folder / f"ac{cutoff}"
        pass_folder.mkdir(exist_ok=True)
        logger.info("Starting AC %d pass", cutoff)
        pass_succeeded = _run_cutoff(
            pass_config, allocation, pass_folder, config_paths, max_ac_cutoff
        )
        succeeded.append(pass_succeeded)
        logger.info("Finished AC %d pass", cutoff)
    if not all(succeeded):
        raise RuntimeError("Benchmark trials or quality checks failed; see pass results")
    completion = {"ac_cutoff": config["ac_cutoff"]}
    (folder / "complete.json").write_text(json.dumps(completion, indent=2) + "\n")
    return folder


def _run_cutoff(config, allocation, folder, config_paths, max_ac_cutoff):
    """Run single-cutoff trials sequentially, retaining rows for failed trials."""
    _write_manifest(config, allocation, folder, config_paths, max_ac_cutoff)
    cutoff = config["ac_cutoff"][0]
    succeeded = True
    settings = config["performance"]
    rng = random.Random(settings["seed"])
    run_fields = [field.name for field in dataclasses.fields(Run)]
    stage_fields = ["run_id"] + [
        field.name for field in dataclasses.fields(performance.Measurement)
    ]
    with (
        (folder / "runs.csv").open("w", newline="") as runs_file,
        (folder / "stages.csv").open("w", newline="") as stages_file,
    ):
        runs_writer = csv.DictWriter(runs_file, fieldnames=run_fields)
        stages_writer = csv.DictWriter(stages_file, fieldnames=stage_fields)
        runs_writer.writeheader()
        stages_writer.writeheader()
        for repeat in range(settings["repeats"]):
            # Dataset and mode order follow configuration within each repeat.
            for dataset, mode in itertools.product(
                settings["datasets"], settings["modes"]
            ):
                combinations = list(
                    itertools.product(settings["methods"], settings["worker_counts"])
                )
                rng.shuffle(combinations)
                for order, (method, workers) in enumerate(combinations):
                    run_id = (
                        f"{dataset}_ac{cutoff}_{mode}_{method}"
                        f"_w{workers}_r{repeat}_o{order}"
                    )
                    trial_folder = folder / run_id
                    trial_folder.mkdir()
                    run = Run(
                        run_id, dataset, cutoff, method, mode, repeat, order, workers
                    )
                    if method != "full_hmm":
                        run.haplotype_batch_size = config["haplotype_batch_size"]
                    run.output_trees = str(trial_folder / "samples_raw.trees")
                    run.output_matches = str(trial_folder / "samples_matches.jsonl")
                    run.log_path = str(trial_folder / "trial.log")
                    spec_path = trial_folder / "trial.json"
                    spec_path.write_text(
                        json.dumps(
                            {
                                "config": config,
                                "allocation": allocation,
                                "max_ac_cutoff": max_ac_cutoff,
                                "run": dataclasses.asdict(run),
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                    logger.info("Starting %s", run_id)
                    result = _launch(spec_path, settings)
                    run.status = result["status"]
                    if run.status != "ok":
                        succeeded = False
                    run.error = result.get("error", "")
                    run.trial_peak_rss_bytes = result["trial_peak_rss_bytes"]
                    if run.status == "ok":
                        run.actual_workers = result["actual_workers"]
                        run.window_peak_rss_bytes = result["window_peak_rss_bytes"]
                        for name in ("wall_s", "user_s", "system_s"):
                            setattr(run, name, result["window"][name])
                        for name, value in result["workload"].items():
                            setattr(run, name, value)
                        for record in result["stages"]:
                            stages_writer.writerow({"run_id": run_id, **record})
                    runs_writer.writerow(dataclasses.asdict(run))
                    runs_file.flush()
                    stages_file.flush()
                    logger.info("Finished %s: %s %s", run_id, run.status, run.error)
    if settings["quality_checks"]:
        reports = quality_reports(config, folder, max_ac_cutoff)
        if any(report["status"] != "ok" for report in reports):
            succeeded = False
    return succeeded


def quality_reports(config, folder, max_ac_cutoff=None):
    """Implementation is below the launcher so quality is never part of a trial."""
    with (folder / "runs.csv").open(newline="") as source:
        runs = list(csv.DictReader(source))
    reports = []
    datasets = {dataset["name"]: dataset for dataset in config["datasets"]}
    for run in runs:
        if run["status"] != "ok":
            continue
        paths = prepared_inputs(config, datasets[run["dataset"]], max_ac_cutoff)
        baseline = next(
            (
                candidate
                for candidate in runs
                if candidate["status"] == "ok"
                and candidate["method"] == "full_hmm"
                and all(
                    candidate[name] == run[name]
                    for name in (
                        "dataset", "ac_cutoff", "mode", "repeat", "requested_workers"
                    )
                )
            ),
            None,
        )
        try:
            report = _quality(config, paths, run, baseline)
        except Exception:
            report = {
                "run_id": run["run_id"],
                "ac_cutoff": int(run["ac_cutoff"]),
                "status": "failed",
                "error": traceback.format_exc(),
            }
        reports.append(report)
    (folder / "quality.json").write_text(json.dumps(reports, indent=2) + "\n")
    return reports


def _quality(config, paths, run, baseline):
    reference = tskit.load(paths.reference)
    output = tskit.load(run["output_trees"])
    cfg = matching._sample_config(
        paths.config, paths.reference, pathlib.Path(run["output_trees"])
    )
    jobs = _sample_jobs(cfg, reference)
    dataset = next(item for item in config["datasets"] if item["name"] == run["dataset"])
    reader = _native_reader(
        cfg,
        reference,
        jobs,
        dataset.get("matching_cache_size"),
        1,
        config["performance"]["read_workers"],
    )
    try:
        observed = np.stack([reader.read_haplotype(job) for job in jobs])
        allele_table = reader.allele_mapper.forward_map()
        num_alleles = reader.get_num_alleles()
    finally:
        reader.shutdown()
    records = _records(pathlib.Path(run["output_matches"]))
    identities = [(job.source, job.sample_id, job.ploidy_index) for job in jobs]
    record_identities = [
        (record["source"], record["sample_id"], record["ploidy_index"])
        for record in records
    ]
    if identities != record_identities:
        raise ValueError("Output identities disagree with native jobs")
    samples = []
    for node in output.nodes():
        if node.id >= reference.num_nodes:
            identity = (
                node.metadata["source"],
                node.metadata["sample_id"],
                node.metadata["ploidy_index"],
            )
            samples.append((identity, node.id))
    node_lookup = dict(samples)
    sample_nodes = [node_lookup[identity] for identity in identities]
    if output.tables.nodes[: reference.num_nodes] != reference.tables.nodes:
        raise ValueError("Output changed the original reference nodes")
    paths_by_node = {node: [] for node in sample_nodes}
    mutations_by_node = {node: [] for node in sample_nodes}
    for edge in output.edges():
        if edge.child in paths_by_node:
            paths_by_node[edge.child].append(
                {"left": edge.left, "right": edge.right, "parent": edge.parent}
            )
    for mutation in output.mutations():
        if mutation.node in mutations_by_node:
            site = mutation.site
            mutations_by_node[mutation.node].append(
                {
                    "position": float(output.site(site).position),
                    "derived_state": reader.allele_mapper.lookup(
                        site, mutation.derived_state
                    ),
                }
            )
    for job, node_id, record in zip(jobs, sample_nodes, records, strict=True):
        node = output.node(node_id)
        individual = tskit.NULL if job.individual_id is None else job.individual_id
        population = tskit.NULL if job.population_id is None else job.population_id
        if (node.individual, node.population, node.flags) != (
            individual,
            population,
            job.node_flags,
        ):
            raise ValueError("Output changed native sample assignments/flags")
        path = sorted(paths_by_node[node_id], key=lambda segment: segment["left"])
        recorded_path = sorted(record["path"], key=lambda segment: segment["left"])
        if path != recorded_path or mutations_by_node[node_id] != record["mutations"]:
            raise ValueError("Tree sequence and JSONL paths/mutations disagree")
    reconstruction_errors = 0
    for variant in output.variants(samples=sample_nodes, isolated_as_missing=False):
        site = variant.site.id
        called = observed[:, site] >= 0
        expected = allele_table[site, observed[called, site]]
        alleles = np.asarray(variant.alleles, dtype=object)
        actual = alleles[variant.genotypes[called]]
        reconstruction_errors += int(np.count_nonzero(expected != actual))
    if reconstruction_errors:
        raise ValueError(f"{reconstruction_errors} called alleles disagree with input")
    positions = reference.sites_position
    parameters = cfg.match.sources["samples"]
    n = reference.num_nodes
    same_log = np.log(1 - parameters.recombination + parameters.recombination / n)
    switch_log = np.log(parameters.recombination / n)
    common_log_likelihood = 0.0
    switches = 0
    mismatches = 0
    agreement_sites = 0
    compared_sites = 0
    baseline_records = (
        None if baseline is None else _records(pathlib.Path(baseline["output_matches"]))
    )
    for index, record in enumerate(records):
        path = sorted(record["path"], key=lambda segment: segment["left"])
        parents = np.full(reference.num_sites, tskit.NULL, dtype=np.int32)
        for segment in path:
            left = np.searchsorted(positions, segment["left"])
            right = np.searchsorted(positions, segment["right"])
            parents[left:right] = segment["parent"]
        domain = np.flatnonzero(parents != tskit.NULL)
        changed = np.count_nonzero(np.diff(parents[domain]))
        switches += int(changed)
        mut_sites = np.asarray(
            [
                np.searchsorted(positions, mutation["position"])
                for mutation in record["mutations"]
            ],
            dtype=np.int64,
        )
        mismatches += len(mut_sites)
        # Use one common full-reference model, one uniform prior per domain,
        # and native equal-emission treatment of missing calls. Seams receive
        # the same transitions as every other adjacent site in the domain.
        emissions = np.log1p(
            -(num_alleles[domain].astype(float) - 1) * parameters.mismatch
        )
        common_log_likelihood += float(np.sum(emissions))
        if len(mut_sites):
            match_emissions = np.log1p(
                -(num_alleles[mut_sites].astype(float) - 1) * parameters.mismatch
            )
            common_log_likelihood += float(
                np.sum(np.log(parameters.mismatch) - match_emissions)
            )
        common_log_likelihood -= np.log(n)
        common_log_likelihood += (
            changed * switch_log + (len(domain) - 1 - changed) * same_log
        )
        if baseline_records is not None:
            control = baseline_records[index]
            control_identity = (
                control["source"],
                control["sample_id"],
                control["ploidy_index"],
            )
            if control_identity != identities[index]:
                raise ValueError("Baseline haplotype identity differs")
            for segment in control["path"]:
                left = np.searchsorted(positions, segment["left"])
                right = np.searchsorted(positions, segment["right"])
                compared_sites += int(right - left)
                agreement_sites += int(
                    np.count_nonzero(parents[left:right] == segment["parent"])
                )
    return {
        "run_id": run["run_id"],
        "ac_cutoff": int(run["ac_cutoff"]),
        "status": "ok",
        "reconstructed_called_allele_errors": reconstruction_errors,
        "switches": switches,
        "mismatches": mismatches,
        "common_full_reference_log_likelihood": common_log_likelihood,
        "likelihood_model": (
            "uniform initial state over full-reference nodes; rho/n switches, "
            "1-rho+rho/n stays, mu mismatch, native missing emission; "
            "one unconditioned domain"
        ),
        "baseline_run_id": None if baseline is None else baseline["run_id"],
        "parent_agreement_sites": agreement_sites if baseline is not None else None,
        "parent_compared_sites": compared_sites if baseline is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    suite_parser = commands.add_parser("suite")
    suite_parser.add_argument("config", type=pathlib.Path)
    suite_parser.add_argument("--allocation", type=int, required=True)
    trial_parser = commands.add_parser("trial")
    trial_parser.add_argument("spec", type=pathlib.Path)
    args = parser.parse_args()
    if args.command == "trial":
        return 0 if trial(args.spec) else 1
    logging.basicConfig(level=logging.INFO)
    config = yaml.safe_load(args.config.read_text())
    run_suite(config, args.allocation, [args.config])
    return 0


if __name__ == "__main__":
    sys.exit(main())
