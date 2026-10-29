"""Isolated, sequential runner measurements: ``uv run -m lib.benchmark suite``."""

import argparse
import concurrent.futures as cf
import csv
import dataclasses
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
import signal
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import psutil
import tsinfer
import tskit
import yaml

from . import haplotypes, matching, performance, utils

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Inputs:
    """Existing ancestor reference and its aligned source/configuration."""

    config: pathlib.Path
    reference: pathlib.Path
    samples: pathlib.Path
    ancestors: pathlib.Path


def prepared_inputs(config, dataset):
    root = pathlib.Path(config["data_dir"]).expanduser().resolve()
    name = dataset["name"]
    return Inputs(
        root / "configs" / f"{name}_ancestor_inference.toml",
        root / "ancestors" / f"{name}_inferred_ancestors.trees",
        root / "samples" / f"{name}_samples_masked.zarr",
        root / "ancestors" / f"{name}_inferred_ancestors.zarr",
    )


def output_directory(config):
    return pathlib.Path("data/performance") / config["performance"]["experiment_id"]


def final_products(config):
    """The suite's declared completion products, shared with diagnostics."""
    products = ["manifest.json", "runs.csv", "stages.csv", "complete.json"]
    if config["performance"]["quality_checks"]:
        products.append("quality.json")
    directory = output_directory(config)
    return [directory / product for product in products]


def required_allocation(config):
    return utils.cpu_allocation(
        max(config["performance"]["worker_counts"]), config["helper_threads"]
    )


def validate_config(config, allocation):
    """Validate performance settings at the suite entry point."""
    utils.validate_config(config)
    settings = config["performance"]
    if not settings["methods"] or not set(settings["methods"]) <= {"full_hmm", "greedy"}:
        raise ValueError("performance.methods: choose full_hmm and/or greedy")
    if not settings["modes"] or not set(settings["modes"]) <= {
        "resident_match",
        "from_reference",
    }:
        raise ValueError(
            "performance.modes: choose resident_match and/or from_reference"
        )
    names = {dataset["name"] for dataset in config["datasets"]}
    if not settings["datasets"] or not set(settings["datasets"]) <= names:
        raise ValueError("performance.datasets must select configured datasets")
    for dataset in config["datasets"]:
        if dataset["name"] in settings["datasets"] and isinstance(
            dataset["min_region_size_kb"], list
        ):
            raise ValueError(
                f"Dataset {dataset['name']}: performance trials require a scalar "
                "min_region_size_kb"
            )
    for name in ("methods", "modes", "worker_counts", "datasets"):
        if len(set(settings[name])) != len(settings[name]):
            raise ValueError(f"performance.{name} must not contain duplicates")
    integers = [settings["repeats"], *settings["worker_counts"]]
    if not settings["worker_counts"] or any(
        type(value) is not int or value < 1 for value in integers
    ):
        raise ValueError(
            "Performance worker counts and repeats must be positive integers"
        )
    pending = config["max_pending_haplotypes"]
    if (
        "greedy" in settings["methods"]
        and pending is not None
        and pending < max(settings["worker_counts"])
    ):
        raise ValueError(
            "max_pending_haplotypes must cover every performance.worker_counts "
            "entry; use null to derive it from each trial's workers"
        )
    for name in (
        "rss_sample_interval_s",
        "trial_timeout_s",
        "quality_max_matrix_mib",
        "resident_max_matrix_mib",
        "resident_max_private_mib",
    ):
        if not np.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f"performance.{name} must be positive")
    if (
        not settings["experiment_id"]
        or pathlib.Path(settings["experiment_id"]).name != settings["experiment_id"]
    ):
        raise ValueError("performance.experiment_id must be a directory name")
    if required_allocation(config) > allocation:
        raise ValueError(
            f"Performance suite requires {required_allocation(config)} CPUs; "
            f"got {allocation}"
        )
    affinity = _cpu_affinity()
    available = len(affinity) if affinity is not None else os.cpu_count()
    if available is not None and required_allocation(config) > available:
        raise ValueError(
            f"Performance suite requires {required_allocation(config)} CPUs, "
            f"but this process can use only {available}"
        )


def _sample_jobs(cfg, reference):
    jobs, _, _ = tsinfer.pipeline._build_jobs_with_metadata(
        cfg, reference.sites_position.astype(np.int32)
    )
    jobs.sort(key=lambda job: job.haplotype_index)
    if (
        not jobs
        or len({job.group for job in jobs}) != 1
        or any(job.time != 0 for job in jobs)
    ):
        raise ValueError("Benchmarks require one nonempty present-day sample group")
    return jobs


def _native_reader(cfg, reference, jobs, cache_size, workers):
    positions = reference.sites_position.astype(np.int32)
    ancestral = np.asarray([site.ancestral_state for site in reference.sites()])
    schedule = [(job.source, job.sample_id, job.ploidy_index) for job in jobs]
    return tsinfer.vcz.HaplotypeReader(
        cfg.sources,
        positions,
        ancestral,
        cache_size_mb=cache_size,
        schedule=schedule,
        num_threads=workers,
        read_workers=1,
    )


def _resident_full_hmm(paths, dataset, output, workers, window):
    """Native HMM baseline with decoded input and indexes outside its window."""
    cfg = matching._sample_config(paths.config, paths.reference, output)
    reference = tskit.load(paths.reference)
    jobs = _sample_jobs(cfg, reference)
    reader = _native_reader(
        cfg, reference, jobs, dataset.get("matching_cache_size"), workers
    )
    try:
        calls = {job.haplotype_index: reader.read_haplotype(job).copy() for job in jobs}
        mapper = reader.allele_mapper
        num_alleles = reader.get_num_alleles()
    finally:
        reader.shutdown()
    indexes = tsinfer.matching.MatcherIndexes(
        reference, vestigial_root=False, num_alleles=num_alleles
    )
    parameters = cfg.match.sources["samples"]
    rho = np.full(reference.num_sites, parameters.recombination)
    mu = np.full(reference.num_sites, parameters.mismatch)
    # Native matchers are private to the executor threads; initialise all before timing.
    matchers = [
        tsinfer.matching.AncestorMatcher(indexes, rho, mu) for _ in range(workers)
    ]
    chunks = np.array_split(np.arange(len(jobs)), workers)

    def match_chunk(indices, matcher):
        results = []
        scratch = np.empty(reference.num_sites, dtype=np.int8)
        for index in indices:
            haplotype = calls[int(index)]
            called = np.flatnonzero(haplotype >= 0)
            start, end = 0, len(haplotype)
            if len(called) > 0:
                start, end = int(called[0]), int(called[-1]) + 1
            left, right, parents = matcher.find_path(haplotype, start, end, scratch)
            mutation_sites = (
                np.flatnonzero(
                    (haplotype[start:end] >= 0)
                    & (haplotype[start:end] != scratch[start:end])
                )
                + start
            )
            result = matching.HaplotypeResult(
                int(index),
                left.copy(),
                right.copy(),
                parents.copy(),
                mutation_sites,
                haplotype[mutation_sites].copy(),
            )
            results.append(result)
        return results

    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        with window:
            futures = [
                executor.submit(match_chunk, indices, matcher)
                for indices, matcher in zip(chunks, matchers, strict=True)
            ]
            results = [result for future in futures for result in future.result()]
    tables = reference.dump_tables()
    nodes = matching._create_sample_nodes(tables, jobs)
    for result in results:
        matching.append_result(tables, nodes[result.job_index], result, mapper)
    tsinfer.matching.finalise_match_tables(tables).dump(output)


@performance.sample_memory()
def trial(spec_path):
    """Measure in a fresh process, recording dependency revisions in the suite."""
    spec = json.loads(spec_path.read_text())
    config, run = spec["config"], spec["run"]
    folder = spec_path.parent
    utils.setup_log(folder / "native.log")
    utils.configure_helpers(config["helper_threads"])
    collector = performance.Collector(stages=config["performance"]["collect_stages"])
    try:
        dataset = next(
            item for item in config["datasets"] if item["name"] == run["dataset"]
        )
        paths = prepared_inputs(config, dataset)
        output = folder / "samples_raw.trees"
        workers = run["requested_workers"]
        if run["mode"] == "resident_match":
            cfg = matching._sample_config(paths.config, paths.reference, output)
            reference = tskit.load(paths.reference)
            jobs = _sample_jobs(cfg, reference)
            if workers > len(jobs):
                raise ValueError("Requested workers exceed the sample haplotype count")
            matrix_bytes = reference.num_sites * len(jobs)
            if matrix_bytes > config["performance"]["resident_max_matrix_mib"] * 1024**2:
                raise ValueError("Resident sample calls exceed resident_max_matrix_mib")
            del reference, cfg, jobs
        window = performance.Window()
        if run["method"] == "greedy":
            temp_parent = os.environ.get("MATCH_EVAL_TEMP_DIR", config["temp_dir"])
            kwargs = {
                "af_cutoff": run["af_cutoff"],
                "min_region_size_kb": run["min_region_size_kb"],
                "focal_choice": config.get("focal_choice", "left"),
                "workers": workers,
                "helper_threads": config["helper_threads"],
                "sample_prefetch_chunks": config["sample_prefetch_chunks"],
                "max_pending_haplotypes": config["max_pending_haplotypes"],
                "temp_dir": None
                if temp_parent is None
                else pathlib.Path(temp_parent).expanduser(),
                "collector": collector,
            }
            if run["mode"] == "from_reference":
                with window:
                    workers = matching.match_stitched_samples(
                        paths.config, paths.reference, output, **kwargs
                    )
            else:
                workers = matching.match_stitched_samples(
                    paths.config,
                    paths.reference,
                    output,
                    resident=True,
                    window=window,
                    **kwargs,
                )
        elif run["mode"] == "from_reference":
            with window:
                matching.match_samples(
                    paths.config,
                    paths.reference,
                    output,
                    workers,
                    dataset.get("matching_cache_size"),
                    read_workers=1,
                )
        else:
            _resident_full_hmm(paths, dataset, output, workers, window)
        if run["method"] == "full_hmm":
            matched_ts = tskit.load(output)
            reference = tskit.load(paths.reference)
            count = matched_ts.num_nodes - reference.num_nodes
            if workers > count:
                raise ValueError("Requested workers exceed the matched haplotype count")
            collector.count(matched_haplotypes=count, full_calls=count)
            del matched_ts, reference
        if workers != run["requested_workers"]:
            raise ValueError("Actual worker count differs from requested_workers")
        result = {
            "status": "ok",
            "window": dataclasses.asdict(window),
            "actual_workers": workers,
            "workload": collector.workload,
            "diagnostics": collector.diagnostics,
            "stages": [
                dataclasses.asdict(record) for record in collector.records.values()
            ],
        }
    except Exception:
        result = {"status": "failed", "error": traceback.format_exc()}
        logger.exception("Trial failed")
    result_path = folder / "result.json"
    # Preserve completed matching measurements even if validation is killed or fails.
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] == "ok" and config["performance"]["quality_checks"]:
        validation = performance.Window()
        try:
            with validation:
                report = quality_check(config, run["dataset"], folder)
            result["quality"] = {"status": "ok", **report}
        except Exception:
            result["status"] = "failed"
            result["error"] = "Validation failed; see quality result"
            result["quality"] = {"status": "failed", "error": traceback.format_exc()}
            logger.exception("Validation failed")
        result["validation_window"] = dataclasses.asdict(validation)
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result["status"] == "ok"


def _launch(spec_path, config):
    """Sample private process-tree memory; kill the entire session on timeout."""
    settings = config["performance"]
    samples = []
    error = ""
    started = time.perf_counter()
    spec = json.loads(spec_path.read_text())
    utils.configure_helpers(config["helper_threads"])
    env = os.environ.copy()
    env["MATCH_EVAL_MEMORY_INTERVAL_S"] = str(settings["rss_sample_interval_s"])
    temp_parent = config["temp_dir"]
    if temp_parent is not None:
        temp_parent = pathlib.Path(temp_parent).expanduser()
        temp_parent.mkdir(parents=True, exist_ok=True)
    # The launcher owns this parent so killed trials cannot strand mapped inputs.
    with (
        tempfile.TemporaryDirectory(
            dir=temp_parent, prefix="matching_trial_"
        ) as mapped_root,
        (spec_path.parent / "trial.log").open("w") as log,
    ):
        memory_root = pathlib.Path(mapped_root) / "memory"
        memory_root.mkdir()
        env["MATCH_EVAL_MEMORY_DIR"] = str(memory_root)
        env["MATCH_EVAL_TEMP_DIR"] = mapped_root
        with subprocess.Popen(
            [sys.executable, "-m", "lib.benchmark", "trial", str(spec_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        ) as child:
            try:
                while child.poll() is None:
                    stamp = time.perf_counter()
                    snapshots = [
                        json.loads(path.read_text())
                        for path in memory_root.glob("memory_*.json")
                    ]
                    private = sum(item["private_bytes"] for item in snapshots)
                    pss_values = [item["pss_bytes"] for item in snapshots]
                    pss = (
                        sum(pss_values)
                        if pss_values and all(value is not None for value in pss_values)
                        else None
                    )
                    ages = [
                        stamp - item["timestamp"] for item in snapshots if item["active"]
                    ]
                    stale = sum(
                        age > 2 * settings["rss_sample_interval_s"] for age in ages
                    )
                    samples.append(
                        {
                            "timestamp": stamp,
                            "private_bytes": private,
                            "pss_bytes": pss,
                            "observed_processes": len(ages),
                            "stale_processes": stale,
                            "missing_samples": len(snapshots) == 0,
                            "oldest_sample_age_s": max(ages, default=0),
                        }
                    )
                    if time.perf_counter() - started > settings["trial_timeout_s"]:
                        error = (
                            f"Trial exceeded timeout ({settings['trial_timeout_s']} s)"
                        )
                    if (
                        spec["run"]["mode"] == "resident_match"
                        and private > settings["resident_max_private_mib"] * 1024**2
                    ):
                        error = (
                            "Resident private memory exceeds resident_max_private_mib"
                        )
                    if error:
                        os.killpg(child.pid, signal.SIGKILL)
                        break
                    try:
                        child.wait(timeout=settings["rss_sample_interval_s"])
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        history_root = spec_path.parent / "memory_history"
        shutil.copytree(memory_root, history_root)
    _write_csv(spec_path.parent / "memory_samples.csv", samples)
    result_path = spec_path.parent / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    if error or not result or child.returncode != 0:
        result["status"] = "failed"
        result["error"] = error or result.get(
            "error", f"Trial exited {child.returncode}; see trial.log"
        )
        if settings["quality_checks"] and "window" in result and "quality" not in result:
            result["quality"] = {"status": "failed", "error": result["error"]}
    result["trial_peak_private_bytes"] = max(
        (row["private_bytes"] for row in samples), default=0
    )
    if "window" in result:
        window = result["window"]
        values = [
            row["private_bytes"]
            for row in samples
            if window["start_timestamp"] <= row["timestamp"] <= window["end_timestamp"]
        ]
        result["window_peak_private_bytes"] = max(values, default=0)
    boundaries = [started, *(row["timestamp"] for row in samples), time.perf_counter()]
    result["memory_longest_observation_gap_s"] = max(
        right - left for left, right in itertools.pairwise(boundaries)
    )
    result["trial_peak_pss_bytes"] = max(
        (row["pss_bytes"] for row in samples if row["pss_bytes"] is not None),
        default=None,
    )
    (spec_path.parent / "memory_summary.json").write_text(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key.startswith("memory_") or key.endswith("pss_bytes")
            },
            indent=2,
        )
        + "\n"
    )
    return result


def _revision(root):
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    )
    return {"path": str(root), "commit": commit, "status": status.splitlines()}


def _cpu_affinity():
    """Return the effective Linux CPU mask, including scheduler/cpuset limits."""
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return None


def _machine_resources():
    """Record host topology and the allocation visible to this suite process."""
    affinity = _cpu_affinity()
    topology = []
    if affinity is not None:
        for cpu in affinity:
            directory = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}")
            package = directory / "topology" / "physical_package_id"
            core = directory / "topology" / "core_id"
            if package.exists() and core.exists():
                numa_nodes = sorted(path.name for path in directory.glob("node[0-9]*"))
                topology.append(
                    {
                        "cpu": cpu,
                        "package": int(package.read_text()),
                        "core": int(core.read_text()),
                        "numa_nodes": numa_nodes,
                    }
                )
    cpu_model = platform.processor()
    cpu_info = pathlib.Path("/proc/cpuinfo")
    if cpu_info.exists():
        for line in cpu_info.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    scheduler_variables = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "SLURM_JOB_CPUS_PER_NODE",
        "SLURM_NTASKS",
        "SLURM_JOB_NUM_NODES",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_CPU_BIND",
        "SLURM_CPU_BIND_LIST",
        "SLURM_HINT",
        "SLURM_THREADS_PER_CORE",
        "PBS_JOBID",
        "PBS_NP",
        "PBS_NODEFILE",
        "NCPUS",
        "NSLOTS",
        "LSB_JOBID",
        "LSB_DJOB_NUMPROC",
    )
    return {
        "cpu_model": cpu_model,
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "cpu_affinity": affinity,
        "allowed_cpu_topology": topology,
        "host_memory_bytes": psutil.virtual_memory().total,
        "scheduler": {
            name: os.environ[name] for name in scheduler_variables if name in os.environ
        },
        "temp_directory": str(pathlib.Path(tempfile.gettempdir()).resolve()),
    }


def _manifest(config, allocation):
    source_paths = [pathlib.Path("Snakefile"), *pathlib.Path("lib").glob("*.py")]
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "numba", "tsinfer", "tskit", "zarr", "numcodecs", "psutil")
    }
    datasets = {}
    for dataset in config["datasets"]:
        if dataset["name"] not in config["performance"]["datasets"]:
            continue
        paths = prepared_inputs(config, dataset)
        reference = tskit.load(paths.reference)
        cfg = matching._sample_config(
            paths.config, paths.reference, pathlib.Path("unused.trees")
        )
        parameters = cfg.match.sources["samples"]
        requested = config["hmm"]
        if (parameters.recombination, parameters.mismatch) != (
            requested["recombination"],
            requested["mismatch"],
        ):
            raise ValueError("Prepared native configuration disagrees with hmm settings")
        jobs = _sample_jobs(cfg, reference)
        if max(config["performance"]["worker_counts"]) > len(jobs):
            raise ValueError(
                f"Dataset {dataset['name']} has only {len(jobs)} haplotypes; "
                "reduce performance.worker_counts for the entire comparison"
            )
        identities = [(job.source, job.sample_id, job.ploidy_index) for job in jobs]
        datasets[dataset["name"]] = {
            "inputs": {
                name: str(value) for name, value in dataclasses.asdict(paths).items()
            },
            "cohort_sha256": hashlib.sha256(json.dumps(identities).encode()).hexdigest(),
            "haplotypes": len(jobs),
            "sites": reference.num_sites,
            "reference_nodes": reference.num_nodes,
            "reference_edges": reference.num_edges,
            "sequence_intervals": reference.metadata["sequence_intervals"],
        }
    return {
        "config": config,
        "allocation": allocation,
        "datasets": datasets,
        "repositories": {
            "pipeline": _revision(pathlib.Path.cwd()),
            "tsinfer": _revision(pathlib.Path(tsinfer.__file__).resolve().parent.parent),
        },
        "packages": versions,
        "python": sys.version,
        "machine": platform.uname()._asdict(),
        "machine_resources": _machine_resources(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "filesystem_cache": (
            "Uncontrolled; fresh processes do not imply cold filesystem caches."
        ),
        "jit_cache": (
            "On-disk Numba cache reused; end-to-end includes worker startup/"
            "cache loading/compilation."
        ),
        "resident_window": (
            "Decoded inputs and per-worker HMM/JIT setup excluded; candidate "
            "discovery, extension, cover, gap matching and immediate assembly "
            "included. Not comparable to historical precomputed-interval numbers."
        ),
        "memory": (
            "Sampled sum of private process-tree USS; mapped input capacity "
            "reported separately. Shared mappings are not multiplied by worker "
            "count; samples can miss short peaks. Whole-trial peaks include validation; "
            "matching comparisons use window_peak_private_bytes. The quality matrix "
            "budget excludes trees, one physical source tile and decoder scratch."
        ),
    }


def trial_runs(config):
    """Enumerate the configured trial identities, sharing each Full HMM baseline."""
    settings = config["performance"]
    datasets = {dataset["name"]: dataset for dataset in config["datasets"]}
    rng = random.Random(settings["seed"])
    runs = []
    for repeat, dataset, mode in itertools.product(
        range(settings["repeats"]), settings["datasets"], settings["modes"]
    ):
        selected = datasets[dataset]
        combinations = []
        for method, workers in itertools.product(
            settings["methods"], settings["worker_counts"]
        ):
            cutoffs = (
                utils.parameter_values(selected["af_cutoff"])
                if method == "greedy"
                else [None]
            )
            for cutoff in cutoffs:
                combinations.append((method, workers, cutoff))
        rng.shuffle(combinations)
        for order, (method, workers, cutoff) in enumerate(combinations):
            suffix = f"_af{cutoff}" if method == "greedy" else ""
            run_id = f"{dataset}_{mode}_{method}{suffix}_w{workers}_r{repeat}_o{order}"
            run = {
                "run_id": run_id,
                "dataset": dataset,
                "mode": mode,
                "method": method,
                "af_cutoff": cutoff,
                "min_region_size_kb": (
                    selected["min_region_size_kb"] if method == "greedy" else None
                ),
                "requested_workers": workers,
                "repeat": repeat,
                "order": order,
            }
            runs.append(run)
    return runs


@dataclasses.dataclass(frozen=True)
class DiagnosticTrialKey:
    """Scientific identity shared by the configured and saved trial inventories."""

    dataset: str
    method: str
    af_cutoff: float | None
    min_region_size_kb: float | None


@dataclasses.dataclass
class TrialOutput:
    """A completed tree and its saved trial specification."""

    tree: pathlib.Path
    run: dict


def diagnostic_key(run):
    return DiagnosticTrialKey(
        run["dataset"], run["method"], run["af_cutoff"], run["min_region_size_kb"]
    )


def diagnostic_trials(config):
    """Validate diagnostic coverage without requiring a completed benchmark yet."""
    settings = config["performance"]
    selector = settings.get("diagnostics_run")
    if selector is None:
        raise ValueError(
            "Benchmark-backed diagnostics requires performance.diagnostics_run"
        )
    if (
        selector["mode"] not in settings["modes"]
        or type(selector["workers"]) is not int
        or selector["workers"] not in settings["worker_counts"]
        or type(selector["repeat"]) is not int
        or not 0 <= selector["repeat"] < settings["repeats"]
    ):
        raise ValueError("performance.diagnostics_run selects an unavailable trial")
    selected = [
        run
        for run in trial_runs(config)
        if run["mode"] == selector["mode"]
        and run["requested_workers"] == selector["workers"]
        and run["repeat"] == selector["repeat"]
    ]
    available = {diagnostic_key(run) for run in selected}
    required = set()
    for dataset in config["datasets"]:
        name = dataset["name"]
        required.add(DiagnosticTrialKey(name, "full_hmm", None, None))
        for af, minimum in itertools.product(
            utils.parameter_values(dataset["af_cutoff"]),
            utils.parameter_values(dataset["min_region_size_kb"]),
        ):
            required.add(DiagnosticTrialKey(name, "greedy", af, minimum))
    missing = required - available
    if missing:
        raise ValueError(f"Diagnostic benchmark coverage is incomplete: {missing}")
    return selected


def diagnostic_outputs(config):
    """Resolve from runs.csv and saved specifications, never from filename identities."""
    expected = {diagnostic_key(run) for run in diagnostic_trials(config)}
    root = output_directory(config)
    selector = config["performance"]["diagnostics_run"]
    selected = {}
    with (root / "runs.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        if (
            row["mode"] != selector["mode"]
            or int(row["requested_workers"]) != selector["workers"]
            or int(row["repeat"]) != selector["repeat"]
        ):
            continue
        folder = root / row["run_id"]
        run = json.loads((folder / "trial.json").read_text())["run"]
        for name in ("run_id", "dataset", "mode", "method"):
            if run[name] != row[name]:
                raise ValueError(
                    f"Trial specification disagrees with runs.csv: {folder}"
                )
        for name in ("requested_workers", "repeat", "order"):
            if run[name] != int(row[name]):
                raise ValueError(
                    f"Trial specification disagrees with runs.csv: {folder}"
                )
        for name in ("af_cutoff", "min_region_size_kb"):
            value = None if row[name] == "" else float(row[name])
            if run[name] != value:
                raise ValueError(
                    f"Trial specification disagrees with runs.csv: {folder}"
                )
        key = diagnostic_key(run)
        if key not in expected:
            raise ValueError(f"Unconfigured diagnostic trial in completed suite: {key}")
        if key in selected or row["status"] != "ok":
            raise ValueError(f"Diagnostic trial is duplicated or unsuccessful: {key}")
        tree = folder / "samples_raw.trees"
        if not tree.is_file():
            raise FileNotFoundError(f"Selected diagnostic trial tree is missing: {tree}")
        selected[key] = TrialOutput(tree, run)
    missing = expected - selected.keys()
    if missing:
        raise ValueError(f"Completed suite lacks required diagnostic trials: {missing}")
    return selected


def run_suite(config, allocation, config_paths=None):
    """Measure each greedy AF cutoff, sharing Full HMM trials across cutoffs."""
    validate_config(config, allocation)
    folder = output_directory(config).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise ValueError("Experiment directory is not empty; choose a new experiment_id")
    (folder / "manifest.json").write_text(
        json.dumps(_manifest(config, allocation), indent=2) + "\n"
    )
    (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copy2("uv.lock", folder / "uv.lock")
    if config_paths is not None:
        for index, path in enumerate(config_paths):
            shutil.copy2(path, folder / f"source_config_{index}.yaml")
    settings = config["performance"]
    rows = []
    stage_rows = []
    quality = []
    for run in trial_runs(config):
        run_id = run["run_id"]
        trial_folder = folder / run_id
        trial_folder.mkdir()
        spec = trial_folder / "trial.json"
        spec.write_text(json.dumps({"config": config, "run": run}, indent=2) + "\n")
        logger.info("Starting %s", run_id)
        result = _launch(spec, config)
        row = {
            **run,
            "status": result["status"],
            "error": result.get("error", ""),
            "trial_peak_private_bytes": result["trial_peak_private_bytes"],
        }
        if "window" in result:
            row.update(
                {key: result["window"][key] for key in ("wall_s", "user_s", "system_s")}
            )
            row.update(result["workload"])
            row["actual_workers"] = result["actual_workers"]
            row["window_peak_private_bytes"] = result["window_peak_private_bytes"]
            stage_rows.extend(
                {"run_id": run_id, **record} for record in result["stages"]
            )
        if "validation_window" in result:
            row["validation_wall_s"] = result["validation_window"]["wall_s"]
        if settings["quality_checks"] and "quality" in result:
            quality.append({"run_id": run_id, **result["quality"]})
        rows.append(row)
        _write_csv(folder / "runs.csv", rows)
        _write_csv(folder / "stages.csv", stage_rows, ["run_id", "stage", "scope"])
        if settings["quality_checks"]:
            (folder / "quality.json").write_text(json.dumps(quality, indent=2) + "\n")
        logger.info("Finished %s: %s", run_id, result["status"])
    if any(row["status"] != "ok" for row in [*rows, *quality]):
        raise RuntimeError(
            "Benchmark trials or quality checks failed; see runs.csv/quality.json"
        )
    (folder / "complete.json").write_text(json.dumps({"runs": len(rows)}) + "\n")
    return folder


def _write_csv(path, rows, default_fields=None):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if len(fields) == 0:
        fields = default_fields
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quality_check(config, dataset_name, folder):
    """Exhaustively validate complete diplotypes in bounded physical-chunk batches.

    The matrix budget bounds decoded source calls, not trees, the source tile,
    remapping buffers or the output variant decoder's scratch.
    """
    dataset = next(item for item in config["datasets"] if item["name"] == dataset_name)
    paths = prepared_inputs(config, dataset)
    reference = tskit.load(paths.reference)
    output = tskit.load(folder / "samples_raw.trees")
    cfg = matching._sample_config(paths.config, paths.reference, folder / "unused.trees")
    jobs = _sample_jobs(cfg, reference)
    if output.tables.nodes[: reference.num_nodes] != reference.tables.nodes:
        raise ValueError("Output changed reference nodes")
    if (
        output.tables.individuals != reference.tables.individuals
        or output.tables.populations != reference.tables.populations
    ):
        raise ValueError("Output changed individual/population metadata")
    sample_nodes = np.arange(reference.num_nodes, output.num_nodes, dtype=np.int32)
    if len(sample_nodes) != len(jobs):
        raise ValueError("Output sample count differs")
    for job, node_id in zip(jobs, sample_nodes, strict=True):
        node = output.node(int(node_id))
        expected = {
            "source": job.source,
            "sample_id": job.sample_id,
            "ploidy_index": job.ploidy_index,
        }
        assignments = (
            job.time,
            job.node_flags,
            -1 if job.individual_id is None else job.individual_id,
            -1 if job.population_id is None else job.population_id,
        )
        if (
            node.metadata != expected
            or (node.time, node.flags, node.individual, node.population) != assignments
        ):
            raise ValueError("Output sample metadata differs")
    if not np.array_equal(output.sites_position, reference.sites_position):
        raise ValueError("Output changed reference sites")
    inputs = haplotypes.prepare_samples(cfg.sources["samples"], reference, jobs)
    store = tsinfer.vcz.open_store(inputs.path)
    ploidy = store["call_genotype"].shape[2]
    budget = config["performance"]["quality_max_matrix_mib"] * 1024**2
    bytes_per_sample = reference.num_sites * ploidy * np.dtype(np.int8).itemsize
    capacity = int(budget // bytes_per_sample)
    if capacity < 1:
        raise ValueError(
            "performance.quality_max_matrix_mib cannot fit one complete sample "
            f"({bytes_per_sample} bytes)"
        )
    errors = 0
    batches = 0
    max_matrix_bytes = 0
    mapper = inputs.allele_mapper.forward_map()
    temp_parent = os.environ.get("MATCH_EVAL_TEMP_DIR", config["temp_dir"])
    if temp_parent is not None:
        temp_parent = pathlib.Path(temp_parent).expanduser()
        temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_parent, prefix="quality_") as temporary:
        directory = pathlib.Path(temporary)
        for chunk in inputs.chunks:
            chunk_jobs = chunk.job_indices.reshape(-1, ploidy)
            for start in range(0, len(chunk.columns), capacity):
                stop = start + capacity
                batch = haplotypes.SampleChunk(
                    chunk.block,
                    chunk.columns[start:stop],
                    chunk_jobs[start:stop].ravel(),
                )
                slab = haplotypes.load_sample_chunk(inputs, batch, directory)
                observed = slab.calls.open()
                try:
                    nodes = sample_nodes[batch.job_indices]
                    for variant in output.variants(
                        samples=nodes, isolated_as_missing=False
                    ):
                        site = variant.site.id
                        called = observed[:, site] >= 0
                        expected = mapper[site, observed[called, site]]
                        actual = np.asarray(variant.alleles, dtype=object)[
                            variant.genotypes[called]
                        ]
                        errors += int(np.count_nonzero(expected != actual))
                    max_matrix_bytes = max(max_matrix_bytes, observed.nbytes)
                    batches += 1
                finally:
                    haplotypes.close_mapping(observed)
                    slab.calls.path.unlink()
    if errors > 0:
        raise ValueError(f"{errors} called alleles differ")
    return {
        "called_allele_errors": errors,
        "batches": batches,
        "max_matrix_bytes": max_matrix_bytes,
        "haplotypes": len(jobs),
        "sample_edges": int(np.count_nonzero(output.edges_child >= reference.num_nodes)),
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
