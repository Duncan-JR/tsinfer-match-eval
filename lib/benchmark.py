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

from . import matching, performance, utils

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
    for name in ("af_cutoff", "min_region_size_kb"):
        if isinstance(config[name], list):
            raise ValueError(f"Performance trials require a scalar {name}")
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
        "resident_max_matrix_mib",
        "resident_max_private_mib",
    ):
        if settings[name] <= 0:
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
        window = performance.Window()
        if run["method"] == "greedy":
            temp_parent = os.environ.get("MATCH_EVAL_TEMP_DIR", config["temp_dir"])
            kwargs = {
                "af_cutoff": config["af_cutoff"],
                "min_region_size_kb": config["min_region_size_kb"],
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
        if workers != run["requested_workers"]:
            raise ValueError("Actual worker count differs from requested_workers")
        result = {
            "status": "ok",
            "window": dataclasses.asdict(window),
            "actual_workers": workers,
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


def _launch(spec_path, config):
    """Sample private process-tree memory; kill the entire session on timeout."""
    settings = config["performance"]
    samples = []
    error = ""
    started = time.perf_counter()
    spec = json.loads(spec_path.read_text())
    utils.configure_helpers(config["helper_threads"])
    env = os.environ.copy()
    env["MATCH_EVAL_MEMORY_DIR"] = str(spec_path.parent)
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
                    private = 0
                    for snapshot_path in spec_path.parent.glob("memory_*.json"):
                        snapshot = json.loads(snapshot_path.read_text())
                        private += snapshot["private_bytes"]
                    samples.append((time.perf_counter(), private))
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
    with (spec_path.parent / "memory_samples.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["timestamp", "private_bytes"])
        writer.writerows(samples)
    result_path = spec_path.parent / "result.json"
    if error or not result_path.exists() or child.returncode != 0:
        result = {
            "status": "failed",
            "error": error or f"Trial exited {child.returncode}; see trial.log",
        }
        if result_path.exists() and not error:
            result["error"] = json.loads(result_path.read_text()).get(
                "error", result["error"]
            )
    else:
        result = json.loads(result_path.read_text())
    result["trial_peak_private_bytes"] = max((value for _, value in samples), default=0)
    if result["status"] == "ok":
        window = result["window"]
        values = [
            value
            for stamp, value in samples
            if window["start_timestamp"] <= stamp <= window["end_timestamp"]
        ]
        result["window_peak_private_bytes"] = max(values, default=0)
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
            "count; samples can miss short peaks."
        ),
    }


def run_suite(config, allocation, config_paths=None):
    """Run worker-count/repeat loops at one scalar biological configuration."""
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
    rng = random.Random(settings["seed"])
    rows = []
    stage_rows = []
    quality = []
    for repeat, dataset, mode in itertools.product(
        range(settings["repeats"]), settings["datasets"], settings["modes"]
    ):
        combinations = list(
            itertools.product(settings["methods"], settings["worker_counts"])
        )
        rng.shuffle(combinations)
        for order, (method, workers) in enumerate(combinations):
            run_id = f"{dataset}_{mode}_{method}_w{workers}_r{repeat}_o{order}"
            trial_folder = folder / run_id
            trial_folder.mkdir()
            run = {
                "run_id": run_id,
                "dataset": dataset,
                "mode": mode,
                "method": method,
                "requested_workers": workers,
                "repeat": repeat,
                "order": order,
            }
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
            if result["status"] == "ok":
                row.update(
                    {
                        key: result["window"][key]
                        for key in ("wall_s", "user_s", "system_s")
                    }
                )
                row.update(result["workload"])
                row["actual_workers"] = result["actual_workers"]
                row["window_peak_private_bytes"] = result["window_peak_private_bytes"]
                stage_rows.extend(
                    {"run_id": run_id, **record} for record in result["stages"]
                )
                if settings["quality_checks"]:
                    try:
                        report = quality_check(config, dataset, trial_folder)
                        quality.append({"run_id": run_id, "status": "ok", **report})
                    except Exception:
                        quality.append(
                            {
                                "run_id": run_id,
                                "status": "failed",
                                "error": traceback.format_exc(),
                            }
                        )
            rows.append(row)
            _write_csv(folder / "runs.csv", rows)
            _write_csv(folder / "stages.csv", stage_rows, ["run_id", "stage", "scope"])
            if settings["quality_checks"]:
                (folder / "quality.json").write_text(
                    json.dumps(quality, indent=2) + "\n"
                )
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
    """Untimed sample/reference identity and called-allele reconstruction checks."""
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
    reader = _native_reader(cfg, reference, jobs, dataset.get("matching_cache_size"), 1)
    try:
        observed = np.stack([reader.read_haplotype(job) for job in jobs])
        mapper = reader.allele_mapper
    finally:
        reader.shutdown()
    errors = 0
    for variant in output.variants(samples=sample_nodes, isolated_as_missing=False):
        site = variant.site.id
        called = observed[:, site] >= 0
        expected = mapper.forward_map()[site, observed[called, site]]
        actual = np.asarray(variant.alleles, dtype=object)[variant.genotypes[called]]
        errors += int(np.count_nonzero(expected != actual))
    if errors > 0:
        raise ValueError(f"{errors} called alleles differ")
    return {
        "called_allele_errors": errors,
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
