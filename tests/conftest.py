"""Configuration and bounded Zarr v3 fixture shared by smoke checks."""

import json
import multiprocessing as mp
import pathlib
import shutil
import threading
import weakref

import pytest
import tskit
import yaml

from lib import benchmark, matching
from tests import smoke_helpers


@pytest.fixture
def smoke_config():
    return yaml.safe_load(pathlib.Path("config.yaml.greedy_smoke").read_text())


@pytest.fixture
def tiny(tmp_path, smoke_config):
    settings = dict(smoke_config["smoke"]["fixtures"]["tiny"])
    settings["seed"] = smoke_config["smoke"]["seed"]
    return smoke_helpers.make_tiny(tmp_path / "tiny", settings, smoke_config["hmm"])


@pytest.fixture
def tiny_config(tiny, smoke_config):
    name = "tiny"
    for directory in ("configs", "ancestors", "samples"):
        (tiny.root / directory).mkdir()
    shutil.copy2(tiny.config, tiny.root / "configs" / f"{name}_ancestor_inference.toml")
    shutil.copy2(
        tiny.reference, tiny.root / "ancestors" / f"{name}_inferred_ancestors.trees"
    )
    config = smoke_config
    config["data_dir"] = str(tiny.root)
    config["datasets"] = [
        {
            "name": name,
            "af_cutoff": 1,
            "min_region_size_kb": 0,
            "zarr_path": str(tiny.source),
        }
    ]
    config["performance"].update(
        datasets=[name],
        worker_counts=[1],
        quality_max_matrix_mib=config["smoke"]["fixtures"]["tiny"][
            "quality_max_matrix_mib"
        ],
    )
    return config


@pytest.fixture
def tiny_output(tiny, tiny_config):
    output = tiny.root / "output"
    output.mkdir()
    matching.match_stitched_samples(
        tiny.config,
        tiny.reference,
        output / "samples_raw.trees",
        af_cutoff=1,
        min_region_size_kb=0,
        workers=1,
        sample_prefetch_chunks=1,
        max_pending_haplotypes=2,
        helper_threads=tiny_config["helper_threads"],
        temp_dir=tiny.root,
    )
    return output


@pytest.fixture
def saved_suite(tmp_path, smoke_config, monkeypatch):
    root = tmp_path / "suite"
    root.mkdir()
    monkeypatch.setattr(benchmark, "output_directory", lambda config: root)
    runs = benchmark.trial_runs(smoke_config)
    for run in runs:
        folder = root / run["run_id"]
        folder.mkdir()
        (folder / "trial.json").write_text(
            json.dumps({"run": run, "config": smoke_config})
        )
        (folder / "samples_raw.trees").touch()
    benchmark._write_csv(root / "runs.csv", [{**run, "status": "ok"} for run in runs])
    return root


@pytest.fixture
def lifetime_records(monkeypatch, tiny):
    processes = []
    threads = []
    references = []
    context = mp.get_context("spawn")
    create_process = context.Process
    create_thread = threading.Thread
    load = tskit.load

    def process(*args, **kwargs):
        result = create_process(*args, **kwargs)
        processes.append(result)
        return result

    def thread(*args, **kwargs):
        result = create_thread(*args, **kwargs)
        if kwargs.get("target") in (matching._load_samples, matching._watch_workers):
            threads.append(result)
        return result

    def reference(path, **kwargs):
        result = load(path, **kwargs)
        if pathlib.Path(path) == tiny.reference:
            references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(context, "Process", process)
    monkeypatch.setattr(threading, "Thread", thread)
    monkeypatch.setattr(tskit, "load", reference)
    return {"processes": processes, "threads": threads, "references": references}
