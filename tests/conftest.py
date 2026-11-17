"""Configuration and bounded Zarr v3 fixture shared by smoke checks."""

import json
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
    config = yaml.safe_load(pathlib.Path("config.yaml.example").read_text())
    for dataset, name, cutoff, minimum in zip(
        config["datasets"],
        ["out_of_africa_n100_1mbp", "tgp_chr20_n100_1mbp"],
        [[0.03, 0.05], 0.02],
        [0, 1],
        strict=True,
    ):
        dataset.update(name=name, af_cutoff=cutoff, min_region_size_kb=minimum)
    config.update(
        {'hmm': {'recombination': 0.01, 'mismatch': 0.01},
         'performance': {'experiment_id': 'tiny_checks',
                         'datasets': ['out_of_africa_n100_1mbp', 'tgp_chr20_n100_1mbp'],
                         'modes': ['from_reference'],
                         'methods': ['full_hmm', 'greedy'],
                         'worker_counts': [1, 2],
                         'repeats': 1,
                         'seed': 20261020,
                         'rss_sample_interval_s': 0.05,
                         'trial_timeout_s': 300,
                         'collect_haplotype_timings': False,
                         'collect_stages': True,
                         'quality_checks': True,
                         'resident_max_matrix_mib': 512,
                         'resident_max_private_mib': 16384,
                         'quality_max_matrix_mib': 256,
                         'diagnostics_run': {'mode': 'from_reference',
                                             'workers': 2,
                                             'repeat': 0}},
         'smoke': {'seed': 20261023,
                   'fixtures': {'tiny': {'variants': 13,
                                         'individuals': 7,
                                         'genotype_chunks': [4, 2, 1],
                                         'selected_samples': ['s0', 's1', 's4', 's6'],
                                         'panel_chunks': [4, 2, 1],
                                         'quality_max_matrix_mib': 3e-05}},
                   'validation_delay': {'timeout_s': 60, 'delay_s': 120}}}
    )
    return config


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
    threads = []
    matching_threads = []
    references = []
    create_thread = threading.Thread
    load = tskit.load

    def thread(*args, **kwargs):
        result = create_thread(*args, **kwargs)
        if kwargs.get("target") == matching._load_samples:
            threads.append(result)
        if kwargs.get("name", "").startswith("matching_"):
            matching_threads.append(result)
        return result

    def reference(path, **kwargs):
        result = load(path, **kwargs)
        if pathlib.Path(path) == tiny.reference:
            references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(threading, "Thread", thread)
    monkeypatch.setattr(tskit, "load", reference)
    return {
        "matching_threads": matching_threads,
        "threads": threads,
        "references": references,
    }
