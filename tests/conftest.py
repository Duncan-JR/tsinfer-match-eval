"""Configuration and bounded Zarr v3 fixture shared by smoke checks."""

import pathlib
import shutil

import pytest
import yaml

from lib import matching
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
