"""Configuration and bounded Zarr v3 fixture shared by smoke checks."""

import pathlib

import pytest
import yaml

from tests import smoke_helpers


@pytest.fixture
def smoke_config():
    return yaml.safe_load(pathlib.Path("config.yaml.greedy_smoke").read_text())


@pytest.fixture
def tiny(tmp_path, smoke_config):
    settings = dict(smoke_config["smoke"]["fixtures"]["tiny"])
    settings["seed"] = smoke_config["smoke"]["seed"]
    return smoke_helpers.make_tiny(tmp_path / "tiny", settings, smoke_config["hmm"])
