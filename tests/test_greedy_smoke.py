"""Focused checks for plans/further_greedy_optimisations.md."""

import copy

import pytest

from lib import benchmark, diagnostics, utils
from tests import smoke_helpers


class TestDatasetParameters:
    def test_inventory_uses_each_dataset(self, smoke_config):
        utils.validate_config(smoke_config)
        products = diagnostics.inventory(smoke_config)
        pairs = [
            (run.dataset, run.af_cutoff, run.min_region_size_kb) for run in products.runs
        ]
        assert pairs == [
            ("out_of_africa_n100_1mbp", 0.03, 0),
            ("out_of_africa_n100_1mbp", 0.05, 0),
            ("tgp_chr20_n100_1mbp", 0.02, 1),
        ]

    def test_list_minima_only_rejected_for_benchmarked_datasets(self, smoke_config):
        smoke_config["datasets"][1]["min_region_size_kb"] = [0, 1]
        assert len(diagnostics.inventory(smoke_config).runs) == 4
        smoke_config["performance"]["datasets"] = [smoke_config["datasets"][0]["name"]]
        benchmark.validate_config(
            smoke_config, benchmark.required_allocation(smoke_config)
        )
        smoke_config["performance"]["datasets"].append(
            smoke_config["datasets"][1]["name"]
        )
        with pytest.raises(ValueError, match="scalar min_region_size_kb"):
            benchmark.validate_config(
                smoke_config, benchmark.required_allocation(smoke_config)
            )

    @pytest.mark.parametrize("key", ["af_cutoff", "min_region_size_kb"])
    def test_invalid_dataset_axes(self, smoke_config, key):
        missing = copy.deepcopy(smoke_config)
        del missing["datasets"][0][key]
        with pytest.raises(ValueError, match=key):
            utils.validate_config(missing)
        obsolete = copy.deepcopy(smoke_config)
        obsolete[key] = 0
        with pytest.raises(ValueError, match="Move.*datasets"):
            utils.validate_config(obsolete)
        smoke_config["datasets"][0][key] = [0.1, 0.1]
        with pytest.raises(ValueError, match="duplicates"):
            utils.validate_config(smoke_config)

    def test_reject_other_datasets_pair(self, smoke_config):
        dataset = smoke_config["datasets"][0]
        with pytest.raises(ValueError, match="not configured"):
            utils.validate_parameter_pair(dataset, 0.02, 1)
        utils.validate_parameter_pair(dataset, 0.03, 0)

    def test_trial_grid(self, smoke_config):
        smoke_config["performance"]["worker_counts"] = [1]
        runs = benchmark.trial_runs(smoke_config)
        assert len(runs) == 5
        assert sum(run["method"] == "full_hmm" for run in runs) == 2
        assert sum(run["method"] == "greedy" for run in runs) == 3
        for run in runs:
            if run["method"] == "greedy":
                dataset = next(
                    d for d in smoke_config["datasets"] if d["name"] == run["dataset"]
                )
                utils.validate_parameter_pair(
                    dataset, run["af_cutoff"], run["min_region_size_kb"]
                )

    def test_tiny_fixture_has_multiple_chunks(self, tiny, tmp_path):

        reference, jobs, samples = smoke_helpers.prepared(tiny, tmp_path)
        assert reference.num_sites == 13
        assert len(jobs) == 8
        assert len(samples.chunks) == 3
        assert len(samples.blocks) == 4
        assert tiny.reference.is_file()
