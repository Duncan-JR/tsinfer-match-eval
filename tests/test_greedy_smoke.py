"""Focused checks for plans/further_greedy_optimisations.md."""

import copy
import csv
import json

import numpy as np
import pytest
import tskit

from lib import benchmark, diagnostics, haplotypes, matching, utils
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


class TestQualityBatches:
    def test_batched_matches_dense(self, tiny, tiny_config, tiny_output, monkeypatch):
        original = haplotypes.load_sample_chunk
        allocations = []

        def load(inputs, chunk, directory):
            slab = original(inputs, chunk, directory)
            allocations.append(int(np.prod(slab.calls.shape)))
            assert len(chunk.job_indices) % 2 == 0
            return slab

        monkeypatch.setattr(haplotypes, "load_sample_chunk", load)
        report = benchmark.quality_check(tiny_config, "tiny", tiny_output)
        assert report["called_allele_errors"] == 0
        assert report["batches"] == 4
        budget = tiny_config["performance"]["quality_max_matrix_mib"] * 1024**2
        assert max(allocations) <= budget
        reference, jobs, _ = smoke_helpers.prepared(tiny, tiny.root)
        cfg = matching._sample_config(
            tiny.config, tiny.reference, tiny_output / "unused"
        )
        reader = benchmark._native_reader(cfg, reference, jobs, 1, 1)
        try:
            observed = np.stack([reader.read_haplotype(job) for job in jobs])
            mapper = reader.allele_mapper.forward_map()
        finally:
            reader.shutdown()
        output = tskit.load(tiny_output / "samples_raw.trees")
        errors = 0
        for variant in output.variants(isolated_as_missing=False):
            site = variant.site.id
            called = observed[:, site] >= 0
            expected = mapper[site, observed[called, site]]
            actual = np.asarray(variant.alleles)[variant.genotypes[called]]
            errors += int(np.count_nonzero(expected != actual))
        assert errors == report["called_allele_errors"]

    def test_one_sample_must_fit(self, tiny_config, tiny_output):
        tiny_config["performance"]["quality_max_matrix_mib"] = 0.000001
        with pytest.raises(ValueError, match="cannot fit one complete sample"):
            benchmark.quality_check(tiny_config, "tiny", tiny_output)

    def test_validation_failure_preserves_matching(self, tiny, tiny_config, monkeypatch):
        folder = tiny.root / "trial"
        folder.mkdir()
        run = benchmark.trial_runs(tiny_config)[0]
        run.update(method="greedy", af_cutoff=1, min_region_size_kb=0)
        spec = folder / "trial.json"
        spec.write_text(json.dumps({"config": tiny_config, "run": run}))

        def fail(*args):
            raise ValueError("altered called allele")

        monkeypatch.setattr(benchmark, "quality_check", fail)
        assert not benchmark.trial(spec)
        result = json.loads((folder / "result.json").read_text())
        assert result["status"] == "failed"
        assert result["window"]["wall_s"] > 0
        assert result["quality"]["status"] == "failed"
        assert (
            result["validation_window"]["start_timestamp"]
            >= result["window"]["end_timestamp"]
        )
        result.update(trial_peak_private_bytes=1, window_peak_private_bytes=1)
        suite = folder / "suite"
        monkeypatch.setattr(benchmark, "output_directory", lambda config: suite)
        monkeypatch.setattr(benchmark, "_manifest", lambda *args: {})
        monkeypatch.setattr(benchmark, "_launch", lambda *args: result)
        tiny_config["performance"]["methods"] = ["greedy"]
        with pytest.raises(RuntimeError, match="quality checks failed"):
            benchmark.run_suite(tiny_config, benchmark.required_allocation(tiny_config))
        rows = list(csv.DictReader((suite / "runs.csv").open()))
        assert len(rows) == 1
        assert float(rows[0]["wall_s"]) > 0
        assert rows[0]["status"] == "failed"
        assert json.loads((suite / "quality.json").read_text())[0]["status"] == "failed"
        assert not (suite / "complete.json").exists()

    def test_changed_called_allele_fails(self, tiny, tiny_config, tiny_output):
        path = tiny_output / "samples_raw.trees"
        output = tskit.load(path)
        variant = next(output.variants(isolated_as_missing=False))
        node = int(output.samples()[1])
        actual = variant.alleles[variant.genotypes[1]]
        changed = "T" if actual == "A" else "A"
        tables = output.dump_tables()
        tables.mutations.add_row(site=0, node=node, derived_state=changed)
        tables.sort()
        tables.build_index()
        tables.compute_mutation_parents()
        tables.tree_sequence().dump(path)
        with pytest.raises(ValueError, match="called alleles differ"):
            benchmark.quality_check(tiny_config, "tiny", tiny_output)

    def test_timeout_covers_validation(self, tiny, tiny_config, monkeypatch):
        folder = tiny.root / "delayed"
        folder.mkdir()
        settings = tiny_config["smoke"]["validation_delay"]
        tiny_config["performance"]["trial_timeout_s"] = settings["timeout_s"]
        run = benchmark.trial_runs(tiny_config)[0]
        run.update(method="greedy", af_cutoff=1, min_region_size_kb=0)
        spec = folder / "trial.json"
        spec.write_text(json.dumps({"config": tiny_config, "run": run}))
        original = benchmark.subprocess.Popen

        def launch(command, **kwargs):
            command = [
                command[0],
                "-m",
                "tests.smoke_helpers",
                str(spec),
                str(settings["delay_s"]),
            ]
            return original(command, **kwargs)

        monkeypatch.setattr(benchmark.subprocess, "Popen", launch)
        result = benchmark._launch(spec, tiny_config)
        assert result["status"] == "failed"
        assert "timeout" in result["error"]
        assert result["window"]["wall_s"] > 0
        assert result["window_peak_private_bytes"] > 0
        assert result["quality"]["status"] == "failed"
