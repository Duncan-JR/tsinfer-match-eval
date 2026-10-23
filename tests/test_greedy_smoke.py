"""Focused checks for plans/further_greedy_optimisations.md."""

import concurrent.futures as cf
import contextlib
import copy
import csv
import dataclasses
import json
import multiprocessing as mp

import numpy as np
import pytest
import tsinfer
import tskit
import zarr

from lib import benchmark, diagnostics, haplotypes, matching, performance, utils
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


class TestDiagnosticTrialSources:
    def test_resolves_selected_specs(self, smoke_config, saved_suite):
        products = diagnostics.inventory(smoke_config)
        assert diagnostics.matching_inputs(
            smoke_config, products
        ) == benchmark.final_products(smoke_config)
        diagnostics.resolve_benchmark_outputs(smoke_config, products)
        assert len(products.full_benchmark_runs) == 2
        for run in products.runs:
            saved = run.benchmark_run
            assert saved["requested_workers"] == 2
            assert saved["repeat"] == 0
            assert saved["af_cutoff"] == run.af_cutoff
            assert saved["min_region_size_kb"] == run.min_region_size_kb
            assert run.stitched == saved_suite / saved["run_id"] / "samples_raw.trees"

    @pytest.mark.parametrize(
        "field,value", [("workers", 3), ("repeat", 1), ("mode", "resident_match")]
    )
    def test_unavailable_selector(self, smoke_config, field, value):
        smoke_config["performance"]["diagnostics_run"][field] = value
        with pytest.raises(ValueError, match="unavailable trial"):
            benchmark.diagnostic_trials(smoke_config)

    @pytest.mark.parametrize("missing", ["method", "dataset"])
    def test_incomplete_coverage(self, smoke_config, missing):
        if missing == "method":
            smoke_config["performance"]["methods"] = ["greedy"]
        else:
            smoke_config["performance"]["datasets"].pop()
        benchmark.validate_config(
            smoke_config, benchmark.required_allocation(smoke_config)
        )
        with pytest.raises(ValueError, match="coverage is incomplete"):
            diagnostics.matching_inputs(
                smoke_config, diagnostics.inventory(smoke_config)
            )

    def test_absent_af_trial(self, smoke_config, saved_suite):
        rows = list(csv.DictReader((saved_suite / "runs.csv").open()))
        rows = [row for row in rows if row["af_cutoff"] != "0.05"]
        benchmark._write_csv(saved_suite / "runs.csv", rows)
        with pytest.raises(ValueError, match="lacks required diagnostic trials"):
            benchmark.diagnostic_outputs(smoke_config)

    def test_deleted_selected_tree(self, smoke_config, saved_suite):
        outputs = benchmark.diagnostic_outputs(smoke_config)
        next(iter(outputs.values())).tree.unlink()
        with pytest.raises(FileNotFoundError, match="Selected diagnostic trial tree"):
            benchmark.diagnostic_outputs(smoke_config)

    def test_ordinary_dependencies(self, smoke_config):
        smoke_config["performance"] = None
        products = diagnostics.inventory(smoke_config)
        paths = diagnostics.matching_inputs(smoke_config, products)
        assert paths == products.full + [run.stitched for run in products.runs]
        diagnostics.resolve_benchmark_outputs(smoke_config, products)
        assert products.full_benchmark_runs == {}


class TestDerivedCounts:
    def test_physical_tiles_match_dense(self, tiny, tmp_path, monkeypatch):
        source = tsinfer.vcz.open_store(tiny.source)
        original_calls = source["call_genotype"][:]
        block_type = type(source["call_genotype"].blocks)
        original = block_type.__getitem__
        reads = []

        def read(indexer, key):
            if indexer.array.path == "call_genotype":
                reads.append(key)
            return original(indexer, key)

        monkeypatch.setattr(block_type, "__getitem__", read)
        output = tmp_path / "annotated.zarr"
        selection = ",".join(f"s{column}" for column in tiny.columns)
        utils.annotate_derived_counts(
            tiny.source, output, {"field": "variant_ancestral_state"}, selection
        )
        result = tsinfer.vcz.open_store(output)
        calls = original_calls[:, tiny.columns, :]
        alleles = source["variant_allele"][:]
        ancestral = source["variant_ancestral_state"][:]
        polarity = ancestral == alleles[:, 0]
        expected = np.where(polarity[:, None, None], calls == 1, calls == 0).sum(
            axis=(1, 2)
        )
        called = (calls >= 0).sum(axis=(1, 2))
        frequencies = np.zeros(len(expected))
        np.divide(expected, called, out=frequencies, where=called > 0)
        np.testing.assert_array_equal(
            result["variant_match_eval_derived_ac"][:], expected
        )
        np.testing.assert_array_equal(
            result["variant_match_eval_derived_af"][:], frequencies
        )
        np.testing.assert_array_equal(source["call_genotype"][:], original_calls)
        for name in source.array_keys():
            if not name.startswith("variant_match_eval"):
                np.testing.assert_array_equal(source[name][:], result[name][:])
        expected_reads = [
            (variant, sample, slice(None))
            for variant in range(4)
            for sample in (0, 2, 3)
        ]
        assert reads == expected_reads


class TestReferencePacking:
    @pytest.mark.parametrize("selection", ["all", "sparse", "empty"])
    def test_frozen_bytes_and_tile_reads(self, tiny, tmp_path, monkeypatch, selection):
        reference, _, samples = smoke_helpers.prepared(tiny, tmp_path)
        store = zarr.open_group(tiny.source, mode="a", use_consolidated=False)
        frequencies = np.full(reference.num_sites, 0.9)
        if selection != "empty":
            frequencies[[1, 7]] = 0.1
        if selection == "all":
            frequencies[2] = 0.1
        store["variant_match_eval_derived_af"][:] = frequencies
        catalog = haplotypes.prepare_reference(
            tiny.panel, samples, reference, 0.5, "left", tmp_path
        )
        snapshot = {
            field.name: getattr(catalog, field.name).copy()
            for field in dataclasses.fields(catalog)
            if isinstance(getattr(catalog, field.name), np.ndarray)
        }
        frozen_calls = haplotypes.MappedCalls(
            tmp_path / "frozen.bin", catalog.packed.shape
        )
        calls = frozen_calls.open("w+")
        haplotypes.close_mapping(calls)
        frozen = dataclasses.replace(catalog, packed=frozen_calls)
        chunks = haplotypes.reference_chunks(tiny.panel, catalog)
        expected_blocks = sum(
            smoke_helpers.frozen_pack_reference(tiny.panel, frozen, chunk)
            for chunk in chunks
        )
        array = tsinfer.vcz.open_store(tiny.panel)["call_genotype"]
        block_type = type(array.blocks)
        original = block_type.__getitem__
        reads = []
        searches = []
        search = np.searchsorted

        def read(indexer, key):
            if indexer.array.path == "call_genotype":
                reads.append(key)
            return original(indexer, key)

        def searchsorted(columns, value, *args, **kwargs):
            searches.append(value)
            assert columns is catalog.columns
            return search(columns, value, *args, **kwargs)

        monkeypatch.setattr(block_type, "__getitem__", read)
        monkeypatch.setattr(haplotypes.np, "searchsorted", searchsorted)
        offsets = smoke_helpers.TrackedOffsets(catalog.offsets)
        catalog.offsets = offsets
        actual_blocks = sum(
            haplotypes.pack_reference_chunk(tiny.panel, catalog, chunk)
            for chunk in chunks
        )
        assert actual_blocks == expected_blocks == len(reads)
        assert len(searches) == 2 * len(chunks)
        assert len(set(reads)) == len(reads)
        height = array.chunks[0]
        overlaps = 0
        for start, stop in zip(catalog.starts, catalog.stops, strict=True):
            overlaps += (stop - 1) // height - start // height + 1
        assert offsets.visits == overlaps + len(catalog.seeds)
        actual = catalog.packed.open()
        expected = frozen.packed.open()
        np.testing.assert_array_equal(actual, expected)
        haplotypes.close_mapping(actual)
        haplotypes.close_mapping(expected)
        for name, values in snapshot.items():
            np.testing.assert_array_equal(getattr(catalog, name), values)

    def test_invalid_derived_seed(self, tiny, tmp_path):
        reference, _, samples = smoke_helpers.prepared(tiny, tmp_path)
        catalog = haplotypes.prepare_reference(
            tiny.panel, samples, reference, 1, "left", tmp_path
        )
        panel = zarr.open_group(tiny.panel, mode="a", use_consolidated=False)
        panel["call_genotype"][catalog.seeds[0], catalog.columns[0], 0] = 0
        with pytest.raises(ValueError, match="not called and derived"):
            haplotypes.pack_reference_chunk(tiny.panel, catalog, 0)

    def test_one_and_two_packing_workers(self, tiny, tmp_path):
        reference, _, samples = smoke_helpers.prepared(tiny, tmp_path)
        snapshots = []
        for workers in (1, 2):
            directory = tmp_path / f"w{workers}"
            directory.mkdir()
            catalog = haplotypes.prepare_reference(
                tiny.panel, samples, reference, 1, "left", directory
            )
            chunks = haplotypes.reference_chunks(tiny.panel, catalog)
            tasks = [(tiny.panel, catalog, chunk) for chunk in chunks]
            with cf.ProcessPoolExecutor(
                max_workers=workers, mp_context=mp.get_context("spawn")
            ) as pool:
                list(pool.map(smoke_helpers.pack_job, tasks))
            calls = catalog.packed.open()
            snapshots.append(calls.copy())
            haplotypes.close_mapping(calls)
        np.testing.assert_array_equal(*snapshots)


class TestMatchingLifetime:
    @pytest.mark.parametrize("workers", [1, 2])
    def test_workers_exit_before_finalise(
        self, tiny, tiny_config, lifetime_records, monkeypatch, workers
    ):
        original = tsinfer.matching.finalise_match_tables
        finalised = []

        def finalise(tables):
            assert len(lifetime_records["processes"]) == workers
            assert all(
                p.exitcode == 0 and not p.is_alive()
                for p in lifetime_records["processes"]
            )
            assert len(lifetime_records["threads"]) == 2
            assert all(not thread.is_alive() for thread in lifetime_records["threads"])
            assert len(lifetime_records["references"]) == 1
            assert all(ref() is None for ref in lifetime_records["references"])
            finalised.append(True)
            return original(tables)

        monkeypatch.setattr(tsinfer.matching, "finalise_match_tables", finalise)
        output = tiny.root / "lifetime.trees"
        matching.match_stitched_samples(
            tiny.config,
            tiny.reference,
            output,
            af_cutoff=1,
            min_region_size_kb=0,
            workers=workers,
            sample_prefetch_chunks=1,
            max_pending_haplotypes=2,
            helper_threads=tiny_config["helper_threads"],
            temp_dir=tiny.root,
        )
        assert finalised == [True]
        assert not list(tiny.root.glob("greedy_*"))

    @pytest.mark.parametrize("failure", ["seed", "finalise"])
    def test_failure_cleans_workers_and_slabs(
        self, tiny, tiny_config, lifetime_records, monkeypatch, failure
    ):
        if failure == "seed":
            panel = zarr.open_group(tiny.panel, mode="a", use_consolidated=False)
            panel["call_genotype"][1, 0, 0] = 0
            message = "not called and derived"
        else:

            def fail(tables):
                assert all(not p.is_alive() for p in lifetime_records["processes"])
                raise RuntimeError("finaliser failure")

            monkeypatch.setattr(tsinfer.matching, "finalise_match_tables", fail)
            message = "finaliser failure"
        with pytest.raises(RuntimeError, match=message):
            matching.match_stitched_samples(
                tiny.config,
                tiny.reference,
                tiny.root / "failed.trees",
                af_cutoff=1,
                min_region_size_kb=0,
                workers=2,
                sample_prefetch_chunks=1,
                max_pending_haplotypes=2,
                helper_threads=tiny_config["helper_threads"],
                temp_dir=tiny.root,
            )
        assert all(not p.is_alive() for p in lifetime_records["processes"])
        assert all(not thread.is_alive() for thread in lifetime_records["threads"])
        assert not list(tiny.root.glob("greedy_*"))

    def test_resident_window_ends_before_shutdown(
        self, tiny, tiny_config, lifetime_records
    ):
        boundaries = []

        @contextlib.contextmanager
        def window():
            assert all(p.is_alive() for p in lifetime_records["processes"])
            assert list(tiny.root.glob("greedy_*/samples_*.bin"))
            boundaries.append("start")
            yield
            assert all(p.is_alive() for p in lifetime_records["processes"])
            assert not list(tiny.root.glob("greedy_*/samples_*.bin"))
            boundaries.append("end")

        matching.match_stitched_samples(
            tiny.config,
            tiny.reference,
            tiny.root / "resident.trees",
            af_cutoff=1,
            min_region_size_kb=0,
            workers=2,
            sample_prefetch_chunks=1,
            max_pending_haplotypes=2,
            helper_threads=tiny_config["helper_threads"],
            temp_dir=tiny.root,
            resident=True,
            window=window(),
        )
        assert boundaries == ["start", "end"]
        assert all(not p.is_alive() for p in lifetime_records["processes"])
        assert all(not thread.is_alive() for thread in lifetime_records["threads"])
        assert not list(tiny.root.glob("greedy_*"))


class TestStageTimings:
    @pytest.mark.parametrize("workers", [1, 2])
    def test_stages_preserve_outputs_and_workload(self, tiny, tiny_config, workers):
        collectors = []
        outputs = []
        for enabled in (False, True):
            collector = performance.Collector(stages=enabled)
            output = tiny.root / f"stages_{enabled}.trees"
            matching.match_stitched_samples(
                tiny.config,
                tiny.reference,
                output,
                af_cutoff=1,
                min_region_size_kb=0,
                workers=workers,
                sample_prefetch_chunks=1,
                max_pending_haplotypes=2,
                helper_threads=tiny_config["helper_threads"],
                temp_dir=tiny.root,
                collector=collector,
            )
            collectors.append(collector)
            outputs.append(smoke_helpers.canonical(output))
        assert outputs[0] == outputs[1]
        assert collectors[0].workload == collectors[1].workload
        assert collectors[0].records == {}
        records = {record.stage: record for record in collectors[1].records.values()}
        phases = [
            "coordinator_process_startup",
            "coordinator_packing_completion",
            "coordinator_worker_readiness",
            "coordinator_matching_drain",
            "coordinator_shutdown",
            "coordinator_result_wait",
            "first_sample_slab_latency",
            "first_result_latency",
        ]
        for name in phases:
            record = records[name]
            assert record.scope == "coordinator"
            assert np.isfinite(record.wall_s) and record.wall_s >= 0
            assert record.user_s is None and record.thread_cpu_s is None
        slab = records["first_sample_slab_latency"].wall_s
        result = records["first_result_latency"].wall_s
        drain = records["coordinator_matching_drain"].wall_s
        assert 0 <= slab <= result <= drain
        assert records["coordinator_result_wait"].wall_s <= drain
        assert records["assembly_output"].wall_s <= drain
        native = records["gap_native_find_path"]
        assert native.call_count == collectors[1].workload["gap_calls"]
        assert records["gap_path_conversion"].call_count == native.call_count
        assert records["gap_mutation_extraction"].call_count == native.call_count
        assert (
            records["gap_final_concatenation"].call_count
            == collectors[1].workload["matched_haplotypes"]
        )
        assert records["gap_matching_assembly"].inclusive
        for name in (
            "gap_native_find_path",
            "gap_path_conversion",
            "gap_mutation_extraction",
            "gap_final_concatenation",
        ):
            record = records[name]
            assert record.parent_stage == "gap_matching_assembly"
            assert record.scope == "thread"
            assert record.service_s >= 0 and record.thread_cpu_s >= 0
            assert record.wall_s is None

    @pytest.mark.parametrize("kind", ["all_focal", "all_gap", "one_site_gap"])
    def test_native_counts_and_shortcuts(self, tiny, tiny_config, tmp_path, kind):
        reference, _, samples = smoke_helpers.prepared(tiny, tmp_path)
        catalog = haplotypes.prepare_reference(
            tiny.panel, samples, reference, 1, "left", tmp_path
        )
        for chunk in haplotypes.reference_chunks(tiny.panel, catalog):
            haplotypes.pack_reference_chunk(tiny.panel, catalog, chunk)
        if kind == "all_gap":
            empty = haplotypes.MappedCalls(tmp_path / "empty.bin", (0,))
            fields = {
                field.name: getattr(catalog, field.name)[:0]
                for field in dataclasses.fields(catalog)
                if isinstance(getattr(catalog, field.name), np.ndarray)
            }
            catalog = dataclasses.replace(catalog, packed=empty, **fields)
        cfg = matching._sample_config(
            tiny.config, tiny.reference, tiny.root / "unused.trees"
        )
        worker = matching.make_worker(
            reference,
            catalog,
            samples.allele_mapper.num_alleles_array(),
            cfg.match.sources["samples"],
        )
        native = smoke_helpers.CountingMatcher(worker.matcher)
        worker.matcher = native
        calls = tsinfer.vcz.open_store(tiny.panel)["call_genotype"][:]
        haplotype = np.concatenate([calls[:6, 1, 0], calls[6:, 2, 0]])
        if kind == "one_site_gap":
            haplotype[0] = 1 - haplotype[0]
        collector = performance.Collector()
        try:
            matching.match_stitched_haplotype(0, haplotype, worker, 1, collector)
        finally:
            haplotypes.close_mapping(worker.exact.packed)
        counts = collector.workload
        assert counts["gap_calls"] == native.calls
        recorded = collector.records.get(("gap_native_find_path", "thread"))
        assert (0 if recorded is None else recorded.call_count) == native.calls
        if kind == "all_gap":
            assert counts["gap_sites"] == reference.num_sites
            assert counts["focal_sites"] == 0
            assert native.calls == 2
        else:
            assert counts["focal_sites"] == reference.num_sites
            assert native.calls == 0
            if kind == "one_site_gap":
                assert counts["one_site_gap_sites"] == 1
