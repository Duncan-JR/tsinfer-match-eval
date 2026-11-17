"""Replay actual greedy windows: uv run -m lib.profile_regions CONFIG DATASET OUTPUT."""

import argparse
import cProfile
import dataclasses
import json
import pathlib
import tempfile
import time

import numpy as np
import tsinfer
import tskit
import yaml

from . import benchmark, haplotypes, matching, performance, utils


@dataclasses.dataclass
class Request:
    job_index: int
    start: int
    end: int
    initial_state: int | None
    final_state: int | None
    source: str = "greedy"


class RecordingMatcher:
    """Capture padded calls without changing the greedy matching decisions."""

    def __init__(self, matcher):
        self.matcher = matcher
        self.requests = []
        self.job_index = None

    def find_path(self, h, start, end, out, **kwargs):
        initial = kwargs["initial_state"]
        final = kwargs["final_state"]
        self.requests.append(
            Request(
                self.job_index,
                start,
                end,
                None if initial is None else int(initial),
                None if final is None else int(final),
            )
        )
        return self.matcher.find_path(h, start, end, out, **kwargs)


def path_score(model, mapper, h, out, path, request):
    """Check the path's alleles and score its actual transitions and emissions."""
    left, _, parent = path
    start, end = request.start, request.end
    positions = model.positions[start:end]
    segments = np.searchsorted(left[::-1], positions, side="right") - 1
    donors = parent[::-1][segments]
    if request.initial_state is not None:
        assert donors[0] == request.initial_state
    if request.final_state is not None:
        assert donors[-1] == request.final_state
    tree = tskit.Tree(model.panel.ts)
    codes = np.empty(end - start, dtype=np.int8)
    for offset, donor in enumerate(donors):
        site_index = start + offset
        tree.seek(model.positions[site_index])
        site = model.panel.ts.site(site_index)
        mutations = {
            mutation.node: mutation.derived_state for mutation in site.mutations
        }
        node = int(donor)
        while node != tskit.NULL and node not in mutations:
            node = tree.parent(node)
        state = site.ancestral_state if node == tskit.NULL else mutations[node]
        codes[offset] = mapper.lookup(site_index, state)
    np.testing.assert_array_equal(out[start:end], codes)
    switched = np.zeros(end - start, dtype=bool)
    for boundary in left[:-1]:
        offset = np.searchsorted(positions, boundary)
        switched[offset] = True
    params = model.parameters
    rho = params.recombination[start:end]
    mu = params.mismatch[start:end]
    sample_size = params.sample_size
    if sample_size is None:
        sample_size = model.panel.ts.num_samples
    transition = np.where(switched, rho / sample_size, 1 - rho + rho / sample_size)
    if request.initial_state is not None:
        transition[0] = 1
    num_alleles = mapper.num_alleles_array()[start:end]
    observed = h[start:end]
    matched = observed == codes
    emission = np.where(matched, 1 - (num_alleles - 1) * mu, mu)
    emission[observed < 0] = 1
    return float(np.sum(np.log(transition) + np.log(emission)))


def replay(model, mapper, queries, requests):
    """Compare constrained scores; tied paths may differ between calls."""
    reused = matching.make_hmm(model)
    scratch = np.empty(model.panel.num_sites, dtype=np.int8)
    records = []
    for request in requests:
        pins = {
            "initial_state": request.initial_state,
            "final_state": request.final_state,
        }
        scores = []
        for mode in ["reused", "cold"]:
            setup = time.perf_counter()
            hmm = reused if mode == "reused" else matching.make_hmm(model)
            setup_s = time.perf_counter() - setup
            h = queries[request.job_index]
            started = time.perf_counter()
            try:
                path = hmm.find_path(
                    h, request.start, request.end, scratch, preserve_outside=True, **pins
                )
            except Exception as error:
                raise RuntimeError(
                    f"{mode} request={request}, "
                    f"parameters={matching.numerical_settings(model.parameters)}"
                ) from error
            elapsed = time.perf_counter() - started
            score = path_score(model, mapper, h, scratch, path, request)
            scores.append(score)
            record = {
                **dataclasses.asdict(request),
                "mode": mode,
                "window_sites": request.end - request.start,
                "gap_sites": request.end
                - request.start
                - int(request.initial_state is not None)
                - int(request.final_state is not None),
                "call_s": elapsed,
                "hmm_setup_s": setup_s,
                "log_score": score,
                "segments": len(path[0]),
                "path_capacity_bytes": sum(array.base.nbytes for array in path),
            }
            native = getattr(hmm, "regional_profile", None)
            if native is not None:
                record["native"] = native
            records.append(record)
        np.testing.assert_allclose(scores[0], scores[1], rtol=1e-12, atol=1e-9)
    return records


def overlap_probes(model, queries, requests, width):
    """Cut overlapping pinned windows from one actual long gap and its path."""
    request = max(requests, key=lambda value: value.end - value.start)
    if request.end - request.start < 2 * width - 1:
        raise ValueError(
            "No real window is long enough for the configured overlap probe"
        )
    h = queries[request.job_index]
    scratch = np.empty(model.panel.num_sites, dtype=np.int8)
    hmm = matching.make_hmm(model)
    left, _, parent = hmm.find_path(
        h,
        request.start,
        request.end,
        scratch,
        initial_state=request.initial_state,
        final_state=request.final_state,
        preserve_outside=True,
    )
    probes = []
    for start in [request.start, request.start + width - 1]:
        end = start + width
        sites = model.positions[[start, end - 1]]
        segments = np.searchsorted(left[::-1], sites, side="right") - 1
        initial, final = parent[::-1][segments]
        probes.append(
            Request(
                request.job_index,
                start,
                end,
                int(initial),
                int(final),
                source="overlap_probe",
            )
        )
    return probes


def run(config_path, dataset_name, output):
    config = yaml.safe_load(config_path.read_text())
    dataset = next(
        value for value in config["datasets"] if value["name"] == dataset_name
    )
    paths = benchmark.prepared_inputs(config, dataset)
    cfg = matching._sample_config(paths.config, paths.reference, output / "unused.trees")
    utils.configure_helpers(config["helper_threads"])
    reference = tskit.load(paths.reference)
    jobs = benchmark._sample_jobs(cfg, reference).jobs
    samples = haplotypes.prepare_samples(cfg.sources["samples"], reference, jobs)
    model = matching.prepare_model(
        reference, samples.allele_mapper, cfg.match.sources["samples"]
    )
    minimum = utils.min_region_size_sites(
        reference.sites_position, dataset["min_region_size_kb"]
    )
    output.mkdir(parents=True, exist_ok=True)
    queries = {}
    collector = performance.Collector()
    profiler = cProfile.Profile()
    with tempfile.TemporaryDirectory(dir=config["temp_dir"]) as temporary:
        directory = pathlib.Path(temporary)
        catalog = haplotypes.prepare_reference(
            paths.ancestors,
            samples,
            reference,
            dataset["af_cutoff"],
            config["focal_choice"],
            directory,
        )
        panel = tsinfer.vcz.open_store(paths.ancestors)
        genotypes = panel["call_genotype"]
        packed = catalog.packed.open("r+")
        try:
            for chunk in haplotypes.reference_chunks(paths.ancestors, catalog):
                haplotypes.pack_reference_chunk(genotypes, catalog, chunk, packed)
            packed.flags.writeable = False
            worker = matching.make_worker(model, catalog, packed)
            recorder = RecordingMatcher(worker.matcher)
            worker.matcher = recorder
            limit = config["regional_profile"]["haplotypes"]
            for chunk in samples.chunks:
                slab = haplotypes.load_sample_chunk(samples, chunk, directory)
                calls = slab.calls.open()
                try:
                    for row, job_index in enumerate(slab.job_indices):
                        if len(queries) >= limit:
                            break
                        job_index = int(job_index)
                        h = calls[row].copy()
                        h.flags.writeable = False
                        queries[job_index] = h
                        recorder.job_index = job_index
                        profiler.enable()
                        matching.match_stitched_haplotype(
                            job_index,
                            h,
                            worker,
                            minimum,
                            collector,
                            condition_gaps=config["condition_gaps"],
                        )
                        profiler.disable()
                finally:
                    haplotypes.close_mapping(calls)
                slab.calls.path.unlink()
                if len(queries) >= limit:
                    break
        finally:
            del worker
            haplotypes.close_mapping(packed)
    requests = recorder.requests
    if not requests:
        raise ValueError("Selected haplotypes have no residual HMM windows")
    requests.extend(
        overlap_probes(
            model, queries, requests, config["regional_profile"]["overlap_probe_sites"]
        )
    )
    records = replay(model, samples.allele_mapper, queries, requests)
    profiler.dump_stats(output / "greedy.prof")
    np.savez(
        output / "queries.npz",
        **{f"job_{key}": value for key, value in queries.items()},
        recombination=model.parameters.recombination,
        mismatch=model.parameters.mismatch,
        allele_codebook=samples.allele_mapper.forward_map().astype(str),
    )
    result = {
        "config": config,
        "dataset": dataset_name,
        "packages": {name: utils.package_revision(name) for name in ["tsinfer", "tsls"]},
        "numerical_settings": matching.numerical_settings(model.parameters),
        "requests": [dataclasses.asdict(request) for request in requests],
        "records": records,
        "workload": collector.workload,
        "stages": [dataclasses.asdict(value) for value in collector.records.values()],
    }
    (output / "regional_profile.json").write_text(json.dumps(result, indent=2) + "\n")
    real_count = sum(request.source == "greedy" for request in requests)
    probe_count = len(requests) - real_count
    print(
        f"Validated {real_count} real windows and {probe_count} overlap probes "
        f"from {len(queries)} haplotypes"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=pathlib.Path)
    parser.add_argument("dataset")
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    run(args.config, args.dataset, args.output)


if __name__ == "__main__":
    main()
