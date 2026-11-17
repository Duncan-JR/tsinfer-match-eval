"""Native baseline and streamed, complete-haplotype greedy stitched matching."""

import collections
import concurrent.futures as cf
import contextlib
import dataclasses
import json
import logging
import pathlib
import queue
import tempfile
import threading
import time
import traceback

import numpy as np
import tsinfer
import tskit
import tsls

from . import greedy, haplotypes, performance, utils

logger = logging.getLogger(__name__)


def match_ancestors(
    config_path: pathlib.Path,
    output_path: pathlib.Path,
    threads: int,
    cache_size: int | None = None,
) -> None:
    """Match ancestors before samples, preserving the raw reference TS.

    Ancestor times must put every ancestor group strictly before the first
    sample group. Sample matching appends its own individual and population
    metadata after the reference rows, following tsinfer's reference API.
    """
    cfg = tsinfer.config.Config.from_toml(config_path)
    jobs = json.loads(tsinfer.pipeline.compute_groups_json(cfg))
    sample_groups = [job["group"] for job in jobs if job["source"] == "samples"]
    first_sample_group = min(sample_groups)
    ancestor_groups = [job["group"] for job in jobs if job["source"] == "ancestors"]
    if any(group >= first_sample_group for group in ancestor_groups):
        raise ValueError("All ancestor groups must precede the first sample group")
    ts = tsinfer.pipeline.match(
        cfg,
        group_stop=first_sample_group,
        num_threads=threads,
        cache_size=cache_size,
        read_workers=1,
    )
    ts.dump(output_path)
    logger.info("Wrote raw ancestor reference to %s", output_path)


def _sample_config(config_path, ancestors_ts_path, output_path):
    """Preserve native sample selection and metadata for both matching runners."""
    cfg = tsinfer.config.Config.from_toml(config_path)
    sample_match = dataclasses.replace(
        cfg.match,
        sources={"samples": cfg.match.sources["samples"]},
        reference_ts=ancestors_ts_path,
        output=output_path,
        path_compression=False,
    )
    return dataclasses.replace(
        cfg,
        sources={"samples": cfg.sources["samples"]},
        ancestors=[],
        match=sample_match,
        post_process=None,
        augment_sites=None,
    )


def match_samples(
    config_path: pathlib.Path,
    ancestors_ts_path: pathlib.Path,
    output_path: pathlib.Path,
    threads: int,
    *,
    sample_reader_threads: int = 1,
    sample_prefetch_chunks: int = 1,
    max_pending_haplotypes: int | None = None,
    helper_threads: dict | None = None,
    temp_dir: pathlib.Path | None = None,
) -> int:
    """Stream full-HMM jobs through the shared-panel worker runner."""
    return match_stitched_samples(
        config_path,
        ancestors_ts_path,
        output_path,
        af_cutoff=0,
        min_region_size_kb=0,
        workers=threads,
        sample_prefetch_chunks=sample_prefetch_chunks,
        max_pending_haplotypes=max_pending_haplotypes,
        helper_threads=helper_threads,
        sample_reader_threads=sample_reader_threads,
        temp_dir=temp_dir,
        full_hmm=True,
    )


@dataclasses.dataclass
class CopyingRuns:
    """Half-open inference-site runs, independent of storage chunk boundaries."""

    left: np.ndarray
    right: np.ndarray
    parent: np.ndarray


@dataclasses.dataclass
class RegionMatch:
    """Owned regional path and mutation sites before native scratch is reused."""

    runs: CopyingRuns
    mutation_site: np.ndarray


@dataclasses.dataclass
class ActiveSlab:
    """Coordinator-owned descriptor and unappended result count."""

    slab: haplotypes.SampleSlab
    remaining: int


@dataclasses.dataclass
class HaplotypeResult:
    """Owned numeric output, safe to enqueue after worker scratch is reused."""

    job_index: int
    left: np.ndarray
    right: np.ndarray
    parent: np.ndarray
    mutation_site: np.ndarray
    mutation_code: np.ndarray
    measurements: performance.Collector | None = None
    timing: performance.HaplotypeTiming | None = None


@dataclasses.dataclass
class HaplotypeTask:
    """One complete row in a slab; no decoded calls travel through queues."""

    slab: haplotypes.MappedCalls
    row: int
    job_index: int
    chunk: int


@dataclasses.dataclass
class MatchingModel:
    """One immutable panel, codebook and set of rates shared by all workers."""

    panel: tsls.TreeSequence
    parameters: tsinfer.matching.HmmParameters
    positions: np.ndarray
    boundaries: np.ndarray


def prepare_model(reference, allele_mapper, parameters):
    """Keep complete reference IDs and use the sample codebook at every site."""
    tsinfer.matching.check_ultimate_root(reference)
    positions = reference.sites_position.astype(np.int32)
    intervals = np.asarray(reference.metadata["sequence_intervals"])
    boundaries = np.unique(np.searchsorted(positions, intervals.ravel()))
    panel = tsls.TreeSequence(reference, alleles=allele_mapper.forward_map())
    rates = tsinfer.matching.hmm_parameters(parameters, positions)
    return MatchingModel(panel, rates, positions, boundaries)


def numerical_settings(parameters):
    """Resolve the native defaults for recorded benchmark/error context."""
    threshold = parameters.likelihood_threshold
    if threshold is None:
        threshold = np.finfo(float).tiny
    return {
        "likelihood_threshold": threshold,
        "prune_threshold": 0.0
        if parameters.prune_threshold is None
        else parameters.prune_threshold,
        "relative_prune_threshold": 0.0
        if parameters.relative_prune_threshold is None
        else parameters.relative_prune_threshold,
    }


def make_hmm(model):
    """Create private native state without changing numerical defaults."""
    params = model.parameters
    return tsls.Hmm(
        model.panel,
        params.recombination,
        params.mismatch,
        sample_size=params.sample_size,
        likelihood_threshold=params.likelihood_threshold,
        prune_threshold=params.prune_threshold,
        relative_prune_threshold=params.relative_prune_threshold,
    )


@dataclasses.dataclass
class Worker:
    """One HMM and reusable numeric scratch owned exclusively by one thread."""

    exact: greedy.ExactState | None
    positions: np.ndarray
    sequence_length: float
    boundaries: np.ndarray
    matcher: tsls.Hmm
    match_out: np.ndarray
    numerical_settings: dict


def make_worker(model, catalog=None, packed=None):
    """Warm the read-only Numba signatures before workers report readiness."""
    exact = None if catalog is None else greedy.make_state(catalog, packed)
    worker = Worker(
        exact,
        model.positions,
        model.panel.ts.sequence_length,
        model.boundaries,
        make_hmm(model),
        np.empty(len(model.positions), dtype=np.int8),
        numerical_settings(model.parameters),
    )
    if exact is not None:
        dummy = np.full(len(model.positions), -1, dtype=np.int8)
        dummy.flags.writeable = False
        order = greedy.candidates(dummy, exact, 0, len(dummy))
        greedy.cover(exact, order, 1)
        catalog = exact.catalog
        partition = greedy.partition_intervals(exact, 0, 0, 0, worker.boundaries)
        bridges = greedy.resolve_one_site_gaps(exact, partition, dummy)
        intervals = greedy.coalesce_copied(partition)
        residual = greedy.residual_windows(
            intervals, catalog.parents, worker.boundaries, True
        )
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int32)
        greedy.assemble_intervals(
            intervals.left,
            intervals.right,
            intervals.ancestor,
            catalog.parents,
            np.zeros(1, dtype=np.uint64),
            empty_float,
            empty_float,
            empty_int,
            residual.gap_left,
            residual.gap_right,
            bridges.corrections,
            dummy,
            worker.match_out,
            worker.positions,
            worker.sequence_length,
        )
    return worker


def match_full_haplotype(job_index, haplotype, worker, collector=None):
    """Run the full window using the worker's reusable HMM and owned results."""
    called = np.flatnonzero(haplotype >= 0)
    start, end = 0, len(haplotype)
    if len(called) > 0:
        start, end = int(called[0]), int(called[-1]) + 1
    region = _match_region(haplotype, worker, start, end, collector)
    result = _assemble_result(
        job_index, [region.runs], [region.mutation_site], haplotype, worker, collector
    )
    if collector is not None:
        collector.count(
            matched_haplotypes=1,
            full_calls=1,
            hmm_batches=1,
            domain_sites=end - start,
            focal_sites=0,
            gap_sites=end - start,
            gap_calls=1,
        )
    return result


def _match_region(
    haplotype,
    worker,
    start,
    end,
    collector=None,
    *,
    initial_state=None,
    final_state=None,
):
    """Collect a regional HMM path and mutations before native scratch is reused."""
    match_start = start
    match_end = end
    if initial_state is not None:
        match_start = start - 1
    if final_state is not None:
        match_end = end + 1
    with performance.measure(
        collector, "gap_native_find_path", "thread", parent_stage="gap_matching_assembly"
    ):
        try:
            left, right, parents = worker.matcher.find_path(
                haplotype,
                match_start,
                match_end,
                worker.match_out,
                initial_state=initial_state,
                final_state=final_state,
                preserve_outside=True,
            )
        except tsls.MatchImpossible as error:
            raise tsls.MatchImpossible(
                f"window=[{match_start}, {match_end}), "
                f"initial_state={initial_state}, final_state={final_state}, "
                f"parameters={worker.numerical_settings}: {error}"
            ) from error
        left, right, parents = tsinfer.matching.copy_node1_for_node0(
            left, right, parents
        )
    with performance.measure(
        collector, "gap_path_conversion", "thread", parent_stage="gap_matching_assembly"
    ):
        left_site = np.searchsorted(worker.positions, left[::-1])
        right_site = np.searchsorted(worker.positions, right[::-1])
        np.maximum(left_site, start, out=left_site)
        np.minimum(right_site, end, out=right_site)
        keep = left_site < right_site
        runs = CopyingRuns(left_site[keep], right_site[keep], parents[::-1][keep])
    with performance.measure(
        collector,
        "gap_mutation_extraction",
        "thread",
        parent_stage="gap_matching_assembly",
    ):
        observed = haplotype[start:end]
        copied = worker.match_out[start:end]
        mutations = np.flatnonzero((observed >= 0) & (observed != copied)) + start
    return RegionMatch(runs, mutations)


def _assemble_result(job_index, blocks, mutation_blocks, haplotype, worker, collector):
    """Concatenate owned segments and apply the existing site-to-BP conventions."""
    left = np.concatenate([block.left for block in blocks])
    right = np.concatenate([block.right for block in blocks])
    parents = np.concatenate([block.parent for block in blocks])
    bp_left = worker.positions[left].astype(float)
    bp_left[left == 0] = 0
    bp_right = np.full(len(right), worker.sequence_length)
    internal = right < len(worker.positions)
    bp_right[internal] = worker.positions[right[internal]]
    sites = np.empty(0, dtype=np.int64)
    if len(mutation_blocks) > 0:
        sites = np.concatenate(mutation_blocks)
    return HaplotypeResult(
        job_index,
        bp_left,
        bp_right,
        parents,
        sites,
        haplotype[sites],
        collector,
    )


def match_stitched_haplotype(
    job_index, haplotype, worker, minimum, collector=None, *, condition_gaps=True
):
    """Match one complete canonical haplotype against a prepared reference.

    Filter exact components by size, then copy supported one-site gaps.
    Correct called bridge disagreements
    with sample mutations. Optionally condition residual HMM gaps on adjacent
    copied parents; disabling this also removes flanking observation padding.
    """
    called = np.flatnonzero(haplotype >= 0)
    start, end = 0, len(haplotype)
    if len(called) > 0:
        start, end = int(called[0]), int(called[-1]) + 1
    state = worker.exact
    with performance.measure(collector, "candidate_extension_cover", "thread"):
        order = greedy.candidates(haplotype, state, start, end)
        count = greedy.cover(state, order, minimum)
    catalog = state.catalog
    with performance.measure(
        collector, "gap_matching_assembly", "thread", inclusive=True
    ):
        with performance.measure(
            collector,
            "gap_interval_preparation",
            "thread",
            parent_stage="gap_matching_assembly",
        ):
            partition = greedy.partition_intervals(
                state, count, start, end, worker.boundaries
            )
            bridges = greedy.resolve_one_site_gaps(state, partition, haplotype)
            intervals = greedy.coalesce_copied(partition)
            residual = greedy.residual_windows(
                intervals, catalog.parents, worker.boundaries, condition_gaps
            )
        if len(residual.windows) > 0:
            with performance.measure(
                collector,
                "gap_native_find_path",
                "thread",
                parent_stage="gap_matching_assembly",
            ):
                offsets, path_left, path_right, path_parent = worker.matcher.find_paths(
                    haplotype, residual.windows, worker.match_out
                )
        else:
            offsets = np.zeros(1, dtype=np.uint64)
            path_left = np.empty(0, dtype=np.float64)
            path_right = path_left
            path_parent = np.empty(0, dtype=np.int32)
        with performance.measure(
            collector,
            "gap_final_concatenation",
            "thread",
            parent_stage="gap_matching_assembly",
        ):
            result_arrays = greedy.assemble_intervals(
                intervals.left,
                intervals.right,
                intervals.ancestor,
                catalog.parents,
                offsets,
                path_left,
                path_right,
                path_parent,
                residual.gap_left,
                residual.gap_right,
                bridges.corrections,
                haplotype,
                worker.match_out,
                worker.positions,
                worker.sequence_length,
            )
            result = HaplotypeResult(job_index, *result_arrays, collector)
    if collector is not None:
        collector.count(
            matched_haplotypes=1,
            domain_sites=end - start,
            focal_sites=residual.focal_sites,
            gap_sites=end - start - residual.focal_sites,
            focal_runs=residual.focal_runs,
            gap_calls=len(residual.windows),
            hmm_batches=int(len(residual.windows) > 0),
            one_site_gap_sites=bridges.sites,
        )
    return result


def _worker_main(
    work,
    events,
    start_matching,
    stopping,
    worker_index,
    panel_genotypes,
    model,
    catalog,
    packed,
    minimum,
    collect,
    diagnostics,
    collect_haplotype_timings,
    condition_gaps,
    full_hmm,
):
    """Persistent packing and matching loop with a readiness barrier."""
    try:
        while not stopping.is_set():
            chunk = work.get()
            if chunk is None:
                break
            collector = performance.Collector(stages=diagnostics)
            with performance.measure(
                collector,
                "reference_packing",
                "thread",
                parent_stage="coordinator_packing_completion",
            ):
                count = haplotypes.pack_reference_chunk(
                    panel_genotypes, catalog, chunk, packed
                )
            collector.count(ancestor_blocks_read=count)
            events.put(("packed", collector))
        if stopping.is_set():
            return
        collector = performance.Collector(stages=diagnostics)
        with performance.measure(
            collector,
            "worker_setup",
            "thread",
            parent_stage="coordinator_worker_readiness",
        ):
            worker = make_worker(model, catalog, packed)
        events.put(("worker_ready", collector))
        start_matching.wait()
        while not stopping.is_set():
            task = work.get()
            if task is None:
                break
            calls = task.slab.open()
            try:
                row = calls[task.row]
                collector = (
                    performance.Collector(stages=diagnostics) if collect else None
                )
                if collect_haplotype_timings:
                    match_start = time.perf_counter()
                    cpu_start = time.thread_time()
                if full_hmm:
                    result = match_full_haplotype(task.job_index, row, worker, collector)
                else:
                    result = match_stitched_haplotype(
                        task.job_index,
                        row,
                        worker,
                        minimum,
                        collector,
                        condition_gaps=condition_gaps,
                    )
                if collect_haplotype_timings:
                    match_end = time.perf_counter()
                    counts = collector.workload
                    result.timing = performance.HaplotypeTiming(
                        worker_index,
                        match_start,
                        match_end,
                        match_end - match_start,
                        time.thread_time() - cpu_start,
                        counts["domain_sites"],
                        counts["focal_sites"],
                        counts["gap_sites"],
                        counts["gap_calls"],
                    )
                if collector is not None:
                    collector.diagnostics["worker_index"] = worker_index
                    collector.diagnostics["traceback_allocator_bytes"] = (
                        worker.matcher.total_memory
                    )
                del row
            finally:
                haplotypes.close_mapping(calls)
            events.put(("result", (task.chunk, result)))
    except Exception:
        events.put(("error", traceback.format_exc()))


def _event(events):
    kind, payload = events.get()
    if kind == "error":
        raise RuntimeError(payload)
    return kind, payload


def _pack_reference(work, events, chunks, cap, collector):
    """Bound submitted-but-uncollected preparation tasks with one admission count."""
    chunks = iter(chunks)
    pending = 0
    exhausted = False
    while not exhausted or pending > 0:
        while not exhausted and pending < cap:
            chunk = next(chunks, None)
            if chunk is None:
                exhausted = True
            else:
                work.put(chunk)
                pending += 1
        if pending > 0:
            _, measurements = _event(events)
            if collector is not None:
                collector.merge(measurements)
            pending -= 1


def _load_samples(inputs, directory, slots, stopping, events, sample_reader_threads):
    """Publish complete slabs; a slot covers loading, waiting and worker use."""
    try:
        for chunk in inputs.chunks:
            slots.acquire()
            if stopping.is_set():
                return
            slab = haplotypes.load_sample_chunk(
                inputs, chunk, directory, sample_reader_threads
            )
            if stopping.is_set():
                return
            events.put(("slab", slab))
    except Exception:
        events.put(("error", traceback.format_exc()))


def _create_sample_nodes(tables, jobs):
    """Reserve canonical child IDs before appending unordered completed results."""
    nodes = []
    for job in jobs:
        metadata = {
            "source": job.source,
            "sample_id": job.sample_id,
            "ploidy_index": job.ploidy_index,
        }
        node = tables.nodes.add_row(
            time=job.time,
            flags=job.node_flags,
            metadata=metadata,
            individual=-1 if job.individual_id is None else job.individual_id,
            population=-1 if job.population_id is None else job.population_id,
        )
        nodes.append(node)
    return nodes


def append_result(tables, child, result, allele_mapper):
    """Narrow local append contract for pre-created nodes and numeric results."""
    tables.edges.append_columns(
        left=result.left,
        right=result.right,
        parent=result.parent,
        child=np.full(len(result.left), child, dtype=np.int32),
    )
    states = allele_mapper.decode_mutations(result.mutation_site, result.mutation_code)
    for site, state in zip(result.mutation_site, states, strict=True):
        tables.mutations.add_row(site=int(site), node=child, derived_state=state)


def _coordinate(
    work,
    events,
    jobs,
    tables,
    nodes,
    samples,
    cap,
    slots,
    collector,
    resident_slabs,
    admitted,
):
    """Drain events and refill continuously; pending includes unappended results."""
    ready = collections.deque()
    live = {}
    next_row = 0
    pending = 0
    completed = 0
    first_slab = True
    first_result = True
    if resident_slabs is not None:
        if admitted is not None:
            collector.elapsed(
                "first_sample_slab_latency",
                0,
                parent_stage="coordinator_matching_drain",
                inclusive=True,
            )
        first_slab = False
        ready.extend(resident_slabs)
        live.update(
            {
                slab.chunk: ActiveSlab(slab, len(slab.job_indices))
                for slab in resident_slabs
            }
        )
    while completed < len(jobs):
        while pending < cap and len(ready) > 0:
            slab = ready[0]
            index = int(slab.job_indices[next_row])
            work.put(HaplotypeTask(slab.calls, next_row, index, slab.chunk))
            pending += 1
            next_row += 1
            if next_row == len(slab.job_indices):
                ready.popleft()
                next_row = 0
        with performance.measure(
            collector,
            "coordinator_result_wait",
            "coordinator",
            parent_stage="coordinator_matching_drain",
        ):
            kind, payload = _event(events)
        if kind == "slab":
            if first_slab and admitted is not None:
                collector.elapsed(
                    "first_sample_slab_latency",
                    time.perf_counter() - admitted,
                    parent_stage="coordinator_matching_drain",
                    inclusive=True,
                )
            first_slab = False
            slab = payload
            ready.append(slab)
            live[slab.chunk] = ActiveSlab(slab, len(slab.job_indices))
            if collector is not None:
                collector.count(sample_blocks_read=slab.blocks_read)
                if collector.stages:
                    collector.add(
                        performance.Measurement(
                            "sample_loading_remapping",
                            "loader",
                            call_count=1,
                            service_s=slab.load_s,
                        )
                    )
        elif kind == "result":
            if first_result and admitted is not None:
                collector.elapsed(
                    "first_result_latency",
                    time.perf_counter() - admitted,
                    parent_stage="coordinator_matching_drain",
                    inclusive=True,
                )
            first_result = False
            chunk, result = payload
            with performance.measure(
                collector,
                "assembly_output",
                "thread",
                parent_stage="coordinator_matching_drain",
            ):
                append_result(
                    tables, nodes[result.job_index], result, samples.allele_mapper
                )
            if result.timing is not None:
                job = jobs[result.job_index]
                collector.haplotype_timings.append(
                    {
                        "job_index": result.job_index,
                        "source": job.source,
                        "sample_id": job.sample_id,
                        "ploidy_index": job.ploidy_index,
                        **dataclasses.asdict(result.timing),
                    }
                )
            if collector is not None and result.measurements is not None:
                details = result.measurements.diagnostics
                worker_index = details["worker_index"]
                worker_details = collector.diagnostics.setdefault(
                    "traceback_allocator_bytes_by_worker", {}
                )
                previous = worker_details.get(worker_index, 0)
                worker_details[worker_index] = max(
                    previous, details["traceback_allocator_bytes"]
                )
                collector.merge(result.measurements)
            pending -= 1
            completed += 1
            live[chunk].remaining -= 1
            if live[chunk].remaining == 0:
                active = live.pop(chunk)
                active.slab.calls.path.unlink()
                if resident_slabs is None:
                    slots.release()
            del result


def match_stitched_samples(
    config_path: pathlib.Path,
    ancestors_ts_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    af_cutoff: float,
    min_region_size_kb: float,
    workers: int,
    sample_prefetch_chunks: int,
    max_pending_haplotypes: int | None,
    helper_threads: dict | None,
    focal_choice: str = "left",
    condition_gaps: bool = True,
    temp_dir: pathlib.Path | None = None,
    collector: performance.Collector | None = None,
    sample_reader_threads: int = 1,
    collect_haplotype_timings: bool = False,
    resident: bool = False,
    window: performance.Window | None = None,
    full_hmm: bool = False,
) -> int:
    """Share one immutable tsls panel and stream bounded whole-haplotype jobs.

    One persistent thread pool handles packing and matching; a loader owns
    bounded sample Zarr reader threads. Outstanding
    jobs and live decoded slabs have separate bounds. Temporary inputs are
    removed only after the loader and workers stop. Returns the actual worker count.

    Diagnostic resident runs preload all slabs and initialise workers before
    ``window``; the benchmark limits their total size. Normal end-to-end timing
    wraps this entire function, including startup, packing and final output.
    """
    if collect_haplotype_timings and collector is None:
        raise ValueError("Haplotype timing collection requires a collector")
    if max_pending_haplotypes is None:
        max_pending_haplotypes = workers
    if max_pending_haplotypes < workers:
        raise ValueError(
            "max_pending_haplotypes must be at least the worker count; "
            "use null to derive it from workers"
        )
    native = tsinfer.config.Config.from_toml(config_path)
    panel_path = native.ancestors[0].path
    del native
    cfg = _sample_config(config_path, ancestors_ts_path, output_path)
    if helper_threads is not None:
        utils.configure_helpers(helper_threads)
    with performance.measure(collector, "input_setup"):
        reference = tskit.load(ancestors_ts_path)
        positions = reference.sites_position.astype(np.int32)
        jobs, individuals, populations = tsinfer.pipeline._build_jobs_with_metadata(
            cfg, positions, reference
        )
        jobs.sort(key=lambda job: job.haplotype_index)
        if len(jobs) == 0 or reference.num_sites == 0:
            raise ValueError("Matching requires samples and reference sites")
        min_focal_region_size = utils.min_region_size_sites(
            reference.sites_position, min_region_size_kb
        )
        logger.info(
            "Minimum connected focal region: %s kb resolves to %d inference sites",
            min_region_size_kb,
            min_focal_region_size,
        )
        samples = haplotypes.prepare_samples(cfg.sources["samples"], reference, jobs)
        prepared = tsinfer.matching.prepare_reference_ts(
            reference,
            individuals=individuals,
            populations=populations if len(populations) > 0 else None,
        )
        tables = prepared.dump_tables()
        del prepared
        metadata = tables.metadata
        metadata["match_eval"] = {
            "af_cutoff": af_cutoff,
            "min_region_size_kb": min_region_size_kb,
            "min_focal_region_size": min_focal_region_size,
            "focal_choice": focal_choice,
            "condition_gaps": condition_gaps,
        }
        tables.metadata = metadata
        if collector is not None:
            collector.count(min_focal_region_size=min_focal_region_size)
        nodes = _create_sample_nodes(tables, jobs)
        del positions, individuals, populations
    if workers > len(jobs):
        raise ValueError(f"Requested {workers} workers for only {len(jobs)} haplotypes")
    work = queue.Queue(maxsize=max_pending_haplotypes)
    events = queue.Queue()
    start_matching = threading.Event()
    stopping = threading.Event()
    slots = threading.Semaphore(sample_prefetch_chunks)
    futures = []
    executor = None
    packed = None
    model = None
    catalog = None
    panel_genotypes = None
    loader = None
    diagnostics = collector is not None and collector.stages
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_dir, prefix="greedy_") as temporary:
        directory = pathlib.Path(temporary)
        try:
            with performance.measure(collector, "reference_catalog"):
                model = prepare_model(
                    reference, samples.allele_mapper, cfg.match.sources["samples"]
                )
                catalog = None
                chunks = []
                if not full_hmm:
                    catalog = haplotypes.prepare_reference(
                        panel_path,
                        samples,
                        reference,
                        af_cutoff,
                        focal_choice,
                        directory,
                    )
                    chunks = haplotypes.reference_chunks(panel_path, catalog)
                    panel = tsinfer.vcz.open_store(panel_path)
                    panel_genotypes = panel["call_genotype"]
                    packed = catalog.packed.open("r+")
            if collector is not None:
                slab_sizes = sorted(
                    [
                        len(chunk.job_indices) * reference.num_sites
                        for chunk in samples.chunks
                    ],
                    reverse=True,
                )
                live_count = len(slab_sizes) if resident else sample_prefetch_chunks
                packed_bytes = 0 if catalog is None else catalog.packed.shape[0]
                collector.count(
                    mapped_input_bound_bytes=packed_bytes + sum(slab_sizes[:live_count]),
                    packed_calls=packed_bytes,
                    eligible_ancestors=0 if catalog is None else len(catalog.seeds),
                )
                collector.diagnostics.update(
                    sample_chunk_shape=samples.chunk_shape,
                    haplotypes_per_slab=[
                        len(chunk.job_indices) for chunk in samples.chunks
                    ],
                    temporary_directory=str(directory),
                )
            del reference
            with performance.measure(
                collector, "coordinator_thread_startup", "coordinator"
            ):
                executor = cf.ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="matching"
                )
                for worker_index in range(workers):
                    futures.append(
                        executor.submit(
                            _worker_main,
                            work,
                            events,
                            start_matching,
                            stopping,
                            worker_index,
                            panel_genotypes,
                            model,
                            catalog,
                            packed,
                            min_focal_region_size,
                            collector is not None,
                            diagnostics,
                            collect_haplotype_timings,
                            condition_gaps,
                            full_hmm,
                        )
                    )
            with performance.measure(
                collector,
                "coordinator_packing_completion",
                "coordinator",
                inclusive=True,
            ):
                _pack_reference(work, events, chunks, max_pending_haplotypes, collector)
                if packed is not None:
                    packed.flags.writeable = False
            with performance.measure(
                collector,
                "coordinator_worker_readiness",
                "coordinator",
                inclusive=True,
            ):
                for _ in futures:
                    work.put(None)
                for _ in futures:
                    _, measurements = _event(events)
                    if collector is not None:
                        collector.merge(measurements)
            resident_slabs = None
            if resident:
                with performance.measure(
                    collector, "resident_sample_preload", "coordinator"
                ):
                    resident_slabs = [
                        haplotypes.load_sample_chunk(
                            samples, chunk, directory, sample_reader_threads
                        )
                        for chunk in samples.chunks
                    ]
                if collector is not None:
                    collector.count(
                        sample_blocks_read=sum(
                            slab.blocks_read for slab in resident_slabs
                        )
                    )
            else:
                loader = threading.Thread(
                    target=_load_samples,
                    args=(
                        samples,
                        directory,
                        slots,
                        stopping,
                        events,
                        sample_reader_threads,
                    ),
                )
                loader.start()
            with window if window is not None else contextlib.nullcontext():
                with performance.measure(
                    collector,
                    "coordinator_matching_drain",
                    "coordinator",
                    inclusive=True,
                ):
                    admitted = time.perf_counter() if collector is not None else None
                    start_matching.set()
                    _coordinate(
                        work,
                        events,
                        jobs,
                        tables,
                        nodes,
                        samples,
                        max_pending_haplotypes,
                        slots,
                        collector,
                        resident_slabs,
                        admitted,
                    )
        finally:
            with performance.measure(collector, "coordinator_shutdown", "coordinator"):
                stopping.set()
                for _ in range(sample_prefetch_chunks):
                    slots.release()
                if loader is not None:
                    loader.join()
                start_matching.set()
                while True:
                    try:
                        work.get_nowait()
                    except queue.Empty:
                        break
                for _ in futures:
                    work.put(None)
                if executor is not None:
                    executor.shutdown(wait=True)
                for future in futures:
                    future.result()
                if packed is not None:
                    haplotypes.close_mapping(packed)
                del model, catalog, cfg
        with performance.measure(collector, "finalise_output"):
            tables.edges.squash()
            ts = tsinfer.matching.finalise_match_tables(tables)
            ts.dump(output_path)
    logger.info("Wrote samples to %s using %d matching threads", output_path, workers)
    return workers
