"""Native baseline and streamed, complete-haplotype greedy stitched matching."""

import collections
import contextlib
import dataclasses
import json
import logging
import multiprocessing as mp
import multiprocessing.connection
import pathlib
import tempfile
import threading
import time
import traceback

import numpy as np
import tsinfer
import tskit

from . import greedy, haplotypes, performance, utils

logger = logging.getLogger(__name__)


def match_ancestors(
    config_path: pathlib.Path,
    output_path: pathlib.Path,
    threads: int,
    cache_size: int | None = None,
) -> None:
    """Match ancestors before samples, preserving the raw reference TS.

    Keeping samples in the configuration preserves their full contig length and
    pre-created individual rows for :func:`match_samples`. Original ancestor
    times must put every ancestor group strictly before the first sample group.
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
    cache_size: int | None = None,
    read_workers: int | None = None,
) -> None:
    """Match samples against the unmodified reference from :func:`match_ancestors`.

    Preserve source selection/order and individual metadata so recreated job
    individual IDs correspond to the reference rows. Raw parent IDs remain
    available for comparisons through node metadata.
    """
    sample_cfg = _sample_config(config_path, ancestors_ts_path, output_path)
    ts = tsinfer.pipeline.match(
        sample_cfg,
        num_threads=threads,
        cache_size=cache_size,
        read_workers=read_workers,
    )
    ts.dump(output_path)
    logger.info("Wrote raw samples to %s", output_path)


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


@dataclasses.dataclass
class HaplotypeTask:
    """One complete row in a slab; no decoded calls travel through queues."""

    slab: haplotypes.MappedCalls
    row: int
    job_index: int
    chunk: int


@dataclasses.dataclass
class Worker:
    """One private HMM and reusable numeric scratch per matching process."""

    exact: greedy.ExactState
    positions: np.ndarray
    sequence_length: float
    boundaries: np.ndarray
    indexes: tsinfer.matching.MatcherIndexes
    matcher: tsinfer.matching.AncestorMatcher
    match_out: np.ndarray


def make_worker(reference, catalog, num_alleles, parameters):
    """Build native state once and warm the actual read-only Numba signatures."""
    positions = reference.sites_position.astype(np.int32)
    intervals = np.asarray(reference.metadata["sequence_intervals"])
    boundaries = np.unique(np.searchsorted(positions, intervals.ravel()))
    indexes = tsinfer.matching.MatcherIndexes(
        reference, vestigial_root=False, num_alleles=num_alleles
    )
    matcher = tsinfer.matching.AncestorMatcher(
        indexes,
        np.full(len(positions), parameters.recombination),
        np.full(len(positions), parameters.mismatch),
        reuse_topology=True,
    )
    worker = Worker(
        greedy.make_state(catalog),
        positions,
        reference.sequence_length,
        boundaries,
        indexes,
        matcher,
        np.empty(len(positions), dtype=np.int8),
    )
    dummy = np.full(len(positions), -1, dtype=np.int8)
    dummy.flags.writeable = False
    order = greedy.candidates(dummy, worker.exact, 0, len(dummy))
    greedy.cover(worker.exact, order, 1)
    return worker


def _match_region(haplotype, worker, start, end, collector=None):
    """Collect a regional HMM path and mutations before native scratch is reused."""
    with performance.measure(
        collector, "gap_native_find_path", "thread", parent_stage="gap_matching_assembly"
    ):
        left, right, parents = worker.matcher.find_path(
            haplotype, start, end, worker.match_out
        )
    with performance.measure(
        collector, "gap_path_conversion", "thread", parent_stage="gap_matching_assembly"
    ):
        left_site = np.searchsorted(worker.positions, left[::-1])
        right_site = np.searchsorted(worker.positions, right[::-1])
        np.maximum(left_site, start, out=left_site)
        np.minimum(right_site, end, out=right_site)
        runs = CopyingRuns(left_site, right_site, parents[::-1].copy())
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


def match_stitched_haplotype(job_index, haplotype, worker, minimum, collector=None):
    """Match one complete canonical haplotype against a prepared reference.

    Filter exact components, then copy supported one-site gaps using
    :func:`lib.greedy.stitched_intervals`. Correct called bridge disagreements
    with sample mutations. Remaining HMM intervals start fresh likelihoods.
    """
    called = np.flatnonzero(haplotype >= 0)
    start, end = 0, len(haplotype)
    if len(called) > 0:
        start, end = int(called[0]), int(called[-1]) + 1
    state = worker.exact
    with performance.measure(collector, "candidate_extension_cover", "thread"):
        order = greedy.candidates(haplotype, state, start, end)
        count = greedy.cover(state, order, minimum)
    blocks = []
    mutations = []
    focal_sites = 0
    focal_runs = 0
    gap_calls = 0
    one_site_gap_sites = 0
    with performance.measure(
        collector, "gap_matching_assembly", "thread", inclusive=True
    ):
        intervals = greedy.stitched_intervals(
            state, count, start, end, worker.boundaries
        )
        for interval in intervals:
            if interval.ancestor is None:
                region = _match_region(
                    haplotype, worker, interval.left, interval.right, collector
                )
                blocks.append(region.runs)
                mutations.append(region.mutation_site)
                gap_calls += 1
            else:
                ancestor = interval.ancestor
                parent = state.catalog.parents[ancestor]
                blocks.append(
                    CopyingRuns(
                        np.asarray([interval.left], dtype=np.int64),
                        np.asarray([interval.right], dtype=np.int64),
                        np.asarray([parent], dtype=np.int32),
                    )
                )
                focal_sites += interval.right - interval.left
                focal_runs += 1
                one_site_gap_sites += len(interval.bridged_sites)
                corrections = []
                for site in interval.bridged_sites:
                    offset = state.catalog.offsets[ancestor]
                    offset += site - state.catalog.starts[ancestor]
                    sample_call = haplotype[site]
                    if sample_call >= 0 and sample_call != state.packed[offset]:
                        corrections.append(site)
                if len(corrections) > 0:
                    mutations.append(np.asarray(corrections, dtype=np.int64))
        with performance.measure(
            collector,
            "gap_final_concatenation",
            "thread",
            parent_stage="gap_matching_assembly",
        ):
            result = _assemble_result(
                job_index, blocks, mutations, haplotype, worker, collector
            )
    if collector is not None:
        collector.count(
            matched_haplotypes=1,
            domain_sites=end - start,
            focal_sites=focal_sites,
            gap_sites=end - start - focal_sites,
            focal_runs=focal_runs,
            gap_calls=gap_calls,
            one_site_gap_sites=one_site_gap_sites,
        )
    return result


@performance.sample_memory()
def _worker_main(
    work,
    events,
    start_matching,
    panel_path,
    reference_path,
    catalog,
    num_alleles,
    parameters,
    minimum,
    helpers,
    collect,
    diagnostics,
):
    """Preparation barrier followed by matching, reusing the same processes."""
    worker = None
    try:
        utils.configure_helpers(helpers)
        while True:
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
                count = haplotypes.pack_reference_chunk(panel_path, catalog, chunk)
            collector.count(ancestor_blocks_read=count)
            events.put(("packed", collector))
        collector = performance.Collector(stages=diagnostics)
        with performance.measure(
            collector,
            "worker_setup",
            "thread",
            parent_stage="coordinator_worker_readiness",
        ):
            reference = tskit.load(reference_path)
            worker = make_worker(reference, catalog, num_alleles, parameters)
            del reference
        events.put(("worker_ready", collector))
        start_matching.wait()
        while True:
            task = work.get()
            if task is None:
                break
            calls = task.slab.open()
            try:
                row = calls[task.row]
                collector = (
                    performance.Collector(stages=diagnostics) if collect else None
                )
                result = match_stitched_haplotype(
                    task.job_index, row, worker, minimum, collector
                )
                del row
            finally:
                haplotypes.close_mapping(calls)
            # Every numeric output owns its data; the sample map is already closed.
            events.put(("result", (task.chunk, result)))
    except Exception:
        events.put(("error", traceback.format_exc()))
    finally:
        if worker is not None:
            haplotypes.close_mapping(worker.exact.packed)


def _watch_workers(processes, events, stopping):
    """Turn even a native crash into an event, without polling the work queues."""
    waiting = {process.sentinel: process for process in processes}
    while len(waiting) > 0:
        ready = multiprocessing.connection.wait(list(waiting))
        for sentinel in ready:
            process = waiting.pop(sentinel)
            if not stopping.is_set():
                events.put(
                    ("error", f"Matching worker {process.pid} exited unexpectedly")
                )


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


def _load_samples(inputs, directory, slots, stopping, events):
    """Publish complete slabs; a slot covers loading, waiting and worker use."""
    try:
        for chunk in inputs.chunks:
            slots.acquire()
            if stopping.is_set():
                return
            slab = haplotypes.load_sample_chunk(inputs, chunk, directory)
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
                collector, "assembly_output", parent_stage="coordinator_matching_drain"
            ):
                append_result(
                    tables, nodes[result.job_index], result, samples.allele_mapper
                )
            if collector is not None and result.measurements is not None:
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
    helper_threads: dict,
    focal_choice: str = "left",
    temp_dir: pathlib.Path | None = None,
    collector: performance.Collector | None = None,
    resident: bool = False,
    window: performance.Window | None = None,
) -> int:
    """Pack once, stream slabs and append completed full-haplotype jobs immediately.

    The entry point receives validated configuration. One spawn pool handles
    packing and matching; a single loader owns sample Zarr reads. Outstanding
    jobs and live decoded slabs have separate bounds. Temporary inputs are
    removed only after loader and children stop. Returns the actual worker count.

    Diagnostic resident runs preload all slabs and initialise workers before
    ``window``; the benchmark limits their total size. Normal end-to-end timing
    wraps this entire function, including startup, packing and final output.
    """
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
    utils.configure_helpers(helper_threads)
    with performance.measure(collector, "input_setup"):
        reference = tskit.load(ancestors_ts_path)
        positions = reference.sites_position.astype(np.int32)
        jobs, individuals, populations = tsinfer.pipeline._build_jobs_with_metadata(
            cfg, positions
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
        tables = reference.dump_tables()
        metadata = tables.metadata
        metadata["match_eval"] = {
            "af_cutoff": af_cutoff,
            "min_region_size_kb": min_region_size_kb,
            "min_focal_region_size": min_focal_region_size,
            "focal_choice": focal_choice,
        }
        tables.metadata = metadata
        if collector is not None:
            collector.count(min_focal_region_size=min_focal_region_size)
        if [row.metadata for row in tables.individuals] != individuals:
            raise ValueError(
                "Reference individuals disagree with selected sample metadata"
            )
        if [row.metadata for row in tables.populations] != populations:
            raise ValueError(
                "Reference populations disagree with selected sample metadata"
            )
        nodes = _create_sample_nodes(tables, jobs)
        del positions, individuals, populations
    if workers > len(jobs):
        raise ValueError(f"Requested {workers} workers for only {len(jobs)} haplotypes")
    ctx = mp.get_context("spawn")
    work = ctx.Queue(maxsize=max_pending_haplotypes)
    events = ctx.Queue()
    start_matching = ctx.Event()
    stopping = threading.Event()
    slots = threading.Semaphore(sample_prefetch_chunks)
    processes = []
    loader = None
    watcher = None
    matched = False
    diagnostics = collector is not None and collector.stages
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_dir, prefix="greedy_") as temporary:
        directory = pathlib.Path(temporary)
        try:
            with performance.measure(collector, "reference_catalog"):
                catalog = haplotypes.prepare_reference(
                    panel_path,
                    samples,
                    reference,
                    af_cutoff,
                    focal_choice,
                    directory,
                )
            if collector is not None:
                slab_sizes = [
                    len(chunk.job_indices) * reference.num_sites
                    for chunk in samples.chunks
                ]
                slab_sizes.sort(reverse=True)
                live_count = len(slab_sizes) if resident else sample_prefetch_chunks
                mapped_bound = catalog.packed.shape[0] + sum(slab_sizes[:live_count])
                collector.count(
                    mapped_input_bound_bytes=mapped_bound,
                    packed_calls=catalog.packed.shape[0],
                    eligible_ancestors=len(catalog.seeds),
                )
            del reference
            with performance.measure(
                collector, "coordinator_process_startup", "coordinator"
            ):
                for _ in range(workers):
                    process = ctx.Process(
                        target=_worker_main,
                        args=(
                            work,
                            events,
                            start_matching,
                            panel_path,
                            ancestors_ts_path,
                            catalog,
                            samples.allele_mapper.num_alleles_array(),
                            cfg.match.sources["samples"],
                            min_focal_region_size,
                            helper_threads,
                            collector is not None,
                            diagnostics,
                        ),
                    )
                    process.start()
                    processes.append(process)
                del cfg
                watcher = threading.Thread(
                    target=_watch_workers, args=(processes, events, stopping)
                )
                watcher.start()
            with performance.measure(
                collector,
                "coordinator_packing_completion",
                "coordinator",
                inclusive=True,
            ):
                _pack_reference(
                    work,
                    events,
                    haplotypes.reference_chunks(panel_path, catalog),
                    max_pending_haplotypes,
                    collector,
                )
            with performance.measure(
                collector, "coordinator_worker_readiness", "coordinator", inclusive=True
            ):
                for _ in processes:
                    work.put(None)
                for _ in processes:
                    _, measurements = _event(events)
                    if collector is not None:
                        collector.merge(measurements)
            del catalog
            resident_slabs = None
            if resident:
                with performance.measure(
                    collector, "resident_sample_preload", "coordinator"
                ):
                    resident_slabs = [
                        haplotypes.load_sample_chunk(samples, chunk, directory)
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
                    args=(samples, directory, slots, stopping, events),
                )
                loader.start()
            with window if window is not None else contextlib.nullcontext():
                with performance.measure(
                    collector,
                    "coordinator_matching_drain",
                    "coordinator",
                    inclusive=True,
                ):
                    admitted = time.perf_counter() if diagnostics else None
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
            matched = True
        finally:
            with performance.measure(collector, "coordinator_shutdown", "coordinator"):
                stopping.set()
                for _ in range(sample_prefetch_chunks):
                    slots.release()
                if loader is not None:
                    loader.join()
                if matched:
                    for _ in processes:
                        work.put(None)
                else:
                    for process in processes:
                        process.terminate()
                for process in processes:
                    process.join()
                if watcher is not None:
                    watcher.join()
                for channel in (work, events):
                    if not matched:
                        channel.cancel_join_thread()
                    channel.close()
                    if matched:
                        channel.join_thread()
        with performance.measure(collector, "finalise_output"):
            tables.edges.squash()
            ts = tsinfer.matching.finalise_match_tables(tables)
            ts.dump(output_path)
    logger.info(
        "Wrote greedy stitched samples to %s using %d workers", output_path, workers
    )
    return workers
