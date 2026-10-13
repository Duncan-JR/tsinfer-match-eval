"""Full-reference and focal-mask stitched sample matching."""

import concurrent.futures as cf
import contextlib
import dataclasses
import json
import logging
import pathlib
import threading

import numba
import numpy as np
import tsinfer
import tskit

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
        cfg, group_stop=first_sample_group, num_threads=threads, cache_size=cache_size
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
    match_file_path: pathlib.Path,
    threads: int,
    cache_size: int | None = None,
) -> None:
    """Match samples against the unmodified reference from :func:`match_ancestors`.

    Preserve source selection/order and individual metadata so recreated job
    individual IDs correspond to the reference rows. Native JSONL has one record
    per haplotype, keyed by (source, sample_id, ploidy_index); parent IDs are raw
    TS node IDs, joined to panel IDs through node metadata. Focal NPZs do not
    participate in matching.
    """
    sample_cfg = _sample_config(config_path, ancestors_ts_path, output_path)
    ts = tsinfer.pipeline.match(
        sample_cfg,
        num_threads=threads,
        match_file=match_file_path,
        cache_size=cache_size,
    )
    ts.dump(output_path)
    logger.info(
        "Wrote raw samples to %s and matches to %s", output_path, match_file_path
    )


@dataclasses.dataclass
class ReducedAncestors:
    """Transient focal reference and reduced-to-original copying node IDs."""

    ts: tskit.TreeSequence
    original_nodes: np.ndarray


def simplify_ancestors_ts(
    ancestors_ts, focal, ancestor_nodes, haplotype_index, ac_cutoff
):
    """Retain roots and eligible panel candidates for one haplotype.

    ``ancestor_nodes`` joins panel IDs to reference node metadata once per
    dataset. Candidate columns are resolved through this join, never used as
    node IDs. Retained internal ancestors remain available copying states.
    """
    start = focal["offsets"][haplotype_index]
    end = focal["offsets"][haplotype_index + 1]
    columns = focal["ancestor_index"][start:end]
    eligible = columns[focal["derived_ac"][columns] <= ac_cutoff]
    focal_nodes = [ancestor_nodes[focal["ancestor_id"][column]] for column in eligible]
    retained_nodes = [0, 1, *focal_nodes]
    reduced_ts, old_to_new = ancestors_ts.simplify(
        retained_nodes, map_nodes=True, filter_sites=False
    )
    original_nodes = np.empty(reduced_ts.num_nodes, dtype=np.int32)
    mapped = np.flatnonzero(old_to_new != tskit.NULL)
    original_nodes[old_to_new[mapped]] = mapped
    return ReducedAncestors(reduced_ts, original_nodes)


@dataclasses.dataclass
class CopyingRuns:
    """Numeric copying runs with half-open bounds on the inference-site axis."""

    left_site: np.ndarray
    right_site: np.ndarray
    parent: np.ndarray


@dataclasses.dataclass
class RegionMatch:
    """Compact copying runs and owned mutations from one independent HMM."""

    runs: CopyingRuns
    mutations: list[tsinfer.matching.Mutation]


def _match_region(
    matcher, haplotype, start, end, positions, match_out, original_nodes=None
):
    """Collect ordered runs and mutations before matched-allele scratch is reused.

    Reduced parents are mapped back to the original reference here. Returned
    arrays and mutations never refer to ``match_out``; only its requested slice
    is read. See :func:`_assemble_stitched_match` for output seam merging.
    """
    left, right, parents = matcher.find_path(haplotype, start, end, match_out)
    left_site = np.searchsorted(positions, left[::-1])
    right_site = np.searchsorted(positions, right[::-1])
    left_site = np.maximum(left_site, start)
    right_site = np.minimum(right_site, end)
    parents = parents[::-1].copy()
    if original_nodes is not None:
        parents = original_nodes[parents]
    runs = CopyingRuns(left_site, right_site, parents)

    observed = haplotype[start:end]
    copied = match_out[start:end]
    mismatch_sites = np.flatnonzero((observed >= 0) & (observed != copied))
    mismatch_sites += start
    mutations = [
        tsinfer.matching.Mutation(
            position=float(positions[site]), derived_state=int(haplotype[site])
        )
        for site in mismatch_sites
    ]
    return RegionMatch(runs, mutations)


def _assemble_stitched_match(run_blocks, mutations, positions, sequence_length):
    """Coalesce ordered runs once and convert to native BP path segments.

    Touching equal parents merge across output seams. Regional HMM calls remain
    independent even when their output runs merge. Mutations arrive in site
    order from :func:`_match_region`; exact greedy focal runs add none.
    """
    left_site = np.concatenate([runs.left_site for runs in run_blocks])
    right_site = np.concatenate([runs.right_site for runs in run_blocks])
    parents = np.concatenate([runs.parent for runs in run_blocks])
    last = 0
    for current in range(1, len(parents)):
        touches = left_site[current] == right_site[last]
        same_parent = parents[current] == parents[last]
        if touches and same_parent:
            right_site[last] = right_site[current]
        else:
            last += 1
            left_site[last] = left_site[current]
            right_site[last] = right_site[current]
            parents[last] = parents[current]

    path = []
    for run in range(last + 1):
        left = left_site[run]
        right = right_site[run]
        bp_left = 0 if left == 0 else float(positions[left])
        bp_right = (
            sequence_length if right == len(positions) else float(positions[right])
        )
        path.append(
            tsinfer.matching.PathSegment(
                left=bp_left, right=bp_right, parent=int(parents[run])
            )
        )
    return tsinfer.matching.MatchResult(path=path, mutations=mutations)


@dataclasses.dataclass
class StitchedInputs:
    """Reference, indexes, and read-only inputs shared by matching threads."""

    ancestors_ts: tskit.TreeSequence
    positions: np.ndarray
    focal: dict | None
    masks: dict
    ancestor_nodes: dict
    interval_boundaries: np.ndarray
    num_alleles: np.ndarray
    recombination: np.ndarray
    mismatch: np.ndarray
    full_indexes: tsinfer.matching.MatcherIndexes
    ac_cutoff: int
    intervals: dict | None = None
    interval_nodes: np.ndarray | None = None


@dataclasses.dataclass
class StitchedWorker:
    """Shared inputs and a reusable full-reference HMM owned by one thread."""

    inputs: StitchedInputs
    full_matcher: tsinfer.matching.AncestorMatcher
    match_out: np.ndarray


@dataclasses.dataclass
class StitchedResults:
    """Native match results for every requested budget of one haplotype."""

    job: tsinfer.config.MatchJob
    results: dict[int, tsinfer.matching.MatchResult]


def _prepare_stitched_inputs(
    ancestors_ts,
    positions,
    focal_path,
    mask_paths,
    ac_cutoff,
    num_alleles,
    source_config,
    intervals_path=None,
):
    """Load focal archives and masks and create shared indexes once."""
    focal = None
    if focal_path is not None:
        with np.load(focal_path, allow_pickle=False) as archive:
            focal = {name: archive[name] for name in archive.files}
    masks = {}
    for mask_path in mask_paths:
        with np.load(mask_path, allow_pickle=False) as archive:
            mask = {name: archive[name] for name in archive.files}
        masks[int(mask["max_mismatches"])] = mask
    ancestor_nodes = {}
    for node in ancestors_ts.nodes():
        metadata = node.metadata
        if metadata.get("source") == "ancestors" and metadata["ploidy_index"] == 0:
            ancestor_nodes[metadata["sample_id"]] = node.id
    ancestor_intervals = None
    interval_nodes = None
    if intervals_path is not None:
        with np.load(intervals_path, allow_pickle=False) as archive:
            ancestor_intervals = {name: archive[name] for name in archive.files}
        if int(ancestor_intervals["num_sites"]) != len(positions):
            raise ValueError("Ancestor intervals disagree with the reference site axis")
        if ac_cutoff > int(ancestor_intervals["max_ac_cutoff"]):
            raise ValueError("AC cutoff exceeds the ancestor interval maximum")
        for mask in masks.values():
            for name in ("sample_id", "ploidy_index", "num_sites", "focal_choice"):
                if not np.array_equal(mask[name], ancestor_intervals[name]):
                    raise ValueError(f"Focal mask {name} disagrees with intervals")
            if int(mask["ac_cutoff"]) != ac_cutoff:
                raise ValueError("Focal mask disagrees with the requested AC cutoff")
        panel_nodes = np.asarray(
            [ancestor_nodes[name] for name in ancestor_intervals["ancestor_id"]],
            dtype=np.int32,
        )
        interval_nodes = panel_nodes[ancestor_intervals["ancestor_index"]]
    intervals = np.asarray(ancestors_ts.metadata["sequence_intervals"])
    interval_boundaries = np.unique(np.searchsorted(positions, intervals.ravel()))
    recombination = np.full(len(positions), source_config.recombination)
    mismatch = np.full(len(positions), source_config.mismatch)
    indexes = tsinfer.matching.MatcherIndexes(
        ancestors_ts, vestigial_root=False, num_alleles=num_alleles
    )
    return StitchedInputs(
        ancestors_ts=ancestors_ts,
        positions=positions,
        focal=focal,
        masks=masks,
        ancestor_nodes=ancestor_nodes,
        interval_boundaries=interval_boundaries,
        num_alleles=num_alleles,
        recombination=recombination,
        mismatch=mismatch,
        full_indexes=indexes,
        ac_cutoff=ac_cutoff,
        intervals=ancestor_intervals,
        interval_nodes=interval_nodes,
    )


@numba.njit(nogil=True)
def _greedy_cover(
    left, right, nodes, run_left, run_right, out_left, out_right, out_parent
):
    """Sweep sorted intervals once over ordered, disjoint coverage components.

    At the first uncovered site choose the containing interval with maximum
    exclusive right bound. Discard scanned alternatives: none extends further
    than the choice. Touching intervals can continue at that exclusive bound.
    Distinct components have no interval spanning between them, as guaranteed
    by :func:`lib.haplotypes.make_focal_mask`. Equal reaches keep the first
    interval in stable left-bound order. Each choice emits one copying run.
    """
    written = 0
    scan = 0
    for region in range(len(run_left)):
        cursor = run_left[region]
        end = run_right[region]
        while cursor < end:
            reach = cursor
            chosen = -1
            while scan < len(left) and left[scan] <= cursor:
                if right[scan] > reach:
                    reach = right[scan]
                    chosen = nodes[scan]
                scan += 1
            if chosen == -1:
                raise ValueError("Ancestor intervals do not cover the focal mask")
            reach = min(reach, end)
            out_left[written] = cursor
            out_right[written] = reach
            out_parent[written] = chosen
            written += 1
            cursor = reach
    return written


def greedy_focal_runs(inputs, haplotype_index, mask, start, end):
    """Return a minimum-switch exact cover of the requested mask's focal runs.

    This replaces the reduced HMM only for budget zero. Saved ancestor IDs are
    joined to original reference nodes during :func:`_prepare_stitched_inputs`.
    Sorting costs O(M log M); the sweep and output capacity cost O(M), for M
    eligible intervals. Inference boundaries need no splitting:
    every saved interval is confined to one inference interval.
    """
    if int(mask["max_mismatches"]) != 0:
        raise ValueError("Greedy focal matching requires max_mismatches=0")
    intervals = inputs.intervals
    if intervals is None:
        raise ValueError("Greedy focal matching requires ancestor intervals")
    first, last = intervals["offsets"][haplotype_index : haplotype_index + 2]
    eligible = np.flatnonzero(intervals["focal_ac"][first:last] <= inputs.ac_cutoff)
    eligible += first
    left = intervals["left_site_index"][eligible, 0]
    order = np.argsort(left, kind="stable")
    rows = eligible[order]
    run_start, run_end = mask["offsets"][haplotype_index : haplotype_index + 2]
    run_slice = slice(run_start, run_end)
    run_left = np.maximum(mask["left_site_index"][run_slice], start)
    run_right = np.minimum(mask["right_site_index"][run_slice], end)
    covered = ~mask["is_gap"][run_slice]
    covered &= run_left < run_right
    left_site = np.empty(len(rows), dtype=np.int64)
    right_site = np.empty(len(rows), dtype=np.int64)
    parents = np.empty(len(rows), dtype=np.int32)
    written = _greedy_cover(
        intervals["left_site_index"][rows, 0],
        intervals["right_site_index"][rows, 0],
        inputs.interval_nodes[rows],
        run_left[covered],
        run_right[covered],
        left_site,
        right_site,
        parents,
    )
    return CopyingRuns(
        left_site[:written].copy(),
        right_site[:written].copy(),
        parents[:written].copy(),
    )


def match_stitched_haplotype(job, haplotype, worker, method="hmm"):
    """Match each focal and gap run independently for every mask budget.

    Coverage is determined solely by :func:`lib.haplotypes.make_focal_mask`.
    HMM runs split at inference-interval boundaries and use absolute site
    indices with neutral missing calls. The ``hmm`` method prepares one reduced
    matcher across budgets. The ``greedy`` method covers saved exact intervals
    without a reduced reference; gaps use the full HMM with either method.
    """
    if method not in ("hmm", "greedy"):
        raise ValueError("Focal method must be 'hmm' or 'greedy'")
    inputs = worker.inputs
    num_sites = len(inputs.positions)
    called_sites = np.flatnonzero(haplotype >= 0)
    start, end = 0, num_sites
    if len(called_sites) > 0:
        start = int(called_sites[0])
        end = int(called_sites[-1]) + 1
    mask_slices = {}
    has_coverage = False
    for budget, mask in inputs.masks.items():
        run_start = mask["offsets"][job.haplotype_index]
        run_end = mask["offsets"][job.haplotype_index + 1]
        mask_slices[budget] = slice(run_start, run_end)
        covered = ~mask["is_gap"][run_start:run_end]
        overlaps = mask["left_site_index"][run_start:run_end] < end
        overlaps &= mask["right_site_index"][run_start:run_end] > start
        has_coverage |= bool(np.any(covered & overlaps))

    focal_matcher = None
    if has_coverage and method == "hmm":
        reduced = simplify_ancestors_ts(
            inputs.ancestors_ts,
            inputs.focal,
            inputs.ancestor_nodes,
            job.haplotype_index,
            inputs.ac_cutoff,
        )
        indexes = tsinfer.matching.MatcherIndexes(
            reduced.ts, vestigial_root=False, num_alleles=inputs.num_alleles
        )
        focal_matcher = tsinfer.matching.AncestorMatcher(
            indexes, inputs.recombination, inputs.mismatch, reuse_topology=True
        )

    greedy_runs = None
    if method == "greedy":
        greedy_runs = greedy_focal_runs(
            inputs, job.haplotype_index, inputs.masks[0], start, end
        )

    results = {}
    for budget, mask in inputs.masks.items():
        worker.full_matcher.reset_topology()
        if focal_matcher is not None:
            focal_matcher.reset_topology()
        run_blocks = []
        mutations = []
        focal_cursor = 0
        run_slice = mask_slices[budget]
        runs = zip(
            mask["left_site_index"][run_slice],
            mask["right_site_index"][run_slice],
            mask["is_gap"][run_slice],
            strict=True,
        )
        for left, right, is_gap in runs:
            left = max(start, int(left))
            right = min(end, int(right))
            if left >= right:
                continue
            if not is_gap and method == "greedy":
                first = focal_cursor
                while (
                    focal_cursor < len(greedy_runs.left_site)
                    and greedy_runs.left_site[focal_cursor] < right
                ):
                    focal_cursor += 1
                focal_slice = slice(first, focal_cursor)
                run_blocks.append(
                    CopyingRuns(
                        greedy_runs.left_site[focal_slice],
                        greedy_runs.right_site[focal_slice],
                        greedy_runs.parent[focal_slice],
                    )
                )
                continue
            boundaries = inputs.interval_boundaries
            internal = boundaries[(boundaries > left) & (boundaries < right)]
            run_bounds = [left, *internal, right]
            matcher = worker.full_matcher if is_gap else focal_matcher
            original_nodes = None if is_gap else reduced.original_nodes
            for run_start, run_end in zip(run_bounds[:-1], run_bounds[1:], strict=True):
                region = _match_region(
                    matcher,
                    haplotype,
                    run_start,
                    run_end,
                    inputs.positions,
                    worker.match_out,
                    original_nodes=original_nodes,
                )
                run_blocks.append(region.runs)
                mutations.extend(region.mutations)
        results[budget] = _assemble_stitched_match(
            run_blocks, mutations, inputs.positions, inputs.ancestors_ts.sequence_length
        )
    return StitchedResults(job, results)


def _worker_stitched_haplotype(job, reader, inputs, thread_state, method):
    """Read one haplotype in its task and reuse the thread's full matcher."""
    worker = getattr(thread_state, "worker", None)
    if worker is None:
        matcher = tsinfer.matching.AncestorMatcher(
            inputs.full_indexes,
            inputs.recombination,
            inputs.mismatch,
            reuse_topology=True,
        )
        match_out = np.empty(len(inputs.positions), dtype=np.int8)
        worker = StitchedWorker(inputs, matcher, match_out)
        thread_state.worker = worker
    haplotype = np.asarray(reader.read_haplotype(job), dtype=np.int8)
    return match_stitched_haplotype(job, haplotype, worker, method=method)


def _write_stitched_matches(
    match_file, paired_results, ac_cutoff, max_mismatches, method
):
    """Append native-shaped JSONL records to an open file in haplotype order."""
    for job, result in paired_results:
        record = {
            "group": job.group,
            "haplotype_index": job.haplotype_index,
            "source": job.source,
            "sample_id": job.sample_id,
            "ploidy_index": job.ploidy_index,
            "time": job.time,
            "path": [dataclasses.asdict(segment) for segment in result.path],
            "mutations": [dataclasses.asdict(mutation) for mutation in result.mutations],
            "ac_cutoff": ac_cutoff,
            "max_mismatches": max_mismatches,
            "method": method,
        }
        match_file.write(json.dumps(record) + "\n")


def _collect_stitched_batch(jobs, reader, inputs, thread_state, method, executor):
    """Match one contiguous batch, releasing its futures before returning."""
    if executor is None:
        results = [
            _worker_stitched_haplotype(job, reader, inputs, thread_state, method)
            for job in jobs
        ]
    else:
        futures = [
            executor.submit(
                _worker_stitched_haplotype, job, reader, inputs, thread_state, method
            )
            for job in jobs
        ]
        results = [future.result() for future in cf.as_completed(futures)]
    results.sort(key=lambda result: result.job.haplotype_index)
    return results


def _append_stitched_batch(
    results, tables, match_files, allele_mapper, ac_cutoff, method
):
    """Append a canonical batch to each budget, retaining no Python results."""
    for budget, budget_tables in tables.items():
        paired_results = [(result.job, result.results[budget]) for result in results]
        tsinfer.matching.append_match_results(
            budget_tables, paired_results, allele_mapper
        )
        _write_stitched_matches(
            match_files[budget], paired_results, ac_cutoff, budget, method
        )


def match_stitched_samples(
    config_path: pathlib.Path,
    ancestors_ts_path: pathlib.Path,
    focal_path: pathlib.Path | None,
    mask_paths: list[pathlib.Path],
    output_paths: list[pathlib.Path],
    match_file_paths: list[pathlib.Path],
    ac_cutoff: int,
    max_mismatches: list[int],
    method: str,
    threads: int,
    cache_size: int | None = None,
    intervals_path: pathlib.Path | None = None,
    *,
    haplotype_batch_size: int,
) -> None:
    """Match one dataset/cutoff/method across all budgets and assemble raw TSs.

    Each native haplotype is read once inside its task. Threads share the
    reference, indexes and reader, reusing one full matcher per thread and one
    transient reduced matcher per haplotype across budgets for method ``hmm``.
    Method ``greedy`` requires ``intervals_path`` and budget zero, and uses no
    reduced matcher. Numba and native HMM calls release the GIL. Simplification
    holds the GIL, so its preparation does not scale with matching threads.
    Contiguous canonical batches bound pending tasks and Python results. Each
    budget's tables start from the immutable reference and are finalised once.
    Native appending preserves original parent IDs, metadata and assignments.
    Final tables, input archives and reader chunks still scale with the output
    or cohort; ``haplotype_batch_size`` bounds only matching tasks/results.
    """
    if (
        isinstance(haplotype_batch_size, bool)
        or not isinstance(haplotype_batch_size, int)
        or haplotype_batch_size <= 0
    ):
        raise ValueError("haplotype_batch_size must be a positive integer")
    if method not in ("hmm", "greedy"):
        raise ValueError("Focal method must be 'hmm' or 'greedy'")
    if method == "greedy":
        if max_mismatches != [0]:
            raise ValueError("Greedy focal matching requires max_mismatches=[0]")
        if intervals_path is None:
            raise ValueError("Greedy focal matching requires ancestor intervals")
    cfg = _sample_config(config_path, ancestors_ts_path, output_paths[0])
    ancestors_ts = tskit.load(ancestors_ts_path)
    positions = ancestors_ts.sites_position.astype(np.int32)
    ancestral_alleles = np.asarray(
        [site.ancestral_state for site in ancestors_ts.sites()]
    )
    jobs, _, _ = tsinfer.pipeline._build_jobs_with_metadata(cfg, positions)
    jobs.sort(key=lambda job: job.haplotype_index)
    workers = min(threads, len(jobs))
    schedule = [(job.source, job.sample_id, job.ploidy_index) for job in jobs]
    reader = tsinfer.vcz.HaplotypeReader(
        cfg.sources,
        positions,
        ancestral_alleles,
        cache_size_mb=cache_size,
        schedule=schedule,
        num_threads=workers,
    )
    try:
        inputs = _prepare_stitched_inputs(
            ancestors_ts,
            positions,
            focal_path,
            mask_paths,
            ac_cutoff,
            reader.get_num_alleles(),
            cfg.match.sources["samples"],
            intervals_path=intervals_path,
        )
        thread_state = threading.local()
        logger.info(
            "Stitching %d haplotypes with %d threads in batches of %d",
            len(jobs),
            workers,
            haplotype_batch_size,
        )
        outputs = list(zip(max_mismatches, output_paths, match_file_paths, strict=True))
        tables = {budget: ancestors_ts.dump_tables() for budget in max_mismatches}
        with contextlib.ExitStack() as file_stack:
            match_files = {}
            for budget, output_path, match_file_path in outputs:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                match_file_path.parent.mkdir(parents=True, exist_ok=True)
                match_files[budget] = file_stack.enter_context(match_file_path.open("w"))
            with contextlib.ExitStack() as executor_stack:
                executor = None
                if workers > 1:
                    executor = executor_stack.enter_context(
                        cf.ThreadPoolExecutor(max_workers=workers)
                    )
                for first in range(0, len(jobs), haplotype_batch_size):
                    batch_jobs = jobs[first : first + haplotype_batch_size]
                    results = _collect_stitched_batch(
                        batch_jobs, reader, inputs, thread_state, method, executor
                    )
                    _append_stitched_batch(
                        results,
                        tables,
                        match_files,
                        reader.allele_mapper,
                        ac_cutoff,
                        method,
                    )
                    del results
            # Each budget's final tables/TS are released before the next one.
            for budget, output_path, match_file_path in outputs:
                budget_tables = tables.pop(budget)
                ts = tsinfer.matching.finalise_match_tables(budget_tables)
                ts.dump(output_path)
                del ts, budget_tables
                logger.info(
                    "Wrote stitched samples to %s and %s", output_path, match_file_path
                )
    finally:
        reader.shutdown()
