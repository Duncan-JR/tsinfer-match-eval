"""Full-reference and focal-mask stitched sample matching."""

import concurrent.futures as cf
import dataclasses
import json
import logging
import pathlib
import threading

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
class SiteMatch:
    """Materialised parents and copied alleles for one requested site slice."""

    parents: np.ndarray
    copied_alleles: np.ndarray


def _match_region(matcher, haplotype, start, end, positions, match_out):
    """Run an independent HMM and copy its output before matcher reuse."""
    left, right, parents = matcher.find_path(haplotype, start, end, match_out)
    site_parents = np.empty(end - start, dtype=np.int32)
    for path_left, path_right, parent in zip(left, right, parents, strict=True):
        site_left = max(start, int(np.searchsorted(positions, path_left)))
        site_right = min(end, int(np.searchsorted(positions, path_right)))
        site_parents[site_left - start : site_right - start] = parent
    copied_alleles = match_out[start:end].copy()
    return SiteMatch(site_parents, copied_alleles)


def merge_paths(
    parents, copied_alleles, haplotype, positions, sequence_length, start, end
):
    """Encode the finished stitched domain as native paths and mutations.

    Adjacent equal parents merge across seams. Only called observations that
    disagree with the final copied alleles create mutations. Missing
    observations never create mutations.
    """
    domain_parents = parents[start:end]
    switches = np.flatnonzero(domain_parents[1:] != domain_parents[:-1]) + start + 1
    boundaries = [start, *switches, end]
    path = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        bp_left = 0 if left == 0 else float(positions[left])
        bp_right = (
            sequence_length if right == len(positions) else float(positions[right])
        )
        path.append(
            tsinfer.matching.PathSegment(
                left=bp_left, right=bp_right, parent=int(parents[left])
            )
        )
    observed = haplotype[start:end]
    copied = copied_alleles[start:end]
    mutation_sites = np.flatnonzero((observed >= 0) & (observed != copied)) + start
    mutations = [
        tsinfer.matching.Mutation(
            position=float(positions[site]), derived_state=int(haplotype[site])
        )
        for site in mutation_sites
    ]
    return tsinfer.matching.MatchResult(path=path, mutations=mutations)


@dataclasses.dataclass
class StitchedInputs:
    """Reference, indexes, and read-only inputs shared by matching threads."""

    ancestors_ts: tskit.TreeSequence
    positions: np.ndarray
    focal: dict
    masks: dict
    ancestor_nodes: dict
    interval_boundaries: np.ndarray
    num_alleles: np.ndarray
    recombination: np.ndarray
    mismatch: np.ndarray
    full_indexes: tsinfer.matching.MatcherIndexes
    ac_cutoff: int


@dataclasses.dataclass
class StitchedWorker:
    """Shared inputs and a reusable full-reference HMM owned by one thread."""

    inputs: StitchedInputs
    full_matcher: tsinfer.matching.AncestorMatcher


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
):
    """Load focal archives and masks and create shared indexes once."""
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
    )


def match_stitched_haplotype(job, haplotype, worker, method="hmm"):
    """Match each focal and gap run independently for every mask budget.

    Coverage is determined solely by :func:`lib.haplotypes.make_focal_mask`.
    Both kinds of run split at inference-interval boundaries and use absolute
    site indices with neutral missing calls. One reduced reference and matcher
    are reused across budgets; paths are recomputed for each component because
    its endpoints determine the independent HMM optimisation.
    """
    if method != "hmm":
        raise ValueError("Only focal method 'hmm' is supported")
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

    match_out = np.empty(num_sites, dtype=np.int8)
    focal_matcher = None
    if has_coverage:
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
            indexes, inputs.recombination, inputs.mismatch
        )

    results = {}
    for budget, mask in inputs.masks.items():
        parents = np.empty(num_sites, dtype=np.int32)
        copied_alleles = np.empty(num_sites, dtype=np.int8)
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
            boundaries = inputs.interval_boundaries
            internal = boundaries[(boundaries > left) & (boundaries < right)]
            run_bounds = [left, *internal, right]
            matcher = worker.full_matcher if is_gap else focal_matcher
            for run_start, run_end in zip(run_bounds[:-1], run_bounds[1:], strict=True):
                region = _match_region(
                    matcher,
                    haplotype,
                    run_start,
                    run_end,
                    inputs.positions,
                    match_out,
                )
                region_parents = region.parents
                if not is_gap:
                    region_parents = reduced.original_nodes[region_parents]
                parents[run_start:run_end] = region_parents
                copied_alleles[run_start:run_end] = region.copied_alleles
        results[budget] = merge_paths(
            parents,
            copied_alleles,
            haplotype,
            inputs.positions,
            inputs.ancestors_ts.sequence_length,
            start,
            end,
        )
    return StitchedResults(job, results)


def _worker_stitched_haplotype(job, reader, inputs, thread_state, method):
    """Read one haplotype in its task and reuse the thread's full matcher."""
    worker = getattr(thread_state, "worker", None)
    if worker is None:
        matcher = tsinfer.matching.AncestorMatcher(
            inputs.full_indexes, inputs.recombination, inputs.mismatch
        )
        worker = StitchedWorker(inputs, matcher)
        thread_state.worker = worker
    haplotype = np.asarray(reader.read_haplotype(job), dtype=np.int8)
    return match_stitched_haplotype(job, haplotype, worker, method=method)


def _write_stitched_matches(path, paired_results, ac_cutoff, max_mismatches, method):
    """Write native-shaped JSONL in haplotype order, with selected parameters."""
    with path.open("w") as match_file:
        for job, result in paired_results:
            record = {
                "group": job.group,
                "haplotype_index": job.haplotype_index,
                "source": job.source,
                "sample_id": job.sample_id,
                "ploidy_index": job.ploidy_index,
                "time": job.time,
                "path": [dataclasses.asdict(segment) for segment in result.path],
                "mutations": [
                    dataclasses.asdict(mutation) for mutation in result.mutations
                ],
                "ac_cutoff": ac_cutoff,
                "max_mismatches": max_mismatches,
                "method": method,
            }
            match_file.write(json.dumps(record) + "\n")


def match_stitched_samples(
    config_path: pathlib.Path,
    ancestors_ts_path: pathlib.Path,
    focal_path: pathlib.Path,
    mask_paths: list[pathlib.Path],
    output_paths: list[pathlib.Path],
    match_file_paths: list[pathlib.Path],
    ac_cutoff: int,
    max_mismatches: list[int],
    method: str,
    threads: int,
    cache_size: int | None = None,
) -> None:
    """Match one dataset/cutoff/method across all budgets and assemble raw TSs.

    Each native haplotype is read once inside its task. Threads share the
    reference, indexes and reader, reusing one full matcher per thread and one
    transient reduced matcher per haplotype across budgets. Simplification
    holds the GIL; threads parallelise HMM calls, not reference preparation.
    Native ``extend_ts`` preserves original reference node IDs and adds one node per
    haplotype, along with individual/population assignments and mutations.
    """
    if method != "hmm":
        raise ValueError("Only focal method 'hmm' is supported")
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
        )
        thread_state = threading.local()
        logger.info("Stitching %d haplotypes with %d threads", len(jobs), workers)
        if workers == 1:
            results = [
                _worker_stitched_haplotype(job, reader, inputs, thread_state, method)
                for job in jobs
            ]
        else:
            with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {
                    executor.submit(
                        _worker_stitched_haplotype,
                        job,
                        reader,
                        inputs,
                        thread_state,
                        method,
                    ): job
                    for job in jobs
                }
                results = []
                for future in cf.as_completed(futures):
                    results.append(future.result())
                    del futures[future]
        results.sort(key=lambda result: result.job.haplotype_index)
        outputs = zip(max_mismatches, output_paths, match_file_paths, strict=True)
        for budget, output_path, match_file_path in outputs:
            paired_results = [(result.job, result.results[budget]) for result in results]
            ts = tsinfer.matching.extend_ts(
                ancestors_ts,
                paired_results=paired_results,
                allele_mapper=reader.allele_mapper,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            match_file_path.parent.mkdir(parents=True, exist_ok=True)
            ts.dump(output_path)
            _write_stitched_matches(
                match_file_path, paired_results, ac_cutoff, budget, method
            )
            logger.info(
                "Wrote stitched samples to %s and %s", output_path, match_file_path
            )
    finally:
        reader.shutdown()
