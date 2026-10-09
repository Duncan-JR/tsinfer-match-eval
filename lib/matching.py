"""Full-reference and focal-mask stitched sample matching."""

import dataclasses
import json
import logging
import multiprocessing
import pathlib

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
    worker. Candidate columns are resolved through this join, never used as
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
    disagree with the final copied alleles create mutations; discarded reduced
    gap mismatches therefore never enter the output.
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
class StitchedWorker:
    """Read-only inputs and mutable full-reference HMM owned by one worker."""

    ancestors_ts: tskit.TreeSequence
    positions: np.ndarray
    focal: dict
    masks: dict
    ancestor_nodes: dict
    interval_boundaries: np.ndarray
    num_alleles: np.ndarray
    recombination: np.ndarray
    mismatch: np.ndarray
    full_matcher: tsinfer.matching.AncestorMatcher
    ac_cutoff: int
    method: str


@dataclasses.dataclass
class ObservedHaplotype:
    """One native job and its single reader observation, queued to a worker."""

    job: tsinfer.config.MatchJob
    haplotype: np.ndarray


@dataclasses.dataclass
class StitchedResults:
    """Native match results for every requested budget of one haplotype."""

    job: tsinfer.config.MatchJob
    results: dict[int, tsinfer.matching.MatchResult]


def _initialise_stitched_worker(
    ancestors_ts_path,
    focal_path,
    mask_paths,
    ac_cutoff,
    method,
    num_alleles,
    source_config,
):
    """Load reference and masks and create full-reference indexes once."""
    global _stitched_worker
    ancestors_ts = tskit.load(ancestors_ts_path)
    positions = ancestors_ts.sites_position.astype(np.int32)
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
    full_matcher = tsinfer.matching.AncestorMatcher(indexes, recombination, mismatch)
    _stitched_worker = StitchedWorker(
        ancestors_ts=ancestors_ts,
        positions=positions,
        focal=focal,
        masks=masks,
        ancestor_nodes=ancestor_nodes,
        interval_boundaries=interval_boundaries,
        num_alleles=num_alleles,
        recombination=recombination,
        mismatch=mismatch,
        full_matcher=full_matcher,
        ac_cutoff=ac_cutoff,
        method=method,
    )


def match_stitched_haplotype(job, haplotype, worker, method="hmm"):
    """Reuse one whole-genome focal HMM, then fill each budget's mask gaps.

    Coverage is determined solely by :func:`lib.haplotypes.make_focal_mask`.
    Gap HMMs start independently at inference-interval boundaries, using the
    full observed array with absolute site indices and neutral missing calls.
    """
    if method != "hmm":
        raise ValueError("Only focal method 'hmm' is supported")
    num_sites = len(worker.positions)
    called_sites = np.flatnonzero(haplotype >= 0)
    start, end = 0, num_sites
    if len(called_sites) > 0:
        start = int(called_sites[0])
        end = int(called_sites[-1]) + 1
    mask_slices = {}
    has_coverage = False
    for budget, mask in worker.masks.items():
        run_start = mask["offsets"][job.haplotype_index]
        run_end = mask["offsets"][job.haplotype_index + 1]
        mask_slices[budget] = slice(run_start, run_end)
        has_coverage |= bool(np.any(~mask["is_gap"][run_start:run_end]))

    match_out = np.empty(num_sites, dtype=np.int8)
    focal_match = None
    if has_coverage:
        reduced = simplify_ancestors_ts(
            worker.ancestors_ts,
            worker.focal,
            worker.ancestor_nodes,
            job.haplotype_index,
            worker.ac_cutoff,
        )
        indexes = tsinfer.matching.MatcherIndexes(
            reduced.ts, vestigial_root=False, num_alleles=worker.num_alleles
        )
        matcher = tsinfer.matching.AncestorMatcher(
            indexes, worker.recombination, worker.mismatch
        )
        focal_match = _match_region(
            matcher, haplotype, 0, num_sites, worker.positions, match_out
        )
        focal_match.parents = reduced.original_nodes[focal_match.parents]

    results = {}
    for budget, mask in worker.masks.items():
        if focal_match is None:
            parents = np.empty(num_sites, dtype=np.int32)
            copied_alleles = np.empty(num_sites, dtype=np.int8)
        else:
            parents = focal_match.parents.copy()
            copied_alleles = focal_match.copied_alleles.copy()
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
            if not is_gap or left >= right:
                continue
            boundaries = worker.interval_boundaries
            internal = boundaries[(boundaries > left) & (boundaries < right)]
            gap_bounds = [left, *internal, right]
            for gap_start, gap_end in zip(gap_bounds[:-1], gap_bounds[1:], strict=True):
                gap_match = _match_region(
                    worker.full_matcher,
                    haplotype,
                    gap_start,
                    gap_end,
                    worker.positions,
                    match_out,
                )
                parents[gap_start:gap_end] = gap_match.parents
                copied_alleles[gap_start:gap_end] = gap_match.copied_alleles
        results[budget] = merge_paths(
            parents,
            copied_alleles,
            haplotype,
            worker.positions,
            worker.ancestors_ts.sequence_length,
            start,
            end,
        )
    return StitchedResults(job, results)


def _worker_stitched_haplotype(task):
    return match_stitched_haplotype(
        task.job, task.haplotype, _stitched_worker, method=_stitched_worker.method
    )


def _observed_haplotypes(jobs, reader):
    for job in jobs:
        haplotype = np.asarray(reader.read_haplotype(job), dtype=np.int8)
        yield ObservedHaplotype(job, haplotype)


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

    Each native haplotype is read once. Queued worker tasks reuse their full
    matcher and transient reduced reference across budgets. Native
    ``extend_ts`` preserves original reference node IDs and adds one node per
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
        initargs = (
            ancestors_ts_path,
            focal_path,
            mask_paths,
            ac_cutoff,
            method,
            reader.get_num_alleles(),
            cfg.match.sources["samples"],
        )
        tasks = _observed_haplotypes(jobs, reader)
        logger.info("Stitching %d haplotypes with %d workers", len(jobs), workers)
        if workers == 1:
            _initialise_stitched_worker(*initargs)
            results = [_worker_stitched_haplotype(task) for task in tasks]
        else:
            context = multiprocessing.get_context("spawn")
            with context.Pool(
                workers, initializer=_initialise_stitched_worker, initargs=initargs
            ) as pool:
                results = list(
                    pool.imap_unordered(_worker_stitched_haplotype, tasks, chunksize=1)
                )
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
