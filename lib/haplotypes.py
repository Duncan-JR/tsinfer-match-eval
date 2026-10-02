"""Direct inferred focal-ancestor comparisons, independent of HMM matching."""

import concurrent.futures as cf
import dataclasses
import logging
import multiprocessing
import pathlib
import time

import numba
import numpy as np
import pandas as pd
import tsinfer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ComparisonData:
    """Validated site axis, resident sample calls, and canonical seed order."""

    panel: object
    positions: np.ndarray
    calls: np.ndarray
    sample_ids: np.ndarray
    ploidy_indices: np.ndarray
    ancestor_ids: np.ndarray
    counts: np.ndarray
    frequencies: np.ndarray
    haplotypes: np.ndarray
    ancestors: np.ndarray
    focals: np.ndarray
    left: np.ndarray
    right: np.ndarray
    bp_left: np.ndarray
    bp_right: np.ndarray
    column_width: int
    site_height: int
    block_seeds: dict


@dataclasses.dataclass
class BlockBounds:
    """Numeric block results indexed by the canonical global seed number."""

    seeds: np.ndarray
    left: np.ndarray
    right: np.ndarray
    setup_seconds: float | None = None
    jit_seconds: float | None = None


def _prepare(
    samples_path,
    ancestors_path,
    focal_path,
    ancestral_state,
    max_ac_cutoff,
    sample_selection,
):
    """Align and polarise calls and expand seeds for construct_focal_ancestor_chunks."""
    samples = tsinfer.vcz.open_store(samples_path)
    panel = tsinfer.vcz.open_store(ancestors_path)
    positions = panel["variant_position"][:]
    sample_positions = samples["variant_position"][:]
    rows = np.searchsorted(sample_positions, positions)
    present = rows < len(sample_positions)
    present[present] = sample_positions[rows[present]] == positions[present]
    if not np.all(present):
        absent = positions[np.flatnonzero(~present)[0]]
        raise ValueError(f"Panel position {absent} is absent from the sample store")
    alleles = samples["variant_allele"].oindex[rows, :]
    ancestral = alleles[:, 0]
    if not ancestral_state.get("is_reference", False):
        ancestral = samples[ancestral_state["field"]].oindex[rows]
    reference_ancestral = ancestral == alleles[:, 0]
    alternate_ancestral = ancestral == alleles[:, 1]
    if not np.all(reference_ancestral | alternate_ancestral):
        raise ValueError("Ancestral state must be one of the two sample alleles")
    canonical_alleles = alleles[:, :2].copy()
    canonical_alleles[~reference_ancestral] = alleles[~reference_ancestral, :2][:, ::-1]
    if not np.array_equal(panel["variant_allele"][:], canonical_alleles):
        raise ValueError("Panel allele order disagrees with sample ancestral polarity")
    columns = tsinfer.vcz.resolve_samples_selection(samples, sample_selection)
    calls = samples["call_genotype"].oindex[rows, columns, :]
    if np.any((calls < -1) | (calls > 1)):
        raise ValueError("Sample calls must be biallelic codes 0, 1, or missing -1")
    swapped = ~reference_ancestral[:, None, None] & (calls >= 0)
    calls[swapped] = 1 - calls[swapped]
    ploidy = calls.shape[2]
    calls = calls.reshape(len(positions), -1)
    ids = np.asarray(samples["sample_id"].oindex[columns].tolist(), dtype=str)
    sample_ids = np.repeat(ids, ploidy)
    ploidy_indices = np.tile(np.arange(ploidy), len(ids))
    ancestor_ids = np.asarray(panel["sample_id"][:].tolist(), dtype=str)
    counts = samples["variant_match_eval_derived_ac"].oindex[rows]
    frequencies = samples["variant_match_eval_derived_af"].oindex[rows]
    starts = panel["sample_start_position"][:]
    ends = panel["sample_end_position"][:]
    intervals = panel["sequence_intervals"][:]
    focal_positions = panel["sample_focal_positions"][:]
    with np.load(focal_path, allow_pickle=False) as candidates:
        for name, expected in (
            ("sample_id", sample_ids),
            ("ploidy_index", ploidy_indices),
            ("ancestor_id", ancestor_ids),
        ):
            if not np.array_equal(candidates[name], expected):
                raise ValueError(f"Focal NPZ {name} disagrees with store identity/order")
        offsets = candidates["offsets"]
        columns = candidates["ancestor_index"]
        derived_ac = candidates["derived_ac"]
        if derived_ac.shape != ancestor_ids.shape:
            raise ValueError("Focal NPZ counts are not panel-aligned")
        if (
            len(offsets) != len(sample_ids) + 1
            or offsets[0] != 0
            or offsets[-1] != len(columns)
            or np.any(np.diff(offsets) < 0)
            or np.any((columns < 0) | (columns >= len(ancestor_ids)))
        ):
            raise ValueError("Invalid focal NPZ ragged candidate indices")
        ancestor_focals = []
        for ancestor, values in enumerate(focal_positions):
            values = values[values >= 0]
            indices = np.searchsorted(positions, values)
            if np.any(indices >= len(positions)):
                raise ValueError("Ancestor focal position is absent from panel")
            if not np.array_equal(positions[indices], values):
                raise ValueError("Ancestor focal position is absent from panel")
            if np.any(counts[indices] != derived_ac[ancestor]):
                raise ValueError("Focal NPZ counts disagree with sample annotations")
            ancestor_focals.append(np.sort(indices))
        haplotypes = []
        ancestors = []
        focals = []
        for haplotype in range(len(sample_ids)):
            selected = columns[offsets[haplotype] : offsets[haplotype + 1]]
            if np.any(np.diff(selected) <= 0):
                raise ValueError("Candidate columns must be sorted and distinct")
            selected = selected[derived_ac[selected] <= max_ac_cutoff]
            for ancestor in selected:
                indices = ancestor_focals[ancestor]
                carried = calls[indices, haplotype] >= 0
                indices = indices[carried]
                if np.any(calls[indices, haplotype] != 1):
                    raise ValueError("Selected focal sample call is not derived")
                haplotypes.extend([haplotype] * len(indices))
                ancestors.extend([ancestor] * len(indices))
                focals.extend(indices)
    haplotypes = np.asarray(haplotypes, dtype=np.int64)
    ancestors = np.asarray(ancestors, dtype=np.int64)
    focals = np.asarray(focals, dtype=np.int64)
    # Each seed is confined to the inference interval containing its focal site.
    interval_indices = np.searchsorted(intervals[:, 0], positions[focals], side="right")
    interval_indices -= 1
    if np.any(interval_indices < 0):
        raise ValueError("Focal site lies outside inference intervals")
    containing = intervals[interval_indices]
    if np.any(positions[focals] >= containing[:, 1]):
        raise ValueError("Focal site lies in an inference interval gap")
    bp_left = np.maximum(starts[ancestors], containing[:, 0]).astype(np.int64)
    bp_right = np.minimum(ends[ancestors], containing[:, 1]).astype(np.int64)
    left = np.searchsorted(positions, bp_left)
    right = np.searchsorted(positions, bp_right)
    if np.any((left > focals) | (right <= focals)):
        raise ValueError("Focal site lies outside ancestor support")
    column_width = panel["call_genotype"].chunks[1]
    site_height = panel["call_genotype"].chunks[0]
    block_ids = ancestors // column_width
    order = np.argsort(block_ids, kind="stable")
    sorted_blocks = block_ids[order]
    unique_blocks, first = np.unique(sorted_blocks, return_index=True)
    block_seeds = {}
    if len(focals) > 0:
        groups = np.split(order, first[1:])
        block_seeds = dict(zip(unique_blocks.tolist(), groups, strict=True))
    return ComparisonData(
        panel,
        positions,
        calls,
        sample_ids,
        ploidy_indices,
        ancestor_ids,
        counts,
        frequencies,
        haplotypes,
        ancestors,
        focals,
        left,
        right,
        bp_left,
        bp_right,
        column_width,
        site_height,
        block_seeds,
    )


@numba.njit
def _sweep_block(
    genotypes,
    site_start,
    samples,
    haplotypes,
    columns,
    focals,
    support,
    events,
    active,
    counters,
    progress,
    bounds,
    direction,
):
    """Advance a directional event sweep through one physical site block.

    Used by :func:`construct_focal_ancestor_chunks`. Bounds are half-open;
    budgets apply independently to each side. With support [0, 10), focal 4,
    left mismatches 3, 1 and right mismatches 6, 8, budgets 0, 1, 2 retain
    [4, 6), [2, 8), [0, 10). The (k + 1)th mismatch is excluded.
    ``progress`` holds event cursor and active length across physical blocks.
    """
    cursor = progress[0]
    active_length = progress[1]
    height = genotypes.shape[0]
    site_stop = site_start + height
    site = site_start if direction == 1 else site_stop - 1
    num_budgets = bounds.shape[1]
    while site_start <= site < site_stop:
        if active_length == 0:
            if cursor == len(events):
                break
            next_site = focals[events[cursor]]
            if next_site < site_start or next_site >= site_stop:
                break
            site = next_site
        while cursor < len(events) and focals[events[cursor]] == site:
            seed = events[cursor]
            local_site = site - site_start
            if genotypes[local_site, columns[seed]] != 1:
                raise ValueError("Selected focal ancestor call is not derived")
            active[active_length] = seed
            active_length += 1
            cursor += 1
        slot = 0
        while slot < active_length:
            seed = active[slot]
            count = counters[seed]
            outside = site >= support[seed] if direction == 1 else site < support[seed]
            remove = outside
            if not outside:
                sample_call = samples[site, haplotypes[seed]]
                ancestor_call = genotypes[site - site_start, columns[seed]]
                boundary = site if direction == 1 else site + 1
                if sample_call == -1 or ancestor_call == -1:
                    for budget in range(count, num_budgets):
                        bounds[seed, budget] = boundary
                    remove = True
                elif sample_call != ancestor_call:
                    bounds[seed, count] = boundary
                    counters[seed] += 1
                    remove = counters[seed] == num_budgets
            if remove:
                active_length -= 1
                active[slot] = active[active_length]
            else:
                slot += 1
        site += direction
    progress[0] = cursor
    progress[1] = active_length


def _block_bounds(data, block, max_mismatches):
    """Read physical ancestor columns once and reuse across all sample seeds.

    Multi-height stores stream forward and backward, preserving the numeric
    sweep state. A full-height block is retained for both directions.
    """
    seeds = data.block_seeds[block]
    column_start = block * data.column_width
    column_end = min(column_start + data.column_width, len(data.ancestor_ids))
    columns = data.ancestors[seeds] - column_start
    haplotypes = data.haplotypes[seeds]
    focals = data.focals[seeds]
    order = np.argsort(focals, kind="stable")
    left = np.repeat(data.left[seeds, None], max_mismatches + 1, axis=1)
    right = np.repeat(data.right[seeds, None], max_mismatches + 1, axis=1)
    genotypes = data.panel["call_genotype"]
    num_sites = len(data.positions)
    resident = None
    if data.site_height >= num_sites:
        resident = genotypes[:, column_start:column_end, 0]
    for direction, bounds, support in (
        (1, right, data.right[seeds]),
        (-1, left, data.left[seeds]),
    ):
        events = order if direction == 1 else order[::-1].copy()
        active = np.empty(len(seeds), dtype=np.int64)
        counters = np.zeros(len(seeds), dtype=np.int64)
        progress = np.zeros(2, dtype=np.int64)
        site_starts = range(0, num_sites, data.site_height)
        if direction == -1:
            site_starts = reversed(site_starts)
        for site_start in site_starts:
            site_end = min(site_start + data.site_height, num_sites)
            if progress[1] == 0:
                if progress[0] == len(events):
                    break
                next_site = focals[events[progress[0]]]
                if not site_start <= next_site < site_end:
                    continue
            calls = resident
            if calls is None:
                calls = genotypes[site_start:site_end, column_start:column_end, 0]
            _sweep_block(
                calls,
                site_start,
                data.calls,
                haplotypes,
                columns,
                focals,
                support,
                events,
                active,
                counters,
                progress,
                bounds,
                direction,
            )
    return BlockBounds(seeds, left, right)


def _initialise_worker(arguments, max_mismatches):
    """Give each spawned process its own read-only stores and numeric kernel."""
    global _worker_data, _worker_max_mismatches, _worker_timings
    started = time.perf_counter()
    _worker_data = _prepare(*arguments)
    _worker_max_mismatches = max_mismatches
    prepared = time.perf_counter()
    # Compile once without reading any ancestor calls or caching on disk.
    _sweep_block(
        np.empty((0, 0), dtype=np.int8),
        0,
        _worker_data.calls,
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.zeros(2, dtype=np.int64),
        np.empty((0, max_mismatches + 1), dtype=np.int64),
        1,
    )
    _worker_timings = (prepared - started, time.perf_counter() - prepared)


def _worker_block(block):
    global _worker_timings
    result = _block_bounds(_worker_data, block, _worker_max_mismatches)
    if _worker_timings is not None:
        result.setup_seconds, result.jit_seconds = _worker_timings
        _worker_timings = None
    return result


def _collect_bounds(data, arguments, max_mismatches, threads):
    """Bound process submissions/results by allocation; propagate worker errors."""
    shape = (len(data.focals), max_mismatches + 1)
    left = np.empty(shape, dtype=np.int64)
    right = np.empty(shape, dtype=np.int64)
    blocks = iter(data.block_seeds)
    workers = min(threads, len(data.block_seeds))
    if workers <= 1:
        for block in blocks:
            result = _block_bounds(data, block, max_mismatches)
            left[result.seeds] = result.left
            right[result.seeds] = result.right
    else:
        context = multiprocessing.get_context("spawn")
        started = time.perf_counter()
        with cf.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialise_worker,
            initargs=(arguments, max_mismatches),
        ) as executor:
            pending = {
                executor.submit(_worker_block, next(blocks)) for _ in range(workers)
            }
            while len(pending) > 0:
                completed, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                for future in completed:
                    result = future.result()
                    if result.setup_seconds is not None:
                        logger.info(
                            "First worker result at %.3fs: metadata setup %.3fs, "
                            "JIT %.3fs (elapsed includes process startup and task)",
                            time.perf_counter() - started,
                            result.setup_seconds,
                            result.jit_seconds,
                        )
                    left[result.seeds] = result.left
                    right[result.seeds] = result.right
                    block = next(blocks, None)
                    if block is not None:
                        pending.add(executor.submit(_worker_block, block))
    return BlockBounds(np.arange(len(data.focals)), left, right)


def _positions(data, bounds):
    """Convert half-open site cells to BP, clipped to support/inference intervals.

    See :func:`construct_focal_ancestor_chunks`. At an internal inference-interval
    end the next site's position is clipped to the containing interval's BP end.
    """
    left = data.positions[bounds.left]
    left = np.maximum(left, data.bp_left[:, None])
    # A terminal right index can equal num_sites; replace it before site lookup.
    lookup = np.minimum(bounds.right, len(data.positions) - 1)
    right = data.positions[lookup].copy()
    terminal = bounds.right == len(data.positions)
    endpoints = np.broadcast_to(data.bp_right[:, None], right.shape)
    right[terminal] = endpoints[terminal]
    right = np.minimum(right, endpoints)
    return BlockBounds(bounds.seeds, left, right)


def _dataframe(data, bounds, max_mismatches):
    """Assemble numeric columns and categorical identifiers without row dictionaries."""
    bp = _positions(data, bounds)
    budgets = max_mismatches + 1
    haplotypes = np.repeat(data.haplotypes, budgets)
    ancestors = np.repeat(data.ancestors, budgets)
    focals = np.repeat(data.focals, budgets)
    num_rows = len(focals)
    return pd.DataFrame(
        {
            "source": pd.Categorical.from_codes(
                np.zeros(num_rows, dtype=np.int8), ["samples"]
            ),
            "sample_id": pd.Categorical(
                data.sample_ids[haplotypes], categories=np.unique(data.sample_ids)
            ),
            "ploidy_index": data.ploidy_indices[haplotypes],
            "ancestor_id": pd.Categorical.from_codes(ancestors, data.ancestor_ids),
            "ancestor_index": ancestors,
            "focal_site_index": focals,
            "focal_position": data.positions[focals],
            "focal_ac": data.counts[focals].astype(np.int64),
            "focal_af": data.frequencies[focals].astype(np.float64),
            "max_mismatches": np.tile(
                np.arange(budgets, dtype=np.int64), len(data.focals)
            ),
            "left_site_index": bounds.left.ravel(),
            "right_site_index": bounds.right.ravel(),
            "left_position": bp.left.ravel(),
            "right_position": bp.right.ravel(),
        }
    )


def construct_focal_ancestor_chunks(
    samples_path: pathlib.Path,
    ancestors_path: pathlib.Path,
    focal_ancestors_path: pathlib.Path,
    ancestral_state: dict,
    max_ac_cutoff: int,
    max_mismatches: int,
    threads: int,
    sample_selection: str | None = None,
) -> pd.DataFrame:
    """Compare each carried focal site with its inferred ancestor haplotype.

    Return one row per NPZ haplotype, eligible ancestor, carried focal site, and
    budget 0..max_mismatches, in that order. Counts use an inclusive maximum AC.
    Each direction tolerates k called mismatches; the anchored half-open interval
    can contain up to 2k in total. Missing calls terminate extension. Comparisons
    use the panel site axis and canonical ancestral/derived codes, never an HMM.
    See :func:`_sweep_block` for a worked example and :func:`_positions` for BP
    conversion. Dataset and panel identity are prepended by the workflow rule.
    """
    for name, value, minimum in (
        ("max_ac_cutoff", max_ac_cutoff, 1),
        ("max_mismatches", max_mismatches, 0),
        ("threads", threads, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    started = time.perf_counter()
    arguments = (
        samples_path,
        ancestors_path,
        focal_ancestors_path,
        ancestral_state,
        max_ac_cutoff,
        sample_selection,
    )
    data = _prepare(*arguments)
    prepared = time.perf_counter()
    bounds = _collect_bounds(data, arguments, max_mismatches, threads)
    compared = time.perf_counter()
    dataframe = _dataframe(data, bounds, max_mismatches)
    logger.info(
        "Constructed %d rows from %d seeds: setup %.3fs, sweeps %.3fs, "
        "assembly %.3fs, total %.3fs, workers %d",
        len(dataframe),
        len(data.focals),
        prepared - started,
        compared - prepared,
        time.perf_counter() - compared,
        time.perf_counter() - started,
        min(threads, len(data.block_seeds)),
    )
    return dataframe
