"""Direct inferred focal-ancestor comparisons, independent of HMM matching."""

import dataclasses
import logging
import multiprocessing
import pathlib

import numba
import numpy as np
import tsinfer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ComparisonData:
    """Validated site axis, resident sample calls, and canonical association order."""

    panel: object
    num_sites: int
    num_ancestors: int
    calls: np.ndarray
    sample_ids: np.ndarray
    ploidy_indices: np.ndarray
    offsets: np.ndarray
    focal_ac: np.ndarray
    haplotypes: np.ndarray
    ancestors: np.ndarray
    focals: np.ndarray
    left: np.ndarray
    right: np.ndarray
    column_width: int
    site_height: int
    block_seeds: dict


@dataclasses.dataclass
class IntervalBounds:
    """Canonical bounds for every association and generated budget."""

    left: np.ndarray
    right: np.ndarray


@dataclasses.dataclass
class BlockBounds:
    """Numeric block results indexed by the canonical global seed number."""

    seeds: np.ndarray
    left: np.ndarray
    right: np.ndarray


def _prepare(
    samples_path,
    ancestors_path,
    focal_path,
    ancestral_state,
    max_ac_cutoff,
    sample_selection,
):
    """Prepare leftmost associations for :func:`construct_focal_ancestor_intervals`."""
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
        ancestor_focals = np.full(len(ancestor_ids), -1, dtype=np.int64)
        for ancestor, values in enumerate(focal_positions):
            values = values[values >= 0]
            if len(values) == 0:
                continue
            position = np.min(values)
            index = np.searchsorted(positions, position)
            if index >= len(positions) or positions[index] != position:
                raise ValueError("Ancestor focal position is absent from panel")
            ancestor_focals[ancestor] = index
        haplotypes = []
        ancestors = []
        focals = []
        output_offsets = [0]
        for haplotype in range(len(sample_ids)):
            selected = columns[offsets[haplotype] : offsets[haplotype + 1]]
            if np.any(np.diff(selected) <= 0):
                raise ValueError("Candidate columns must be sorted and distinct")
            selected = selected[derived_ac[selected] <= max_ac_cutoff]
            for ancestor in selected:
                index = ancestor_focals[ancestor]
                if index == -1:
                    continue
                if counts[index] != derived_ac[ancestor]:
                    raise ValueError("Focal NPZ counts disagree with sample annotations")
                if calls[index, haplotype] != 1:
                    raise ValueError(
                        "Selected leftmost focal sample call is not derived"
                    )
                haplotypes.append(haplotype)
                ancestors.append(ancestor)
                focals.append(index)
            output_offsets.append(len(focals))
        focal_ac = derived_ac[ancestors].astype(np.int64)
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
        panel=panel,
        num_sites=len(positions),
        num_ancestors=len(ancestor_ids),
        calls=calls,
        sample_ids=sample_ids,
        ploidy_indices=ploidy_indices.astype(np.int64),
        offsets=np.asarray(output_offsets, dtype=np.int64),
        focal_ac=focal_ac,
        haplotypes=haplotypes,
        ancestors=ancestors,
        focals=focals,
        left=left.astype(np.int64),
        right=right.astype(np.int64),
        column_width=column_width,
        site_height=site_height,
        block_seeds=block_seeds,
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

    Used by :func:`construct_focal_ancestor_intervals`. Bounds are half-open;
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
    column_end = min(column_start + data.column_width, data.num_ancestors)
    columns = data.ancestors[seeds] - column_start
    haplotypes = data.haplotypes[seeds]
    focals = data.focals[seeds]
    order = np.argsort(focals, kind="stable")
    left = np.repeat(data.left[seeds, None], max_mismatches + 1, axis=1)
    right = np.repeat(data.right[seeds, None], max_mismatches + 1, axis=1)
    genotypes = data.panel["call_genotype"]
    num_sites = data.num_sites
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
    """Give each spawned process its own read-only stores and numeric inputs."""
    global _worker_data, _worker_max_mismatches
    _worker_data = _prepare(*arguments)
    _worker_max_mismatches = max_mismatches


def _worker_block(block):
    return _block_bounds(_worker_data, block, _worker_max_mismatches)


def _collect_bounds(data, arguments, max_mismatches, threads):
    """Assemble block results in canonical association order; propagate failures."""
    shape = (len(data.focals), max_mismatches + 1)
    left = np.empty(shape, dtype=np.int64)
    right = np.empty(shape, dtype=np.int64)
    workers = min(threads, len(data.block_seeds))
    if workers <= 1:
        for block in data.block_seeds:
            result = _block_bounds(data, block, max_mismatches)
            left[result.seeds] = result.left
            right[result.seeds] = result.right
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            workers,
            initializer=_initialise_worker,
            initargs=(arguments, max_mismatches),
        ) as pool:
            results = pool.imap_unordered(_worker_block, data.block_seeds, chunksize=1)
            for result in results:
                left[result.seeds] = result.left
                right[result.seeds] = result.right
    return IntervalBounds(left, right)


def construct_focal_ancestor_intervals(
    samples_path: pathlib.Path,
    ancestors_path: pathlib.Path,
    focal_ancestors_path: pathlib.Path,
    output_path: pathlib.Path,
    ancestral_state: dict,
    max_ac_cutoff: int,
    max_mismatches: int,
    threads: int,
    sample_selection: str | None = None,
) -> None:
    """Write compressed raw intervals anchored at each ancestor's leftmost focal.

    One association per eligible ancestor is retained in focal-NPZ haplotype
    order, then increasing ancestor column. Exact AC uses an inclusive maximum.
    Budget column k tolerates k called mismatches on each side; the matching
    derived focal consumes neither budget. Missing calls terminate extension.
    Bounds are half-open on the inferred panel axis and confined to ancestor
    support and the containing inference interval. See :func:`_sweep_block` for
    a worked example. No HMM or coverage statistics enter this comparison.
    """
    for name, value, minimum in (
        ("max_ac_cutoff", max_ac_cutoff, 1),
        ("max_mismatches", max_mismatches, 0),
        ("threads", threads, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    arguments = (
        samples_path,
        ancestors_path,
        focal_ancestors_path,
        ancestral_state,
        max_ac_cutoff,
        sample_selection,
    )
    data = _prepare(*arguments)
    workers = min(threads, len(data.block_seeds))
    logger.info("Comparing %d associations with %d workers", len(data.focals), workers)
    bounds = _collect_bounds(data, arguments, max_mismatches, threads)
    if np.any(
        (bounds.left < 0)
        | (bounds.left > data.focals[:, None])
        | (bounds.right <= data.focals[:, None])
        | (bounds.right > data.num_sites)
    ):
        raise ValueError(
            "Interval bounds must contain the chosen focal on the panel axis"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sample_id=data.sample_ids,
        ploidy_index=data.ploidy_indices,
        offsets=data.offsets,
        focal_ac=data.focal_ac,
        left_site_index=bounds.left,
        right_site_index=bounds.right,
        num_sites=np.int64(data.num_sites),
        max_ac_cutoff=np.int64(max_ac_cutoff),
        max_mismatches=np.int64(max_mismatches),
    )
    logger.info("Wrote %d associations to %s", len(data.focals), output_path)
