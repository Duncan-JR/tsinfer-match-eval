"""Inferred focal candidates, direct intervals, and their coverage masks."""

import concurrent.futures as cf
import dataclasses
import itertools
import logging
import pathlib

import numba
import numpy as np
import tsinfer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SampleHaplotypes:
    """Panel-aligned canonical calls and identities in sample-major order."""

    calls: np.ndarray
    sample_id: np.ndarray
    ploidy_index: np.ndarray
    sample_rows: np.ndarray


def _load_sample_haplotypes(samples, panel, sample_selection, site_indices=None):
    """Polarise sample allele strings using the inferred panel's ancestry.

    Shared by :func:`find_focal_ancestors` and interval :func:`_prepare`.
    Check the cross-store position mapping at this loading boundary. Optional
    ``site_indices`` restrict calls and polarisation to selected panel sites.
    """
    positions = panel["variant_position"][:]
    if site_indices is not None:
        positions = positions[site_indices]
    sample_positions = samples["variant_position"][:]
    rows = np.searchsorted(sample_positions, positions)
    present = rows < len(sample_positions)
    present[present] = sample_positions[rows[present]] == positions[present]
    if not np.all(present):
        absent = positions[np.flatnonzero(~present)[0]]
        raise ValueError(f"Panel position {absent} is absent from the sample store")
    columns = tsinfer.vcz.resolve_samples_selection(samples, sample_selection)
    genotypes = samples["call_genotype"].oindex[rows, columns, :]
    alleles = samples["variant_allele"].oindex[rows, :]
    if site_indices is None:
        ancestral = panel["variant_allele"][:, 0]
    else:
        ancestral = panel["variant_allele"].oindex[site_indices, 0]
    called = genotypes >= 0
    allele_indices = np.maximum(genotypes, 0)
    called_alleles = np.take_along_axis(alleles[:, None, :], allele_indices, axis=2)
    derived = called_alleles != ancestral[:, None, None]
    calls = np.where(called, derived, -1).astype(np.int8)
    ploidy = genotypes.shape[2]
    num_haplotypes = genotypes.shape[1] * ploidy
    calls = calls.reshape(len(positions), num_haplotypes)
    ids = np.asarray(samples["sample_id"].oindex[columns].tolist(), dtype=str)
    sample_id = np.repeat(ids, ploidy)
    ploidy_index = np.tile(np.arange(ploidy, dtype=np.int64), len(ids))
    return SampleHaplotypes(calls, sample_id, ploidy_index, rows)


def _select_focal_sites(panel, focal_choice):
    """Select one endpoint seed per ancestor for :func:`find_focal_ancestors`."""
    positions = panel["variant_position"][:]
    focal_positions = panel["sample_focal_positions"][:]
    sites = np.full(len(focal_positions), -1, dtype=np.int64)
    for ancestor, values in enumerate(focal_positions):
        values = values[values >= 0]
        if len(values) == 0:
            continue
        position = np.min(values) if focal_choice == "left" else np.max(values)
        sites[ancestor] = np.searchsorted(positions, position)
    return sites


def find_focal_ancestors(
    samples_path: pathlib.Path,
    ancestors_path: pathlib.Path,
    output_path: pathlib.Path,
    focal_choice: str = "left",
    sample_selection: str | None = None,
) -> None:
    """Write inferred candidates defined by a derived call at one endpoint seed.

    Choose the smallest focal position for ``left`` and largest for ``right``.
    Other focal calls do not affect membership. Save all candidates, without AC
    filtering, in sorted panel-column order for each haplotype, including empty
    rows. :func:`make_ancestor_intervals` reuses these saved seeds.
    Direct callers supply a valid choice; the Snakefile validates configuration.
    """
    samples = tsinfer.vcz.open_store(samples_path)
    panel = tsinfer.vcz.open_store(ancestors_path)
    focal_sites = _select_focal_sites(panel, focal_choice)
    ancestor_ids = np.asarray(panel["sample_id"][:].tolist(), dtype=str)
    anchored = np.flatnonzero(focal_sites >= 0)
    seed_sites, seed_indices = np.unique(focal_sites[anchored], return_inverse=True)
    haplotypes = _load_sample_haplotypes(
        samples, panel, sample_selection, site_indices=seed_sites
    )
    seed_rows = haplotypes.sample_rows[seed_indices]
    derived_ac = np.zeros(len(ancestor_ids), dtype=np.int64)
    derived_ac[anchored] = samples["variant_match_eval_derived_ac"].oindex[seed_rows]
    offsets = [0]
    candidate_rows = []
    for haplotype in range(len(haplotypes.sample_id)):
        seed_calls = haplotypes.calls[seed_indices, haplotype]
        candidates = anchored[seed_calls == 1]
        candidate_rows.append(candidates)
        offsets.append(offsets[-1] + len(candidates))
    ancestor_index = np.empty(0, dtype=np.int64)
    if len(candidate_rows) > 0:
        ancestor_index = np.concatenate(candidate_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sample_id=haplotypes.sample_id,
        ploidy_index=haplotypes.ploidy_index,
        ancestor_id=ancestor_ids,
        offsets=np.asarray(offsets, dtype=np.int64),
        ancestor_index=ancestor_index,
        derived_ac=derived_ac,
        focal_site_index=focal_sites,
        focal_choice=np.asarray(focal_choice),
    )
    logger.info(
        "Wrote focal candidates for %d haplotypes to %s",
        len(haplotypes.sample_id),
        output_path,
    )


@dataclasses.dataclass
class ComparisonData:
    """Panel axis, resident sample calls, and canonical association order."""

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
    max_ac_cutoff,
    sample_selection,
):
    """Load and filter saved seed associations for interval comparison."""
    samples = tsinfer.vcz.open_store(samples_path)
    panel = tsinfer.vcz.open_store(ancestors_path)
    loaded = _load_sample_haplotypes(samples, panel, sample_selection)
    positions = panel["variant_position"][:]
    ancestor_ids = np.asarray(panel["sample_id"][:].tolist(), dtype=str)
    starts = panel["sample_start_position"][:]
    ends = panel["sample_end_position"][:]
    intervals = panel["sequence_intervals"][:]
    with np.load(focal_path, allow_pickle=False) as candidates:
        for name, expected in (
            ("sample_id", loaded.sample_id),
            ("ploidy_index", loaded.ploidy_index),
            ("ancestor_id", ancestor_ids),
        ):
            if not np.array_equal(candidates[name], expected):
                raise ValueError(f"Focal NPZ {name} disagrees with store identity/order")
        offsets = candidates["offsets"]
        columns = candidates["ancestor_index"]
        derived_ac = candidates["derived_ac"]
        ancestor_focals = candidates["focal_site_index"]
        haplotype_rows = []
        ancestor_rows = []
        output_offsets = [0]
        for haplotype in range(len(loaded.sample_id)):
            selected = columns[offsets[haplotype] : offsets[haplotype + 1]]
            selected = selected[derived_ac[selected] <= max_ac_cutoff]
            haplotype_rows.append(np.full(len(selected), haplotype, dtype=np.int64))
            ancestor_rows.append(selected)
            output_offsets.append(output_offsets[-1] + len(selected))
        haplotypes = np.empty(0, dtype=np.int64)
        ancestors = np.empty(0, dtype=np.int64)
        if len(ancestor_rows) > 0:
            haplotypes = np.concatenate(haplotype_rows)
            ancestors = np.concatenate(ancestor_rows)
        focals = ancestor_focals[ancestors]
        focal_ac = derived_ac[ancestors]
    # Each seed is confined to the inference interval containing its focal site.
    interval_indices = np.searchsorted(intervals[:, 0], positions[focals], side="right")
    interval_indices -= 1
    containing = intervals[interval_indices]
    bp_left = np.maximum(starts[ancestors], containing[:, 0]).astype(np.int64)
    bp_right = np.minimum(ends[ancestors], containing[:, 1]).astype(np.int64)
    left = np.searchsorted(positions, bp_left)
    right = np.searchsorted(positions, bp_right)
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
        calls=loaded.calls,
        sample_ids=loaded.sample_id,
        ploidy_indices=loaded.ploidy_index,
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


@numba.njit(nogil=True)
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

    Used by :func:`make_ancestor_intervals`. Bounds are half-open;
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


def _collect_bounds(data, max_mismatches, threads):
    """Run independent block sweeps with shared read-only calls and associations.

    The Numba sweep releases the GIL. Each block owns its state and outputs;
    assemble results in canonical association order and propagate failures.
    """
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be an integer >= 1")
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
        with cf.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_block_bounds, data, block, max_mismatches): block
                for block in data.block_seeds
            }
            for future in cf.as_completed(futures):
                result = future.result()
                left[result.seeds] = result.left
                right[result.seeds] = result.right
                del futures[future]
    return IntervalBounds(left, right)


def make_ancestor_intervals(
    samples_path: pathlib.Path,
    ancestors_path: pathlib.Path,
    focal_ancestors_path: pathlib.Path,
    output_path: pathlib.Path,
    max_ac_cutoff: int,
    max_mismatches: int,
    threads: int,
    focal_choice: str = "left",
    sample_selection: str | None = None,
) -> None:
    """Write compressed raw intervals anchored at the saved selected focal seed.

    One association per eligible ancestor is retained in focal-NPZ haplotype
    order, then increasing ancestor column. Exact AC uses an inclusive maximum.
    Budget column k tolerates k called mismatches on each side; the matching
    derived focal consumes neither budget. Missing calls terminate extension.
    Bounds are half-open on the inferred panel axis and confined to ancestor
    support and the containing inference interval. See :func:`_sweep_block` for
    a worked example. No HMM or coverage statistics enter this comparison.
    Seeds come from :func:`find_focal_ancestors`; ``focal_choice`` records
    provenance only. Direct callers supply valid AC, budget, and choice values.
    """
    data = _prepare(
        samples_path,
        ancestors_path,
        focal_ancestors_path,
        max_ac_cutoff,
        sample_selection,
    )
    workers = min(threads, len(data.block_seeds))
    logger.info("Comparing %d associations with %d workers", len(data.focals), workers)
    bounds = _collect_bounds(data, max_mismatches, threads)
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
        focal_choice=np.asarray(focal_choice),
    )
    logger.info("Wrote %d associations to %s", len(data.focals), output_path)


def make_focal_mask(
    intervals_path: pathlib.Path,
    output_path: pathlib.Path,
    ac_cutoff: int,
    max_mismatches: int,
) -> None:
    """Write one coverage mask; see :func:`make_focal_masks` for batch generation."""
    make_focal_masks(intervals_path, [output_path], [ac_cutoff], [max_mismatches])


def make_focal_masks(
    intervals_path: pathlib.Path,
    output_paths: list[pathlib.Path],
    ac_cutoffs: list[int],
    max_mismatches: list[int],
) -> None:
    """Load intervals once and write every requested cutoff/budget mask.

    ``output_paths`` follows cutoff-major, then budget order, matching the
    Snakefile's expansion. Each output keeps :func:`make_focal_mask`'s format.
    """
    with np.load(intervals_path, allow_pickle=False) as archive:
        intervals = {name: archive[name] for name in archive.files}
    selections = itertools.product(ac_cutoffs, max_mismatches)
    for output_path, (cutoff, budget) in zip(output_paths, selections, strict=True):
        _write_focal_mask(intervals, output_path, cutoff, budget)


def _write_focal_mask(intervals, output_path, ac_cutoff, max_mismatches):
    """Partition the inference-site axis into focal-covered and all-gap runs.

    Union eligible associations from :func:`make_ancestor_intervals`, merging
    touching bounds. Keep its complete haplotype roster, including empty slices.
    The Snakefile supplies an AC cutoff and budget within the generated maxima.
    """
    sample_id = intervals["sample_id"]
    ploidy_index = intervals["ploidy_index"]
    association_offsets = intervals["offsets"]
    focal_ac = intervals["focal_ac"]
    left = intervals["left_site_index"][:, max_mismatches]
    right = intervals["right_site_index"][:, max_mismatches]
    num_sites = int(intervals["num_sites"])
    focal_choice = intervals["focal_choice"]

    offsets = [0]
    run_left = []
    run_right = []
    is_gap = []
    for haplotype in range(len(sample_id)):
        start = association_offsets[haplotype]
        end = association_offsets[haplotype + 1]
        eligible = np.flatnonzero(focal_ac[start:end] <= ac_cutoff) + start
        order = np.argsort(left[eligible], kind="stable")
        covered = []
        for association in eligible[order]:
            interval_left = int(left[association])
            interval_right = int(right[association])
            if len(covered) > 0 and interval_left <= covered[-1][1]:
                covered[-1][1] = max(covered[-1][1], interval_right)
            else:
                covered.append([interval_left, interval_right])
        cursor = 0
        for interval_left, interval_right in covered:
            if cursor < interval_left:
                run_left.append(cursor)
                run_right.append(interval_left)
                is_gap.append(True)
            run_left.append(interval_left)
            run_right.append(interval_right)
            is_gap.append(False)
            cursor = interval_right
        if cursor < num_sites or len(covered) == 0:
            run_left.append(cursor)
            run_right.append(num_sites)
            is_gap.append(True)
        offsets.append(len(run_left))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sample_id=sample_id,
        ploidy_index=ploidy_index,
        offsets=np.asarray(offsets, dtype=np.int64),
        left_site_index=np.asarray(run_left, dtype=np.int64),
        right_site_index=np.asarray(run_right, dtype=np.int64),
        is_gap=np.asarray(is_gap, dtype=bool),
        num_sites=np.int64(num_sites),
        ac_cutoff=np.int64(ac_cutoff),
        max_mismatches=np.int64(max_mismatches),
        focal_choice=focal_choice,
    )
    logger.info("Wrote focal mask for %d haplotypes to %s", len(sample_id), output_path)
