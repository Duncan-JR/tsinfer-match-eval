"""Exact covers and supported one-site gap copying on the inference-site axis.

Catalog arrays are validated by :mod:`lib.haplotypes`. Kernels use worker-owned
scratch; callers stable-sort candidate indices between extension and covering.
"""

import dataclasses

import numba
import numpy as np

from . import haplotypes


@dataclasses.dataclass
class ExactState:
    """Mapped input, support lookup and private scratch shared by both callers."""

    catalog: haplotypes.ReferenceCatalog
    packed: np.ndarray
    support_order: np.ndarray
    support_starts: np.ndarray
    prefix_best: np.ndarray
    candidate_left: np.ndarray
    candidate_right: np.ndarray
    candidate_ids: np.ndarray
    run_left: np.ndarray
    run_right: np.ndarray
    run_ids: np.ndarray


def make_state(catalog, packed=None):
    """Allocate private scratch around immutable catalog and packed calls."""
    if packed is None:
        packed = catalog.packed.open()
    capacity = len(catalog.seeds)
    support_order = np.argsort(catalog.starts, kind="stable")
    support_starts = catalog.starts[support_order]
    support_order.flags.writeable = False
    support_starts.flags.writeable = False
    prefix_best = np.empty(capacity, dtype=np.int64)
    scratch = [np.empty(capacity, dtype=np.int64) for _ in range(6)]
    return ExactState(
        catalog,
        packed,
        support_order,
        support_starts,
        prefix_best,
        *scratch,
    )


def candidates(haplotype, state, start, end):
    """Prepare activated supports and exact intervals once for all minima."""
    catalog = state.catalog
    support_prefix(
        haplotype, catalog.seeds, catalog.stops, state.support_order, state.prefix_best
    )
    count = extend_exact(
        haplotype,
        state.packed,
        catalog.seeds,
        catalog.starts,
        catalog.stops,
        catalog.offsets,
        start,
        end,
        state.candidate_left,
        state.candidate_right,
        state.candidate_ids,
    )
    return np.argsort(state.candidate_left[:count], kind="stable")


def cover(state, order, minimum):
    """Write a cover into reusable scratch, returning its number of runs."""
    return cover_components(
        state.candidate_left,
        state.candidate_right,
        state.candidate_ids,
        order,
        minimum,
        state.run_left,
        state.run_right,
        state.run_ids,
    )


@numba.njit(cache=True, nogil=True)
def support_prefix(haplotype, seeds, stops, order, prefix_best):
    """Keep the first activated ancestor with furthest stop in support order."""
    best = -1
    for index in range(len(order)):
        ancestor = order[index]
        if haplotype[seeds[ancestor]] == 1:
            if best == -1 or stops[ancestor] > stops[best]:
                best = ancestor
        prefix_best[index] = best


@numba.njit(cache=True, nogil=True)
def extend_exact(
    haplotype, packed, seeds, starts, stops, offsets, start, end, left, right, ids
):
    """Skip missing observations, stopping at disagreements or unknown references.

    Intersect maximum supports with the sample domain before extension so
    terminal missing calls cannot contribute to a minimum-size component.
    """
    count = 0
    for ancestor in range(len(seeds)):
        seed = seeds[ancestor]
        if haplotype[seed] != 1:
            continue
        lo_bound = max(starts[ancestor], start)
        hi_bound = min(stops[ancestor], end)
        offset = offsets[ancestor] - starts[ancestor]
        lo = seed
        while lo > lo_bound:
            site = lo - 1
            sample_call = haplotype[site]
            ancestor_call = packed[offset + site]
            if ancestor_call < 0:
                break
            if sample_call >= 0 and sample_call != ancestor_call:
                break
            lo = site
        hi = seed + 1
        while hi < hi_bound:
            sample_call = haplotype[hi]
            ancestor_call = packed[offset + hi]
            if ancestor_call < 0:
                break
            if sample_call >= 0 and sample_call != ancestor_call:
                break
            hi += 1
        left[count] = lo
        right[count] = hi
        ids[count] = ancestor
        count += 1
    return count


@numba.njit(cache=True, nogil=True)
def cover_components(left, right, ids, order, minimum, out_left, out_right, out_ids):
    """Cover each touching union with fewest runs, rejecting short whole unions.

    Every candidate is consumed once. Ties keep the first interval in stable
    left order; consumed alternatives cannot extend beyond the chosen reach.
    """
    scan = 0
    written = 0
    while scan < len(order):
        component_start = left[order[scan]]
        cursor = component_start
        saved = written
        while True:
            reach = cursor
            chosen = -1
            while scan < len(order) and left[order[scan]] <= cursor:
                candidate = order[scan]
                if right[candidate] > reach:
                    reach = right[candidate]
                    chosen = ids[candidate]
                scan += 1
            if chosen == -1:
                break
            out_left[written] = cursor
            out_right[written] = reach
            out_ids[written] = chosen
            written += 1
            cursor = reach
        if cursor - component_start < minimum:
            written = saved
    return written


@dataclasses.dataclass
class IntervalPartition:
    """A complete domain partition and the original neighbours of each gap."""

    left: np.ndarray
    right: np.ndarray
    ancestor: np.ndarray
    left_neighbour: np.ndarray
    right_neighbour: np.ndarray


@dataclasses.dataclass
class OneSiteCopies:
    corrections: np.ndarray
    sites: int


@dataclasses.dataclass
class CopiedIntervals:
    """Coalesced copies and boundary-split residual gaps."""

    left: np.ndarray
    right: np.ndarray
    ancestor: np.ndarray


@dataclasses.dataclass
class ResidualWindows:
    windows: np.ndarray
    gap_left: np.ndarray
    gap_right: np.ndarray
    focal_sites: int
    focal_runs: int


def partition_intervals(state, count, start, end, boundaries):
    """Partition copied spans and gaps without changing copying decisions."""
    arrays = _partition_intervals(
        state.run_left, state.run_right, state.run_ids, count, start, end, boundaries
    )
    return IntervalPartition(*arrays)


@numba.njit(cache=True, nogil=True)
def _partition_intervals(run_left, run_right, run_ids, count, start, end, boundaries):
    capacity = 2 * count + len(boundaries) + 2
    left = np.empty(capacity, dtype=np.int64)
    right = np.empty(capacity, dtype=np.int64)
    ancestor = np.empty(capacity, dtype=np.int64)
    left_neighbour = np.empty(capacity, dtype=np.int64)
    right_neighbour = np.empty(capacity, dtype=np.int64)
    n = 0
    cursor = start
    previous = -1
    for run in range(count + 1):
        following = -1
        gap_end = end
        if run < count:
            following = run_ids[run]
            gap_end = run_left[run]
        first = np.searchsorted(boundaries, cursor, side="right")
        last = np.searchsorted(boundaries, gap_end, side="left")
        for boundary in range(first, last + 1):
            stop = gap_end if boundary == last else boundaries[boundary]
            if cursor < stop:
                left[n] = cursor
                right[n] = stop
                ancestor[n] = -1
                left_neighbour[n] = previous
                right_neighbour[n] = following
                n += 1
            cursor = stop
        if run < count:
            left[n] = gap_end
            right[n] = run_right[run]
            ancestor[n] = following
            left_neighbour[n] = -1
            right_neighbour[n] = -1
            n += 1
            cursor = run_right[run]
            previous = following
    return (left[:n], right[:n], ancestor[:n], left_neighbour[:n], right_neighbour[:n])


def resolve_one_site_gaps(state, partition, haplotype):
    """Copy supported one-site gaps using their original neighbours."""
    catalog = state.catalog
    corrections, sites = _resolve_one_site_gaps(
        partition.left,
        partition.right,
        partition.ancestor,
        partition.left_neighbour,
        partition.right_neighbour,
        catalog.starts,
        catalog.stops,
        catalog.offsets,
        state.support_starts,
        state.prefix_best,
        state.packed,
        haplotype,
    )
    return OneSiteCopies(corrections, sites)


@numba.njit(cache=True, nogil=True)
def _resolve_one_site_gaps(
    left,
    right,
    ancestor,
    left_neighbour,
    right_neighbour,
    starts,
    stops,
    offsets,
    support_starts,
    prefix_best,
    packed,
    haplotype,
):
    corrections = np.empty(len(left), dtype=np.int64)
    num_corrections = 0
    sites = 0
    for j in range(len(left)):
        if ancestor[j] >= 0 or right[j] - left[j] != 1:
            continue
        site = left[j]
        chosen = -1
        for candidate in (left_neighbour[j], right_neighbour[j]):
            if candidate >= 0 and starts[candidate] <= site < stops[candidate]:
                offset = offsets[candidate] + site - starts[candidate]
                if packed[offset] >= 0:
                    chosen = candidate
                    break
        if chosen == -1:
            index = np.searchsorted(support_starts, site, side="right") - 1
            if index >= 0:
                candidate = prefix_best[index]
                # The prefix search supplies the start bound. If its best
                # ancestor is unsupported, leave this gap to the HMM.
                if candidate >= 0 and site < stops[candidate]:
                    offset = offsets[candidate] + site - starts[candidate]
                    if packed[offset] >= 0:
                        chosen = candidate
        if chosen >= 0:
            ancestor[j] = chosen
            sites += 1
            offset = offsets[chosen] + site - starts[chosen]
            if haplotype[site] >= 0 and haplotype[site] != packed[offset]:
                corrections[num_corrections] = site
                num_corrections += 1
    return corrections[:num_corrections], sites


def coalesce_copied(partition):
    """Merge compatible copies in place, retaining inference boundaries in gaps."""
    count = _coalesce_copied(partition.left, partition.right, partition.ancestor)
    return CopiedIntervals(
        partition.left[:count], partition.right[:count], partition.ancestor[:count]
    )


@numba.njit(cache=True, nogil=True)
def _coalesce_copied(left, right, ancestor):
    count = 0
    for j in range(len(left)):
        if count > 0 and ancestor[j] >= 0 and ancestor[count - 1] == ancestor[j]:
            right[count - 1] = right[j]
        else:
            left[count] = left[j]
            right[count] = right[j]
            ancestor[count] = ancestor[j]
            count += 1
    return count


def residual_windows(intervals, parents, boundaries, condition_gaps):
    """Construct native windows and pins from the complete partition."""
    arrays = _residual_windows(
        intervals.left,
        intervals.right,
        intervals.ancestor,
        parents,
        boundaries,
        condition_gaps,
    )
    return ResidualWindows(*arrays)


@numba.njit(cache=True, nogil=True)
def _residual_windows(left, right, ancestor, parents, boundaries, condition_gaps):
    n = len(left)
    windows = np.empty((n, 4), dtype=np.int32)
    gap_left = np.empty(n, dtype=np.int64)
    gap_right = np.empty(n, dtype=np.int64)
    num_gaps = 0
    focal_sites = 0
    focal_runs = 0
    for j in range(n):
        if ancestor[j] >= 0:
            focal_sites += right[j] - left[j]
            focal_runs += 1
            continue
        initial = -1
        final = -1
        if condition_gaps:
            boundary = np.searchsorted(boundaries, left[j])
            at_boundary = boundary < len(boundaries) and boundaries[boundary] == left[j]
            if j > 0 and ancestor[j - 1] >= 0 and not at_boundary:
                initial = parents[ancestor[j - 1]]
            boundary = np.searchsorted(boundaries, right[j])
            at_boundary = boundary < len(boundaries) and boundaries[boundary] == right[j]
            if j + 1 < n and ancestor[j + 1] >= 0 and not at_boundary:
                final = parents[ancestor[j + 1]]
        windows[num_gaps, 0] = left[j] - int(initial != -1)
        windows[num_gaps, 1] = right[j] + int(final != -1)
        windows[num_gaps, 2] = initial
        windows[num_gaps, 3] = final
        gap_left[num_gaps] = left[j]
        gap_right[num_gaps] = right[j]
        num_gaps += 1
    return (
        windows[:num_gaps],
        gap_left[:num_gaps],
        gap_right[:num_gaps],
        focal_sites,
        focal_runs,
    )


@numba.njit(cache=True, nogil=True)
def assemble_intervals(
    interval_left,
    interval_right,
    interval_ancestor,
    parents,
    path_offsets,
    path_left,
    path_right,
    path_parent,
    gap_left,
    gap_right,
    corrections,
    haplotype,
    matched,
    positions,
    sequence_length,
):
    """Clip and canonicalise paths in one pass, then collect gap corrections."""
    capacity = len(interval_left) + len(path_parent)
    left = np.empty(capacity, dtype=np.float64)
    right = np.empty(capacity, dtype=np.float64)
    parent = np.empty(capacity, dtype=np.int32)
    mutation_capacity = np.sum(gap_right - gap_left) + len(corrections)
    mutations = np.empty(mutation_capacity, dtype=np.int64)
    num_mutations = len(corrections)
    mutations[:num_mutations] = corrections
    count = 0
    gap = 0
    for j in range(len(interval_left)):
        if interval_ancestor[j] >= 0:
            a = interval_left[j]
            b = interval_right[j]
            left[count] = 0 if a == 0 else positions[a]
            right[count] = sequence_length if b == len(positions) else positions[b]
            parent[count] = parents[interval_ancestor[j]]
            count += 1
        else:
            first_edge = count
            for k in range(
                int(path_offsets[gap + 1]) - 1, int(path_offsets[gap]) - 1, -1
            ):
                a = max(gap_left[gap], np.searchsorted(positions, path_left[k]))
                b = min(gap_right[gap], np.searchsorted(positions, path_right[k]))
                if a < b:
                    p = 1 if path_parent[k] == 0 else path_parent[k]
                    bp_left = 0 if a == 0 else positions[a]
                    bp_right = sequence_length if b == len(positions) else positions[b]
                    if count > first_edge and parent[count - 1] == p:
                        right[count - 1] = bp_right
                    else:
                        left[count] = bp_left
                        right[count] = bp_right
                        parent[count] = p
                        count += 1
            for site in range(gap_left[gap], gap_right[gap]):
                if haplotype[site] >= 0 and haplotype[site] != matched[site]:
                    mutations[num_mutations] = site
                    num_mutations += 1
            gap += 1
    sites = np.sort(mutations[:num_mutations])
    return left[:count], right[:count], parent[:count], sites, haplotype[sites]
