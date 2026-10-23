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


def make_state(catalog):
    """Open the shared packed reference and allocate process-local scratch."""
    catalog.make_readonly()
    capacity = len(catalog.seeds)
    support_order = np.argsort(catalog.starts, kind="stable")
    support_starts = catalog.starts[support_order]
    support_order.flags.writeable = False
    support_starts.flags.writeable = False
    prefix_best = np.empty(capacity, dtype=np.int64)
    scratch = [np.empty(capacity, dtype=np.int64) for _ in range(6)]
    return ExactState(
        catalog,
        catalog.packed.open(),
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


@dataclasses.dataclass
class MatchInterval:
    """One copied catalog ancestor or independent HMM interval (``None``).

    ``bridged_sites`` records only one-site copying decisions, allowing matching
    to compare their known packed alleles without scanning exact runs again.
    """

    left: int
    right: int
    ancestor: int | None
    bridged_sites: list[int] = dataclasses.field(default_factory=list)


def _known_support(state, ancestor, site):
    catalog = state.catalog
    if ancestor < 0:
        return False
    if site < catalog.starts[ancestor] or site >= catalog.stops[ancestor]:
        return False
    offset = catalog.offsets[ancestor] + site - catalog.starts[ancestor]
    return state.packed[offset] >= 0


def _gap_parent(state, site, left, right):
    """Prefer retained neighbours, then one bounded activated-support lookup."""
    for ancestor in (left, right):
        if _known_support(state, ancestor, site):
            return int(ancestor)
    index = np.searchsorted(state.support_starts, site, side="right") - 1
    if index >= 0:
        ancestor = state.prefix_best[index]
        if _known_support(state, ancestor, site):
            return int(ancestor)
    return None


def _partition_intervals(state, count, start, end, boundaries):
    """Split only uncovered intervals; neighbours always come from the cover."""
    cursor = start
    left_parent = -1
    for run in range(count + 1):
        right_parent = -1
        gap_end = end
        if run < count:
            right_parent = int(state.run_ids[run])
            gap_end = int(state.run_left[run])
        first = np.searchsorted(boundaries, cursor, side="right")
        last = np.searchsorted(boundaries, gap_end, side="left")
        for stop in [*boundaries[first:last], gap_end]:
            stop = int(stop)
            if cursor < stop:
                ancestor = None
                bridged = []
                if stop - cursor == 1:
                    ancestor = _gap_parent(state, cursor, left_parent, right_parent)
                    if ancestor is not None:
                        bridged = [cursor]
                yield MatchInterval(cursor, stop, ancestor, bridged)
            cursor = stop
        if run < count:
            cursor = int(state.run_right[run])
            yield MatchInterval(gap_end, cursor, right_parent)
            left_parent = right_parent


def stitched_intervals(state, count, start, end, boundaries):
    """Emit ordered, disjoint copying/HMM decisions without modifying cover scratch.

    Hold one pending interval to extend a chosen neighbour's endpoint. Catalog
    supports cannot cross inference boundaries, so equal copied ancestors can
    be joined safely. HMM pieces remain independent at every boundary.
    """
    pending = None
    for interval in _partition_intervals(state, count, start, end, boundaries):
        if pending is not None:
            same_parent = (
                pending.ancestor is not None and pending.ancestor == interval.ancestor
            )
            if same_parent and pending.right == interval.left:
                pending.right = interval.right
                pending.bridged_sites.extend(interval.bridged_sites)
                continue
            yield pending
        pending = interval
    if pending is not None:
        yield pending


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
