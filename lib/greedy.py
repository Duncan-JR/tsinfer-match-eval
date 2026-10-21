"""Exact seeded extension and minimum-run covers on the inference-site axis.

Catalog arrays are validated by :mod:`lib.haplotypes`. Kernels use worker-owned
scratch; callers stable-sort candidate indices between extension and covering.
"""

import dataclasses

import numba
import numpy as np

from . import haplotypes


@dataclasses.dataclass
class ExactState:
    """Exact-only mapped input and private scratch, shared by both callers."""

    catalog: haplotypes.ReferenceCatalog
    packed: np.ndarray
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
    scratch = [np.empty(capacity, dtype=np.int64) for _ in range(6)]
    return ExactState(catalog, catalog.packed.open(), *scratch)


def candidates(haplotype, state):
    """Extend and stable-sort once, before covering one or several minima."""
    catalog = state.catalog
    count = extend_exact(
        haplotype,
        state.packed,
        catalog.seeds,
        catalog.starts,
        catalog.stops,
        catalog.offsets,
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
def extend_exact(haplotype, packed, seeds, starts, stops, offsets, left, right, ids):
    """Discover derived seeds and extend locally within their clipped supports."""
    count = 0
    for ancestor in range(len(seeds)):
        seed = seeds[ancestor]
        if haplotype[seed] != 1:
            continue
        start = starts[ancestor]
        stop = stops[ancestor]
        offset = offsets[ancestor] - start
        lo = seed
        while lo > start:
            site = lo - 1
            call = haplotype[site]
            if call < 0 or call != packed[offset + site]:
                break
            lo = site
        hi = seed + 1
        while hi < stop:
            call = haplotype[hi]
            if call < 0 or call != packed[offset + hi]:
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
