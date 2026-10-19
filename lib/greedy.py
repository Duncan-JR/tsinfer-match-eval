"""Exact seeded extension and minimum-run covers on the inference-site axis.

Catalog arrays are validated by :mod:`lib.haplotypes`. Kernels use worker-owned
scratch; callers stable-sort candidate indices between extension and covering.
"""

import numba


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
