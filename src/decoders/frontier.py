"""Approximate logical odds from a sparse graph of detector pairs and boundaries."""

import numpy as np
from numba import njit
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import splu


MAX_HITS = 60
MAX_WIDTH = 12


def path_model(model, n):
    """Sum projected graph walks in each logical class for one logical observable.

    A pair weight sums arbitrary walks between two detectors. A boundary weight
    sums walks terminated by a boundary edge. Their products approximate a fault
    likelihood: different walks may share edges and the projection loses fault
    correlations. Return None if the walk expansion or logical labels are unsupported.
    """
    if (model.logical_offset or np.any(model.detector_offset)
            or any(f.logical > 1 or (f.logical and not f.detectors) for f in model.faults)
            or any(e.logical > 1 for e in model.edges)):
        return None
    adjacent = [[] for _ in range(n)]
    live = set()
    for e in model.edges:
        if e.logical:
            live.update(e.detectors)
        if len(e.detectors) == 2:
            u, v = e.detectors
            adjacent[u].append(v)
            adjacent[v].append(u)
    stack = list(live)
    while stack:
        for v in adjacent[stack.pop()]:
            if v not in live:
                stack.append(v)
                live.add(v)
    keep = np.array(sorted(live), dtype=np.int32)
    m = len(keep)
    if not 0 < m <= 1024:
        return None

    # Lift each detector to two vertices representing the accumulated logical XOR.
    remap = {d: i for i, d in enumerate(keep)}
    rows, cols, odds = [], [], []
    exits = np.zeros((2 * m, 2))
    for e in model.edges:
        if e.detectors[0] not in live:
            continue
        weight = e.probability / (1 - e.probability)
        i = remap[e.detectors[0]]
        for sector in (0, 1):
            u = i + sector * m
            parity = sector ^ e.logical
            if len(e.detectors) == 1:
                exits[u, parity] += weight
            else:
                v = remap[e.detectors[1]] + parity * m
                rows.extend((u, v))
                cols.extend((v, u))
                odds.extend((weight, weight))
    adjacency = coo_matrix((odds, (rows, cols)), shape=(2*m, 2*m)).tocsc()
    if adjacency.sum(axis=1).max() >= .9:
        return None
    # (I-A)^-1 = I + A + A^2 + ... under the row-sum bound above.
    green = splu(eye(2*m, format="csc") - adjacency).solve(np.eye(2*m))
    return (keep, np.ascontiguousarray(green[:m, :m]), np.ascontiguousarray(green[:m, m:]),
            np.ascontiguousarray((green @ exits)[:m]))


@njit(cache=True)
def _build_pairs(hits, count, even, odd, boundary, cutoff, first, neighbors, degree, affinity, scale):
    """Keep pair edges relative to the strongest local weights at both endpoints."""
    for i in range(count):
        u = hits[i]
        scale[i] = max(1e-40, boundary[u, 0] + boundary[u, 1])
        first[i] = i
        degree[i] = 0
    for i in range(count):
        u = hits[i]
        for j in range(i):
            v = hits[j]
            w = even[u, v] + odd[u, v]
            affinity[j, i] = w
            root = np.sqrt(w)
            scale[i] = max(scale[i], root)
            scale[j] = max(scale[j], root)
    for i in range(count):
        for j in range(i+1, count):
            if affinity[i, j] > cutoff * scale[i] * scale[j]:
                neighbors[i, degree[i]] = j
                degree[i] += 1
                first[j] = min(first[j], i)


@njit(cache=True)
def _contract(hits, count, even, odd, boundary, width, first, neighbors, degree, work):
    """Contract a matching partition function, returning even/odd logical masses.

    Scan backwards. A frontier contains consumed vertices still connected to the
    unprocessed prefix. A set bit reserves that vertex for a pair from the prefix.
    Each uncovered vertex either terminates at a boundary or pairs with exactly
    one available neighbor. Multiplication convolves the two logical XOR classes.
    """
    previous_sites, next_sites, slot, old_bits, neighbor_bits, previous, current, reindex, bit_index = work
    # Each site j occupies a slot from first[j]+1 through j. Detect a
    # width overflow before spending time on a partial contraction.
    slot[:count] = 0
    for j in range(count):
        if first[j] < j:
            slot[first[j] + 1] += 1
            if j + 1 < count:
                slot[j + 1] -= 1
    occupied = 0
    for j in range(count):
        occupied += slot[j]
        if occupied > width:
            return 0., 0.
    previous[0, 0], previous[0, 1] = 1., 0.
    active = 0
    for i in range(count-1, -1, -1):
        u = hits[i]
        reserve_i = first[i] < i
        size = 0
        if reserve_i:
            next_sites[0] = i
            old_bits[0] = 0
            size = 1
        for k in range(degree[i]):
            neighbor_bits[k] = 1 << slot[neighbors[i,k]]
        for k in range(active):
            site = previous_sites[k]
            if first[site] < i:
                if size == width:
                    return 0., 0.
                next_sites[size] = site
                old_bits[size] = 1 << k
                size += 1
        for k in range(size):
            slot[next_sites[k]] = k

        largest = 0.
        for mask in range(1 << size):
            # Project the new frontier's reservation mask back into the old bit order.
            mapped = 0
            if mask:
                bit = mask & -mask
                mapped = reindex[mask ^ bit] | old_bits[bit_index[bit]]
            reindex[mask] = mapped
            if reserve_i and mask & 1:
                a, b = previous[mapped, 0], previous[mapped, 1]
            else:
                a = boundary[u, 0]*previous[mapped, 0] + boundary[u, 1]*previous[mapped, 1]
                b = boundary[u, 0]*previous[mapped, 1] + boundary[u, 1]*previous[mapped, 0]
                for k in range(degree[i]):
                    bit = neighbor_bits[k]
                    if not (mapped & bit):
                        v = hits[neighbors[i, k]]
                        paired = mapped | bit
                        a += even[u, v]*previous[paired, 0] + odd[u, v]*previous[paired, 1]
                        b += even[u, v]*previous[paired, 1] + odd[u, v]*previous[paired, 0]
            current[mask, 0], current[mask, 1] = a, b
            largest = max(largest, a, b)
        # A common rescaling preserves all ratios, including across frontier states.
        # If every nonzero path is already lost, leave the shot to the fallback.
        if largest == 0.:
            return 0., 0.
        if largest < 1e-80 or largest > 1e80:
            for k in range(1 << size):
                current[k, 0] /= largest
                current[k, 1] /= largest
        active = size
        previous, current = current, previous
        previous_sites, next_sites = next_sites, previous_sites
    return previous[0, 0], previous[0, 1]


@njit(cache=True)
def frontier_screen(dets, keep, even, odd, boundary, width, cutoff, certainty):
    """Return logical decisions and a mask of shots requiring fallback matching."""
    if width < 1 or width > MAX_WIDTH:
        raise ValueError("frontier width must be from 1 to 12")
    if not np.isfinite(cutoff) or cutoff < 0 or not np.isfinite(certainty) or certainty < 1:
        raise ValueError("cutoff must be nonnegative and the odds threshold at least one")
    answer = np.zeros(len(dets), np.bool_)
    pending = np.ones(len(dets), np.bool_)
    hits = np.empty(MAX_HITS, np.int32)
    first = np.empty_like(hits)
    degree = np.empty_like(hits)
    neighbors = np.empty((MAX_HITS, MAX_HITS), np.int32)
    affinity = np.empty((MAX_HITS, MAX_HITS))
    scale = np.empty(MAX_HITS)
    masks = 1 << width
    bit_index = np.empty(masks, np.int32)
    for k in range(width):
        bit_index[1 << k] = k
    work = (np.empty_like(hits), np.empty_like(hits), np.empty_like(hits),
            np.empty(width, np.int32), np.empty_like(hits), np.empty((masks, 2)),
            np.empty((masks, 2)), np.empty(masks, np.int32), bit_index)
    for shot, row in enumerate(dets):
        count = 0
        for d in range(len(keep)):
            if row[keep[d]]:
                if count == MAX_HITS:
                    count += 1
                    break
                hits[count] = d
                count += 1
        if count > MAX_HITS:
            continue
        _build_pairs(hits, count, even, odd, boundary, cutoff, first, neighbors, degree, affinity, scale)
        a, b = _contract(hits, count, even, odd, boundary, width, first, neighbors, degree, work)
        if max(a, b) > certainty * min(a, b):
            pending[shot] = False
            answer[shot] = b > a
    return answer, pending
