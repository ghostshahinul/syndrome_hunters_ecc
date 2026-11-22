"""Likelihood of a logical correction from local fault evidence and a path sum.

Compute the local joint probabilities for the projected DEM, clip their odds,
then approximate the logical likelihood with independent edges and a sum of
shortest paths over all ways of pairing the fired detectors.
"""
from collections import deque
from typing import NamedTuple

import numpy as np
from numba import njit


def _hadamard(values):
    """Unnormalized Walsh transform of a parity distribution."""
    values = values.copy()
    width = 1
    while width < len(values):
        pairs = values.reshape(-1, 2 * width)
        left = pairs[:, :width].copy()
        pairs[:, :width] += pairs[:, width:]
        pairs[:, width:] = left - pairs[:, width:]
        width *= 2
    return values


def _active_detectors(edges, num_detectors):
    adjacent = [[] for _ in range(num_detectors)]
    for edge in edges:
        if len(edge.detectors) == 2:
            u, v = edge.detectors
            adjacent[u].append(v)
            adjacent[v].append(u)
    active = {d for edge in edges if edge.logical for d in edge.detectors}
    stack = list(active)
    while stack:
        for neighbor in adjacent[stack.pop()]:
            if neighbor not in active:
                active.add(neighbor)
                stack.append(neighbor)
    return active


class _Paths(NamedTuple):
    keep: np.ndarray
    phase: np.ndarray
    order: np.ndarray
    starts: np.ndarray
    predecessor: np.ndarray
    edge: np.ndarray
    boundaries: np.ndarray


def _paths(primary, keep):
    n = len(keep)
    lookup = {detector: i for i, detector in enumerate(keep)}
    adjacent = [[] for _ in keep]
    boundaries = []
    for index, edge in enumerate(primary):
        u = lookup[edge.detectors[0]]
        if len(edge.detectors) == 1:
            boundaries.append((u, index, edge.logical))
        else:
            v = lookup[edge.detectors[1]]
            adjacent[u].append((v, index, edge.logical))
            adjacent[v].append((u, index, edge.logical))

    # Move logical labels to the boundary. A nontrivial internal logical
    # cycle needs a two-sheet graph and goes to the general decoder.
    phase = np.full(n, -1, np.int8)
    for root in range(n):
        if phase[root] >= 0:
            continue
        phase[root] = 0
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v, _, label in adjacent[u]:
                wanted = phase[u] ^ label
                if phase[v] < 0:
                    phase[v] = wanted
                    queue.append(v)
                elif phase[v] != wanted:
                    return None

    # Each source has its own shortest-path DAG. Store its predecessor lists
    # in breadth-first order so all path sums take a single forward scan.
    order = np.empty((n, n), np.int32)
    starts = np.empty((n, n + 1), np.int32)
    previous, arcs = [], []
    for source in range(n):
        distance = np.full(n, -1, dtype=np.int32)
        distance[source] = 0
        queue = deque([source])
        sequence = []
        while queue:
            u = queue.popleft()
            sequence.append(u)
            for v, _, _ in adjacent[u]:
                if distance[v] < 0:
                    distance[v] = distance[u] + 1
                    queue.append(v)
        sequence.extend(np.flatnonzero(distance < 0))
        order[source] = sequence
        for rank, u in enumerate(sequence):
            starts[source, rank] = len(arcs)
            if distance[u] >= 0:
                for v, index, _ in adjacent[u]:
                    if distance[v] == distance[u] - 1:
                        previous.append(v)
                        arcs.append(index)
        starts[source, n] = len(arcs)

    boundary = np.asarray([(u, index, phase[u] ^ label) for u, index, label in boundaries],
                          dtype=np.int32).reshape(-1, 3)
    return _Paths(keep, phase, order, starts, np.asarray(previous, np.int32),
                  np.asarray(arcs, np.int32), boundary)


class _Evidence(NamedTuple):
    odds: np.ndarray
    starts: np.ndarray
    updates: np.ndarray


def _evidence(model, primary, active, width):
    rates = np.asarray(model.probabilities)
    event_bias = np.log1p(-2 * rates)
    auxiliary = [[] for _ in rates]
    updates = [[] for _ in model.detector_events]
    odds = np.empty((len(primary), 1 << width), np.float64)
    for detector, events in enumerate(model.detector_events):
        if detector not in active:
            for event in events:
                auxiliary[event].append(detector)
    for edge_id, edge in enumerate(primary):
        events = model.edge_events[edge.detectors, edge.logical]
        scores = {}
        for event in events:
            for d in auxiliary[event]:
                scores[d] = scores.get(d, 0.) + rates[event]
        # Direct partners rank first. Nearby correlated checks help identify
        # other faults that could explain the same partner syndrome.
        for d, score in list(scores.items()):
            for event in model.detector_events[d]:
                for neighbor in auxiliary[event]:
                    scores[neighbor] = scores.get(neighbor, 0.) + score * rates[event]
        region = sorted(scores, key=scores.get, reverse=True)[:width]
        # Bit 0 is the edge parity. Each subsequent bit is an observed
        # opposite-sector check. A fault contributes its full projected mask.
        masks = {event: 1 for event in events}
        for bit, detector in enumerate(region):
            updates[detector].append((edge_id, 1 << bit))
            for event in model.detector_events[detector]:
                masks[event] = masks.get(event, 0) | (2 << bit)
        log_bias = np.bincount(list(masks.values()), weights=event_bias[list(masks)],
                               minlength=2 << len(region))
        # E[(-1)^(k.s)] = product(1-2p) over faults with odd k.mask.
        moments = np.exp(np.minimum(0., 0.5 * (log_bias.sum() - _hadamard(log_bias))))
        joint = np.maximum(_hadamard(moments), 1e-16)
        local_odds = np.clip(joint[1::2] / joint[::2], 1e-8, 2.)
        odds[edge_id] = np.resize(local_odds, 1 << width)
    starts = np.r_[0, np.cumsum([len(items) for items in updates])].astype(np.int32)
    flat = np.asarray([item for items in updates for item in items], dtype=np.int32).reshape(-1, 2)
    return _Evidence(odds, starts, flat)


def _pairing_schedule(capacity):
    """Keep only subsets reachable by pairing the first remaining detector."""
    reach = np.zeros(1 << capacity, dtype=bool)
    bit_index = np.zeros(len(reach), dtype=np.int32)
    for i in range(capacity):
        reach[(2 << i) - 1] = True
        bit_index[1 << i] = i
    for mask in range(len(reach) - 1, 0, -1):
        if reach[mask]:
            rest = mask & (mask - 1)
            reach[rest] = True
            bits = rest
            while bits:
                bit = bits & -bits
                reach[rest ^ bit] = True
                bits ^= bit
    return np.flatnonzero(reach)[1:], bit_index


@njit(cache=True)
def _predict(rows, graph, evidence, capacity, masks, bit_index, min_margin):
    n = len(graph.keep)
    hits = np.empty(capacity, np.int32)
    paths = np.empty(n)
    weights = np.empty(len(evidence.odds))
    keys = np.zeros(len(weights), np.int32)
    boundary_mass = np.empty(capacity)
    boundary_diff = np.empty(capacity)
    pairs = np.empty((capacity, capacity))
    difference = np.empty(1 << capacity)
    mass = np.empty_like(difference)
    mass[0] = difference[0] = 1.
    answers = np.zeros(len(rows), np.bool_)
    pending = np.ones(len(rows), np.bool_)
    for shot, row in enumerate(rows):
        count = 0
        offset = 0
        for node in range(n):
            if row[graph.keep[node]]:
                if count == capacity:
                    count += 1
                    break
                hits[count] = node
                count += 1
                offset ^= graph.phase[node]
        if count > capacity:
            continue
        if count == 0:
            pending[shot] = False
            continue
        keys[:] = 0
        for detector in range(len(row)):
            if row[detector]:
                for pos in range(evidence.starts[detector], evidence.starts[detector + 1]):
                    edge, bit = evidence.updates[pos]
                    keys[edge] |= bit
        for edge in range(len(weights)):
            weights[edge] = evidence.odds[edge, keys[edge]]
        for i in range(count):
            source = hits[i]
            paths[source] = 1.
            for rank in range(1, n):
                total = 0.
                for pos in range(graph.starts[source, rank], graph.starts[source, rank + 1]):
                    total += paths[graph.predecessor[pos]] * weights[graph.edge[pos]]
                paths[graph.order[source, rank]] = total
            even = odd = 0.
            for node, edge, label in graph.boundaries:
                term = paths[node] * weights[edge]
                if label:
                    odd += term
                else:
                    even += term
            boundary_mass[i] = even + odd
            boundary_diff[i] = even - odd
            for j in range(i):
                pairs[j, i] = paths[hits[j]]

        # Difference is the even-class mass minus the odd-class mass. After
        # fixing the first defect's partner, the remaining partition is reused.
        full = (1 << count) - 1
        for mask in masks:
            if mask > full:
                break
            i = bit_index[mask & -mask]
            rest = mask & (mask - 1)
            total = boundary_mass[i] * mass[rest]
            signed = boundary_diff[i] * difference[rest]
            bits = rest
            while bits:
                bit = bits & -bits
                weight = pairs[i, bit_index[bit]]
                total += weight * mass[rest ^ bit]
                signed += weight * difference[rest ^ bit]
                bits ^= bit
            mass[mask] = total
            difference[mask] = signed
        if abs(difference[full]) > min_margin * mass[full]:
            answers[shot] = (difference[full] < 0) != bool(offset)
            pending[shot] = False
    return answers, pending


class SoftPathModel:
    """Predict one logical bit or defer the shot. Input uses the DEM reference offset."""
    def __init__(self, model, num_detectors, width=7, capacity=13):
        self.graph = None
        active = _active_detectors(model.edges, num_detectors)
        primary = [edge for edge in model.edges if edge.detectors[0] in active]
        # Bound work by graph size. Exact half-probability faults have zero
        # Fourier bias; the generic matcher can decode those models directly.
        if (not active or any(edge.logical > 1 for edge in primary)
                or len(active) * (len(primary) + len(active)) > 3_000_000
                or any(p >= 0.5 for p in model.probabilities)):
            return
        keep = np.asarray(sorted(active), dtype=np.int32)
        self.graph = _paths(primary, keep)
        if self.graph is not None:
            self.evidence = _evidence(model, primary, active, width)
            self.capacity = min(len(active), capacity)
            self.schedule = _pairing_schedule(self.capacity)

    def predict(self, centered_bits, min_margin=.65):
        if self.graph is None:
            return (np.zeros(len(centered_bits), dtype=bool),
                    np.ones(len(centered_bits), dtype=bool))
        return _predict(centered_bits, self.graph, self.evidence, self.capacity,
                        *self.schedule, min_margin)
