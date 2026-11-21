"""Condition graph-edge growth costs on detector clicks in a correlated partner."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from src.decoders.faults import edge_arrays
from src.decoders.union_find import growth_unit


def conditioned_graph(model, n):
    left, right, weights, labels = edge_arrays(model, n)
    unit = growth_unit(weights)
    # Only components carrying a logical mask can affect the answer. Do not
    # connect separate components through the virtual boundary during this pass.
    neighbours = [[] for _ in range(n)]
    for u, v in zip(left, right):
        if v < n:
            neighbours[u].append(v)
            neighbours[v].append(u)
    live = np.zeros(n, bool)
    stack = list(left[labels != 0])
    while stack:
        u = stack.pop()
        if not live[u]:
            live[u] = True
            stack.extend(neighbours[u])
    keep = np.flatnonzero(live)
    remap = np.empty(n + 1, np.int32)
    remap[keep], remap[n] = np.arange(len(keep)), len(keep)
    take = live[left]
    left, right, weights, labels = left[take], right[take], weights[take], labels[take]
    length = np.maximum(1, np.rint(weights / unit)).astype(np.int32)
    graph = remap[left], remap[right], length, labels
    edge_ids = {((int(u),) if v == n else (int(u), int(v)), int(label)): e
                for e, (u, v, label) in enumerate(zip(left, right, labels))}

    # For a parity of independent faults, E[(-1)^parity] = product(1-2p).
    # Each event set describes the Bernoulli variables whose XOR is measured.
    bias = [1 - 2 * p for p in model.probabilities]

    def moment(events):
        return math.prod(bias[i] for i in events)

    @lru_cache(maxsize=None)
    def evidence_moments(sites):
        a = model.detector_events[sites[0]]
        if len(sites) == 1:
            return (a,), (1 - moment(a))
        b = model.detector_events[sites[1]]
        ab = a ^ b
        return (a, b, ab), (1 - moment(a) - moment(b) + moment(ab))

    rules = {}
    for first, second in sorted(model.couplings):
        for key, partner in ((first, second[0]), (second, first[0])):
            e = edge_ids.get(key)
            if e is None or set(key[0]).intersection(partner):
                continue
            evidence, denominator = evidence_moments(partner)
            if denominator < 1e-14:
                continue
            events = model.edge_events[key]
            signed = moment(events) - moment(events ^ evidence[0])
            if len(partner) == 2:
                signed += moment(events ^ evidence[2]) - moment(events ^ evidence[1])
            # Exact single-edge posterior given these one or two partner clicks.
            # Combining multiple visible partners by taking the strongest
            # discount is a heuristic and avoids multiplying overlapping evidence.
            q = min(0.5, (1 - signed / denominator) / 2)
            if q > 0:
                ticks = max(1, round((math.log1p(-q) - math.log(q)) / unit))
                if ticks < length[e]:
                    rules[partner, e] = ticks
    rows = sorted((sites[0], sites[1] if len(sites) == 2 else -1, e, ticks)
                  for (sites, e), ticks in rules.items())
    a, other, edge, ticks = np.array(rows, dtype=np.int32).reshape(-1, 4).T
    starts = np.r_[0, np.cumsum(np.bincount(a, minlength=n))].astype(np.int32)
    return keep, graph, (starts, other.copy(), edge.copy(), ticks.copy())
