"""Stage 4: growing-radius greedy completion. Always resolves every lit node.

For r = 1, 2, ...: candidates are pairs of still-lit nodes whose shortest path has at
most r edges, plus (lit node, BOUNDARY) whose boundary path has at most r edges. They
are taken in order (path weight, lowest confidence of the endpoints, node ids), and a
candidate is accepted if both endpoints are still lit: its precomputed shortest-path
edges are flipped and both endpoints cleared.

The radius counts edges (hops) on the weight-shortest path; ordering uses the weight.
With uniform edge probabilities the two coincide. Termination: every node has a
finite boundary path, so at r = max boundary hops each remaining node has a candidate.
Shots are independent, so this stage parallelizes over shots.
"""

from __future__ import annotations

import numpy as np

from src.decoders.cube_collapse.graph import CheckGraph


def complete_shot(lit_row: np.ndarray, flip_row: np.ndarray, conf_row: np.ndarray,
                  graph: CheckGraph) -> tuple[int, int]:
    """Resolve one shot in place. Returns (lit nodes cleared, max radius used)."""
    n = graph.n_nodes
    nodes = np.flatnonzero(lit_row)
    if nodes.size == 0:
        return 0, 0

    # Candidate list: (a, b) with b == n meaning BOUNDARY. Sorted once; the radius
    # filter only selects a prefix-independent subset, so order is preserved.
    ia, ib = np.triu_indices(nodes.size, k=1)
    a = np.concatenate([nodes[ia], nodes])
    b = np.concatenate([nodes[ib], np.full(nodes.size, n)])
    dist = graph.dist[a, b]
    hops = graph.hops[a, b]
    conf_b = np.where(b < n, conf_row[np.minimum(b, n - 1)], 0)
    conf = np.maximum(conf_row[a], conf_b)
    order = np.lexsort((b, a, conf, dist))
    a, b, hops = a[order], b[order], hops[order]

    lit = lit_row.astype(bool)
    remaining = nodes.size
    max_r = 0
    r_limit = int(graph.hops[nodes, n].max())
    for r in range(1, r_limit + 1):
        for i in np.flatnonzero(hops <= r):
            u, v = a[i], b[i]
            if not lit[u] or (v < n and not lit[v]):
                continue
            path = graph.path_edges[u, v, : graph.path_len[u, v]]
            flip_row[path] ^= 1
            lit[u] = False
            remaining -= 1
            if v < n:
                lit[v] = False
                remaining -= 1
            max_r = r
        if remaining == 0:
            break
    assert remaining == 0, "Stage 4 failed to resolve all lit nodes"
    lit_row[:] = 0
    return int(nodes.size), max_r


def stage_s4(lit, flips, confidence, graph) -> tuple[np.ndarray, np.ndarray]:
    """Mutates lit/flips in place. Returns (cleared per shot, radius per shot)."""
    S = lit.shape[0]
    cleared = np.zeros(S, dtype=np.int32)
    radius = np.zeros(S, dtype=np.int32)
    for s in np.flatnonzero(lit.any(axis=1)):
        cleared[s], radius[s] = complete_shot(lit[s], flips[s], confidence[s], graph)
    return cleared, radius
