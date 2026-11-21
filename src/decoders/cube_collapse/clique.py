"""Stage 3: Clique local pass (graph version of Clique_L1 + L2, arXiv:2507.11481).

A clique is a center node plus its neighbors. Rules, applied in this order:
  S1 (L1)       lit center with exactly 1 lit neighbor -> flip that edge, clear both.
  S2 (L2)       unlit center with exactly 2 lit neighbors -> flip both center edges,
                clear both; unlit degree-4 center with all 4 neighbors lit -> flip all 4.
  S3 (boundary) lit center with 0 lit neighbors and a boundary edge -> flip it.
Deviation from the paper: Clique_L1 also decodes a center with 3 lit neighbors locally;
here that case is left lit and resolved by Stage 4.

Parallel schedule: two cliques conflict iff their closed neighborhoods intersect
(graph distance <= 2). The conflict graph is greedy-colored; each stage processes the
colors one after another and all cliques of a color at once, vectorized over shots and
cliques. Cliques of one color touch disjoint node and edge sets, so the fancy-indexed
updates below never collide.
"""

from __future__ import annotations

import numpy as np

from src.decoders.cube_collapse.graph import CheckGraph


def color_cliques(graph: CheckGraph) -> list[np.ndarray]:
    """Greedy coloring (node order) of the clique conflict graph -> centers per color."""
    n = graph.n_nodes
    closed = [{u, *graph.nbr[u][graph.nbr[u] >= 0].tolist()} for u in range(n)]
    colors = np.full(n, -1, dtype=np.int32)
    for u in range(n):
        used = {colors[v] for v in range(u) if closed[u] & closed[v]}
        c = 0
        while c in used:
            c += 1
        colors[u] = c
    return [np.flatnonzero(colors == c).astype(np.int32) for c in range(int(colors.max()) + 1)]


def _neighbor_state(lit: np.ndarray, graph: CheckGraph, centers: np.ndarray):
    nb = graph.nbr[centers]                       # (k, D)
    valid = nb >= 0
    L = lit[:, np.where(valid, nb, 0)] & valid    # (S, k, D) uint8
    return nb, L, L.sum(axis=2)


def stage_s1(lit, flips, graph, color_groups) -> np.ndarray:
    """Mutates lit/flips in place; returns lit nodes cleared per shot.

    Known limitation (measured): on a 3-node lit chain whose end has a boundary edge,
    S1 pairs the end greedily; 14 of 210 weight-2 errors at d=5 fail this way.
    Restricting S1 to isolated pairs was tried and is worse overall (more load on
    Stage 4), see the Phase 1 notes.
    """
    cleared = np.zeros(lit.shape[0], dtype=np.int32)
    for centers in color_groups:
        nb, L, cnt = _neighbor_state(lit, graph, centers)
        s, k = np.nonzero((lit[:, centers] == 1) & (cnt == 1))
        if s.size == 0:
            continue
        j = np.argmax(L[s, k], axis=1)
        lit[s, centers[k]] = 0
        lit[s, nb[k, j]] = 0
        flips[s, graph.nbr_edge[centers[k], j]] ^= 1
        np.add.at(cleared, s, 2)
    return cleared


def stage_s2(lit, flips, graph, color_groups) -> np.ndarray:
    cleared = np.zeros(lit.shape[0], dtype=np.int32)
    for centers in color_groups:
        nb, L, cnt = _neighbor_state(lit, graph, centers)
        unlit = lit[:, centers] == 0
        full4 = (graph.degree[centers] == 4)[None, :] & (cnt == 4)
        s, k = np.nonzero(unlit & ((cnt == 2) | full4))
        if s.size == 0:
            continue
        rows, j = np.nonzero(L[s, k])
        shot, kk = s[rows], k[rows]
        lit[shot, nb[kk, j]] = 0
        flips[shot, graph.nbr_edge[centers[kk], j]] ^= 1
        np.add.at(cleared, shot, 1)
    return cleared


def stage_s3(lit, flips, graph, color_groups) -> np.ndarray:
    cleared = np.zeros(lit.shape[0], dtype=np.int32)
    for centers in color_groups:
        _, _, cnt = _neighbor_state(lit, graph, centers)
        has_b = (graph.boundary_edge[centers] >= 0)[None, :]
        s, k = np.nonzero((lit[:, centers] == 1) & (cnt == 0) & has_b)
        if s.size == 0:
            continue
        lit[s, centers[k]] = 0
        flips[s, graph.boundary_edge[centers[k]]] ^= 1
        np.add.at(cleared, s, 1)
    return cleared
