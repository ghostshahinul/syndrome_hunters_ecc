"""Shared helpers for the Cube-Collapse tests."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.circuits import code_capacity_circuit
from src.decoders.cube_collapse import CubeCollapseDecoder
from src.decoders.cube_collapse.graph import BOUNDARY, CheckGraph


@lru_cache(maxsize=None)
def cc_decoder(d: int, basis: str = "z") -> CubeCollapseDecoder:
    dem = code_capacity_circuit(d, 0.05, basis).detector_error_model(decompose_errors=True)
    return CubeCollapseDecoder(dem)


def edge_between(g: CheckGraph, u: int, v: int) -> int:
    """Edge id between nodes u and v (v = BOUNDARY for u's boundary edge)."""
    if v == BOUNDARY:
        e = int(g.boundary_edge[u])
    else:
        hits = np.flatnonzero(((g.edge_u == u) & (g.edge_v == v)) | ((g.edge_u == v) & (g.edge_v == u)))
        e = int(hits[0]) if hits.size else -1
    assert e >= 0, f"no edge between {u} and {v}"
    return e


def errors_to_case(g: CheckGraph, edges) -> tuple[np.ndarray, np.ndarray]:
    """Edge-error set -> (flat syndrome (1, n), true observable flips (1, n_obs))."""
    corr = np.zeros((1, g.n_edges), dtype=np.uint8)
    for e in edges:
        corr[0, e] ^= 1
    return g.syndrome(corr), g.observables(corr)


def decode_errors(dec: CubeCollapseDecoder, edges):
    flat, truth = errors_to_case(dec.graph, edges)
    result = dec.decode_flat(flat)
    assert (dec.graph.syndrome(result.correction) == flat).all()
    return result, truth


def bulk_nodes(g: CheckGraph) -> np.ndarray:
    """Degree-4 nodes without a boundary edge."""
    return np.flatnonzero((g.degree == 4) & (g.boundary_edge < 0))


def far(g: CheckGraph, group_a, group_b, min_hops: int = 3) -> bool:
    return all(g.hops[a, b] >= min_hops for a in group_a for b in group_b)


def path_edges(g: CheckGraph, u: int, v: int) -> list[int]:
    t = g.n_nodes if v == BOUNDARY else v
    return g.path_edges[u, t, : g.path_len[u, t]].tolist()
