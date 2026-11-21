"""Reference baselines for validation and benchmarks. Never used inside the pipeline."""

from __future__ import annotations

import numpy as np
import pymatching

from src.decoders.cube_collapse.graph import BOUNDARY, CheckGraph


def pymatching_2d(graph: CheckGraph) -> pymatching.Matching:
    """MWPM on exactly the Cube-Collapse 2D check graph (same edges, weights, masks)."""
    m = pymatching.Matching()
    for e in range(graph.n_edges):
        u, v = int(graph.edge_u[e]), int(graph.edge_v[e])
        kw = dict(
            fault_ids=set(np.flatnonzero(graph.edge_obs[e]).tolist()),
            weight=float(graph.edge_weight[e]),
            error_probability=float(graph.edge_prob[e]),
        )
        if v == BOUNDARY:
            m.add_boundary_edge(u, **kw)
        else:
            m.add_edge(u, v, **kw)
    return m


def decode_2d(matching: pymatching.Matching, flat: np.ndarray, n_obs: int) -> np.ndarray:
    """(shots, n) syndromes -> (shots, n_obs) bool predictions."""
    pred = matching.decode_batch(np.asarray(flat, dtype=np.uint8))
    if pred.shape[1] < n_obs:  # pymatching sizes output by the largest fault id used
        pred = np.pad(pred, ((0, 0), (0, n_obs - pred.shape[1])))
    return pred[:, :n_obs].astype(bool)
