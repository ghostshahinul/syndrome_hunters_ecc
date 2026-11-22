"""MWPM3D: our own minimum-weight perfect matching on the shared 3D detector graph.

Per shot with lit detectors L (k of them):
  1. Dijkstra (scipy.sparse.csgraph) from every lit detector over the detector graph plus the
     boundary node; edge length w_e = log((1-p)/p), using the most likely edge per pair.
  2. Matching graph: one node per lit detector and one boundary copy b_i per lit detector.
     Edges d_i-d_j with length dist(i, j), d_i-b_i with length dist(i, boundary), and b_i-b_j
     with length 0 (unused boundary copies pair up with each other for free).
  3. networkx.max_weight_matching(maxcardinality=True) on the negated lengths gives a
     minimum-weight perfect matching (exact blossom algorithm).
  4. Prediction = XOR of the observable bits along each matched shortest path.
This is the textbook MWPM decoder, written by us. PyMatching is only used to check it.
It is exact but slow (pure-Python blossom, O(k^3)), so its intended use is as a third tier
for the few shots that UnionFind3D is likely to get wrong (see Hierarchical3).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp
import stim
from scipy.sparse.csgraph import dijkstra

from src.decoders.base import Decoder
from src.decoders.cube3d.graph import DetectorGraph3D, build_detector_graph
from src.decoders.cube3d.union_find import bits_to_bool


class MWPM3D:
    """Exact MWPM engine on a DetectorGraph3D (any detection-event set)."""

    def __init__(self, graph: DetectorGraph3D):
        self.graph = g = graph
        n, B = g.n_det, g.boundary
        # most likely edge per unordered node pair (boundary = node n)
        best: dict[tuple[int, int], int] = {}
        for e in range(g.n_edges):
            u, v = int(g.edge_u[e]), int(g.edge_v[e])
            key = (min(u, v), max(u, v))
            if key not in best or g.edge_weight[e] < g.edge_weight[best[key]]:
                best[key] = e
        rows, cols, w = [], [], []
        self.pair_edge = np.full((n + 1, n + 1), -1, dtype=np.int32)
        for (u, v), e in best.items():
            rows.append(u)
            cols.append(v)
            w.append(max(float(g.edge_weight[e]), 1e-9))  # csgraph treats 0 as "no edge"
            self.pair_edge[u, v] = self.pair_edge[v, u] = e
        self.csr = sp.csr_matrix((w, (rows, cols)), shape=(n + 1, n + 1))
        self.B = B

    def _path_bits(self, pred_row: np.ndarray, src: int, dst: int) -> int:
        bits = 0
        cur = dst
        while cur != src:
            prev = int(pred_row[cur])
            if prev < 0:
                raise RuntimeError(f"no path {src} -> {dst}")
            bits ^= int(self.graph.edge_obs_bits[self.pair_edge[prev, cur]])
            cur = prev
        return bits

    def decode_shot(self, row: np.ndarray) -> int:
        lit = np.flatnonzero(row)
        k = len(lit)
        if k == 0:
            return 0
        dist, pred = dijkstra(self.csr, directed=False, indices=lit, return_predecessors=True)
        db = dist[:, self.B]
        dd = dist[:, lit]
        G = nx.Graph()
        for a in range(k):
            G.add_edge(a, k + a, weight=-float(db[a]))
        # Sparse but exact: a pair edge is only useful when it is shorter than sending both
        # ends to the boundary; boundary copies only need to pair along the kept pair edges
        # (in a perfect matching, unused copies belong to detector pairs matched together).
        ia, ib = np.nonzero(np.triu(dd < db[:, None] + db[None, :], k=1))
        for a, b in zip(ia.tolist(), ib.tolist()):
            G.add_edge(a, b, weight=-float(dd[a, b]))
            G.add_edge(k + a, k + b, weight=0.0)
        bits = 0
        for x, y in nx.max_weight_matching(G, maxcardinality=True):
            a, b = min(x, y), max(x, y)
            if a >= k:            # two boundary copies
                continue
            if b == k + a:        # lit detector to the boundary
                bits ^= self._path_bits(pred[a], int(lit[a]), self.B)
            else:                 # two lit detectors
                bits ^= self._path_bits(pred[a], int(lit[a]), int(lit[b]))
        return bits

    def run(self, detection_events: np.ndarray) -> np.ndarray:
        dets = np.asarray(detection_events)
        if dets.ndim != 2 or dets.shape[1] != self.graph.n_det:
            raise ValueError(f"expected (shots, {self.graph.n_det}), got {dets.shape}")
        bits = np.array([self.decode_shot(r) for r in dets], dtype=np.uint64)
        return bits_to_bool(bits, self.graph.n_obs)


class MWPM3DDecoder(Decoder):
    """Registry entry for the standalone MWPM (slow; mainly a correctness reference tier)."""

    name = "mwpm_3d"
    slow = True  # pure-Python blossom (networkx)

    def __init__(self, dem: stim.DetectorErrorModel):
        super().__init__(dem)
        self.engine = MWPM3D(build_detector_graph(dem))

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        return self.engine.run(self._check_input(detection_events))
