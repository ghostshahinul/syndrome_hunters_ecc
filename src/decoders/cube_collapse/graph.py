"""Stage 2: the 2D check graph and all precomputed tables.

Nodes are the selected check sites (keyed by (x, y)); edges are the space-like DEM
components (both detectors at selected sites in the same time layer) and boundary
components (one detector). Parallel components with the same endpoints and observable
mask are merged (independent-XOR of probabilities per layer, then averaged over layers).
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import stim

from src.decoders.cube_collapse.layout import CubeLayout, LayoutError, Site

BOUNDARY = -1
P_MIN, P_MAX = 1e-15, 0.5 - 1e-9  # keep weights finite and positive


@dataclass(frozen=True, eq=False)
class CheckGraph:
    """All arrays are read-only lookup tables; node id = index into `sites`.

    Column `n` (== n_nodes) of dist / hops / path_* is the BOUNDARY target.
    """

    sites: tuple[Site, ...]
    edge_u: np.ndarray        # (E,) int32
    edge_v: np.ndarray        # (E,) int32, BOUNDARY (-1) for boundary edges
    edge_obs: np.ndarray      # (E, n_obs) bool
    edge_prob: np.ndarray     # (E,) float64
    edge_weight: np.ndarray   # (E,) float64, log((1-p)/p)
    nbr: np.ndarray           # (n, max_deg) int32, -1 padded (node neighbors only)
    nbr_edge: np.ndarray      # (n, max_deg) int32, -1 padded
    degree: np.ndarray        # (n,) int32
    boundary_edge: np.ndarray  # (n,) int32, -1 if none
    dist: np.ndarray          # (n, n+1) float64 shortest-path weight
    hops: np.ndarray          # (n, n+1) int32 number of edges on that path
    path_edges: np.ndarray    # (n, n+1, L) int32, -1 padded
    path_len: np.ndarray      # (n, n+1) int32
    incidence: np.ndarray     # (n, E) uint8

    @property
    def n_nodes(self) -> int:
        return len(self.sites)

    @property
    def n_edges(self) -> int:
        return len(self.edge_u)

    @property
    def n_obs(self) -> int:
        return self.edge_obs.shape[1]

    def syndrome(self, correction: np.ndarray) -> np.ndarray:
        """(shots, E) edge flips -> (shots, n) uint8 node parities."""
        return ((np.asarray(correction, dtype=np.int32) @ self.incidence.T.astype(np.int32)) & 1).astype(np.uint8)

    def observables(self, correction: np.ndarray) -> np.ndarray:
        """(shots, E) edge flips -> (shots, n_obs) bool observable flips."""
        return ((np.asarray(correction, dtype=np.int32) @ self.edge_obs.astype(np.int32)) & 1).astype(bool)

    def signature(self) -> set[tuple]:
        """Coordinate-keyed edge set {(endpoint coords, obs tuple)} for graph comparisons."""
        sig = set()
        for e in range(self.n_edges):
            u, v = int(self.edge_u[e]), int(self.edge_v[e])
            ends = (self.sites[u], "B") if v == BOUNDARY else tuple(sorted((self.sites[u], self.sites[v])))
            sig.add((ends, tuple(np.flatnonzero(self.edge_obs[e]).tolist())))
        return sig


def _xor_prob(a: float, b: float) -> float:
    return a * (1 - b) + b * (1 - a)


def _collect_edges(dem: stim.DetectorErrorModel, layout: CubeLayout):
    """{(u, v): {obs_tuple: {t: p}}} from the space-like / boundary DEM components."""
    det_site = np.full(dem.num_detectors, -1, dtype=np.int64)
    det_t = np.full(dem.num_detectors, -1, dtype=np.int64)
    for t in range(layout.n_layers):
        det_site[layout.det_index[t]] = np.arange(layout.n_sites)
        det_t[layout.det_index[t]] = t

    buckets: dict[tuple[int, int], dict[tuple, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        components: list[list[stim.DemTarget]] = [[]]
        for target in inst.targets_copy():
            if target.is_separator():
                components.append([])
            else:
                components[-1].append(target)
        for comp in components:
            dets = [t.val for t in comp if t.is_relative_detector_id()]
            obs = tuple(sorted(t.val for t in comp if t.is_logical_observable_id()))
            if len(dets) not in (1, 2) or (det_site[dets] < 0).any():
                continue
            if len(dets) == 2 and det_t[dets[0]] != det_t[dets[1]]:
                continue  # time-like or space-time diagonal: not part of the 2D graph
            s = sorted(int(x) for x in det_site[dets])
            if len(s) == 2 and s[0] == s[1]:
                continue
            key = (s[0], s[1]) if len(s) == 2 else (s[0], BOUNDARY)
            t = int(det_t[dets[0]])
            layer = buckets[key][obs]
            layer[t] = _xor_prob(layer.get(t, 0.0), p)
    return buckets


def _shortest_paths(n: int, adj: list[list[tuple[int, int, float]]]):
    """Dijkstra from every node; target n is the BOUNDARY sink (never relaxed through).
    Ties broken by (distance, hops, predecessor id) so results are deterministic."""
    INF = math.inf
    dist = np.full((n, n + 1), INF)
    hops = np.full((n, n + 1), -1, dtype=np.int32)
    paths: list[list[list[int]]] = []
    for s in range(n):
        d = [INF] * (n + 1)
        h = [1 << 30] * (n + 1)
        pred = [-1] * (n + 1)
        pred_e = [-1] * (n + 1)
        done = [False] * (n + 1)
        d[s], h[s] = 0.0, 0
        heap = [(0.0, 0, s)]
        while heap:
            du, hu, u = heapq.heappop(heap)
            if done[u]:
                continue
            done[u] = True
            if u == n:
                continue
            for v, e, w in adj[u]:
                if done[v]:
                    continue
                cand = (du + w, hu + 1, u)
                if cand < (d[v], h[v], pred[v] if pred[v] >= 0 else n + 1):
                    d[v], h[v], pred[v], pred_e[v] = cand[0], cand[1], u, e
                    heapq.heappush(heap, (d[v], h[v], v))
        row = []
        for t in range(n + 1):
            path = []
            if d[t] < INF:
                x = t
                while x != s:
                    path.append(pred_e[x])
                    x = pred[x]
                path.reverse()
                dist[s, t], hops[s, t] = d[t], h[t]
            row.append(path)
        paths.append(row)
    return dist, hops, paths


def build_check_graph(dem: stim.DetectorErrorModel, layout: CubeLayout | None = None) -> CheckGraph:
    layout = layout or CubeLayout.from_dem(dem)
    n = layout.n_sites
    buckets = _collect_edges(dem, layout)

    keys = sorted(buckets, key=lambda k: (k[0], n if k[1] == BOUNDARY else k[1]))
    edge_u, edge_v, edge_obs, edge_p = [], [], [], []
    for key in keys:
        variants = buckets[key]
        if len(variants) != 1:
            raise LayoutError(
                f"edge {layout.sites[key[0]]}-{'B' if key[1] < 0 else layout.sites[key[1]]} "
                f"has {len(variants)} different observable masks {sorted(variants)}"
            )
        obs, per_layer = next(iter(variants.items()))
        edge_u.append(key[0])
        edge_v.append(key[1])
        edge_obs.append(obs)
        edge_p.append(float(np.mean(list(per_layer.values()))))

    E = len(keys)
    obs_arr = np.zeros((E, dem.num_observables), dtype=bool)
    for e, obs in enumerate(edge_obs):
        obs_arr[e, list(obs)] = True
    prob = np.clip(np.array(edge_p, dtype=np.float64), P_MIN, P_MAX)
    weight = np.log((1 - prob) / prob)
    eu = np.array(edge_u, dtype=np.int32)
    ev = np.array(edge_v, dtype=np.int32)

    nbr_lists: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    boundary_edge = np.full(n, -1, dtype=np.int32)
    adj: list[list[tuple[int, int, float]]] = [[] for _ in range(n)]
    incidence = np.zeros((n, E), dtype=np.uint8)
    for e in range(E):
        u, v = int(eu[e]), int(ev[e])
        incidence[u, e] = 1
        if v == BOUNDARY:
            boundary_edge[u] = e
            adj[u].append((n, e, float(weight[e])))
        else:
            incidence[v, e] = 1
            nbr_lists[u].append((v, e))
            nbr_lists[v].append((u, e))
            adj[u].append((v, e, float(weight[e])))
            adj[v].append((u, e, float(weight[e])))
    for lst in nbr_lists:
        lst.sort()

    degree = np.array([len(lst) for lst in nbr_lists], dtype=np.int32)
    max_deg = max(int(degree.max(initial=0)), 1)
    nbr = np.full((n, max_deg), -1, dtype=np.int32)
    nbr_edge = np.full((n, max_deg), -1, dtype=np.int32)
    for u, lst in enumerate(nbr_lists):
        for j, (v, e) in enumerate(lst):
            nbr[u, j], nbr_edge[u, j] = v, e

    dist, hops, paths = _shortest_paths(n, adj)
    unreachable = np.flatnonzero(~np.isfinite(dist[:, n]))
    assert unreachable.size == 0, f"nodes {[layout.sites[i] for i in unreachable]} cannot reach the boundary"

    L = max((len(p) for row in paths for p in row), default=1) or 1
    path_edges = np.full((n, n + 1, L), -1, dtype=np.int32)
    path_len = np.zeros((n, n + 1), dtype=np.int32)
    for s in range(n):
        for t in range(n + 1):
            p = paths[s][t]
            path_edges[s, t, : len(p)] = p
            path_len[s, t] = len(p)

    for arr in (eu, ev, obs_arr, prob, weight, nbr, nbr_edge, degree, boundary_edge,
                dist, hops, path_edges, path_len, incidence):
        arr.setflags(write=False)
    return CheckGraph(layout.sites, eu, ev, obs_arr, prob, weight, nbr, nbr_edge, degree,
                      boundary_edge, dist, hops, path_edges, path_len, incidence)
