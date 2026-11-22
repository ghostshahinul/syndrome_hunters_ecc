"""3D detector graph shared by Cube3D-Local, UnionFind3D and the Hierarchical decoder.

Nodes are ALL detectors (every time layer) plus one boundary node (id = n_det). Every
component of circuit.detector_error_model(decompose_errors=True) becomes an edge:
2 detectors -> edge, 1 detector -> boundary edge. Components with identical endpoints
AND identical observable mask are merged (independent-XOR of their probabilities);
parallel components with different observable masks stay separate edges.

Edge types come from detector coordinates (x, y, t):
  boundary   1 detector
  space      both detectors in the same time layer
  time       same (x, y), different t           (measurement errors)
  diagonal   different (x, y) and different t   (space-time diagonal, e.g. hook/CNOT faults)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import stim

from src.circuits import tier_circuit

EDGE_TYPES = ("space", "time", "diagonal", "boundary")
T_SPACE, T_TIME, T_DIAGONAL, T_BOUNDARY = range(4)
P_MIN, P_MAX = 1e-15, 0.5 - 1e-9  # keep weights finite and positive


@dataclass(frozen=True, eq=False)
class DetectorGraph3D:
    """Read-only lookup tables. Edge e connects edge_u[e] and edge_v[e]; edge_v == n_det
    (the boundary node) for boundary edges."""

    n_det: int
    n_obs: int
    coords: np.ndarray        # (n_det, 3) float64: x, y, t
    edge_u: np.ndarray        # (E,) int32
    edge_v: np.ndarray        # (E,) int32, n_det for boundary edges
    edge_obs: np.ndarray      # (E, n_obs) bool
    edge_obs_bits: np.ndarray  # (E,) uint64 packed observable mask (bit k = observable k)
    edge_prob: np.ndarray     # (E,) float64
    edge_weight: np.ndarray   # (E,) float64, log((1-p)/p)
    edge_type: np.ndarray     # (E,) uint8 index into EDGE_TYPES
    nbr: np.ndarray           # (n_det, max_deg) int32 distinct detector neighbors, -1 padded
    nbr_edge: np.ndarray      # (n_det, max_deg) int32 highest-probability edge to that neighbor
    degree: np.ndarray        # (n_det,) int32 number of distinct detector neighbors
    boundary_edge: np.ndarray  # (n_det,) int32 highest-probability boundary edge, -1 if none
    inc_edge: np.ndarray      # (n_det, max_inc) int32 ALL incident edges (parallels, boundary), -1 padded
    n_components: int         # DEM components read (before merging)
    n_undetectable: int       # components with no detector (cannot be decoded; reported only)

    @property
    def boundary(self) -> int:
        return self.n_det

    @property
    def n_edges(self) -> int:
        return len(self.edge_u)

    def type_counts(self) -> dict[str, int]:
        return {name: int((self.edge_type == i).sum()) for i, name in enumerate(EDGE_TYPES)}

    def syndrome(self, correction: np.ndarray) -> np.ndarray:
        """(shots, E) edge flips -> (shots, n_det) bool detection events they explain."""
        corr = np.asarray(correction, dtype=np.uint8)
        out = np.zeros((corr.shape[0], self.n_det + 1), dtype=np.uint8)
        for e in np.flatnonzero(corr.any(axis=0)):
            out[:, self.edge_u[e]] ^= corr[:, e]
            out[:, self.edge_v[e]] ^= corr[:, e]
        return out[:, : self.n_det].astype(bool)

    def observables(self, correction: np.ndarray) -> np.ndarray:
        """(shots, E) edge flips -> (shots, n_obs) bool observable flips."""
        corr = np.asarray(correction, dtype=np.int64)
        return ((corr @ self.edge_obs.astype(np.int64)) & 1).astype(bool)


def _components(dem: stim.DetectorErrorModel):
    """Yield (probability, detectors tuple, observables tuple) for every decomposed component."""
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets: list[int] = []
        obs: list[int] = []
        for target in inst.targets_copy() + [stim.target_separator()]:
            if target.is_separator():
                yield p, tuple(sorted(dets)), tuple(sorted(obs))
                dets, obs = [], []
            elif target.is_relative_detector_id():
                dets.append(target.val)
            elif target.is_logical_observable_id():
                obs.append(target.val)


def _edge_type(c: np.ndarray, u: int, v: int, boundary: int) -> int:
    if v == boundary:
        return T_BOUNDARY
    (xu, yu, tu), (xv, yv, tv) = c[u, :3], c[v, :3]
    if tu == tv:
        return T_SPACE
    if xu == xv and yu == yv:
        return T_TIME
    return T_DIAGONAL


def build_detector_graph(dem: stim.DetectorErrorModel) -> DetectorGraph3D:
    n, n_obs = dem.num_detectors, dem.num_observables
    if n_obs > 64:
        raise ValueError(f"at most 64 observables supported, got {n_obs}")
    raw_coords = dem.get_detector_coordinates()
    coords = np.full((n, 3), np.nan)
    for det, c in raw_coords.items():
        coords[det, : min(3, len(c))] = c[:3]
    if np.isnan(coords).any():
        missing = np.flatnonzero(np.isnan(coords).any(axis=1))
        raise ValueError(f"detectors {missing[:5].tolist()} lack (x, y, t) coordinates")

    merged: dict[tuple[int, int, tuple[int, ...]], float] = defaultdict(float)
    n_components = n_undetectable = 0
    for p, dets, obs in _components(dem):
        n_components += 1
        if len(dets) == 0:
            n_undetectable += 1
            continue
        if len(dets) > 2:
            raise ValueError(f"component with {len(dets)} detectors {dets}: DEM is not graphlike "
                             "(build it with decompose_errors=True)")
        key = (dets[0], dets[1] if len(dets) == 2 else n, obs)
        q = merged[key]
        merged[key] = q * (1 - p) + p * (1 - q)

    keys = sorted(merged)
    E = len(keys)
    edge_u = np.array([k[0] for k in keys], dtype=np.int32)
    edge_v = np.array([k[1] for k in keys], dtype=np.int32)
    edge_obs = np.zeros((E, n_obs), dtype=bool)
    for e, k in enumerate(keys):
        edge_obs[e, list(k[2])] = True
    obs_bits = (edge_obs.astype(np.uint64) << np.arange(n_obs, dtype=np.uint64)).sum(axis=1, dtype=np.uint64) \
        if n_obs else np.zeros(E, dtype=np.uint64)
    prob = np.clip(np.array([merged[k] for k in keys], dtype=np.float64), P_MIN, P_MAX)
    weight = np.log((1 - prob) / prob)
    etype = np.array([_edge_type(coords, k[0], k[1], n) for k in keys], dtype=np.uint8)

    # Best edge per (node, neighbor) and per node -> boundary: highest probability, then lowest id.
    best: dict[tuple[int, int], int] = {}
    incident: list[list[int]] = [[] for _ in range(n)]
    for e in range(E):
        u, v = int(edge_u[e]), int(edge_v[e])
        incident[u].append(e)
        if v != n:
            incident[v].append(e)
        for a, b in ((u, v), (v, u)) if v != n else ((u, v),):
            cur = best.get((a, b))
            if cur is None or prob[e] > prob[cur]:
                best[(a, b)] = e

    nbr_lists: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    boundary_edge = np.full(n, -1, dtype=np.int32)
    for (a, b), e in best.items():
        if b == n:
            boundary_edge[a] = e
        else:
            nbr_lists[a].append((b, e))
    degree = np.array([len(lst) for lst in nbr_lists], dtype=np.int32)
    max_deg = max(int(degree.max(initial=0)), 1)
    nbr = np.full((n, max_deg), -1, dtype=np.int32)
    nbr_edge = np.full((n, max_deg), -1, dtype=np.int32)
    for a, lst in enumerate(nbr_lists):
        lst.sort()
        for j, (b, e) in enumerate(lst):
            nbr[a, j], nbr_edge[a, j] = b, e
    max_inc = max(max((len(x) for x in incident), default=1), 1)
    inc_edge = np.full((n, max_inc), -1, dtype=np.int32)
    for a, lst in enumerate(incident):
        inc_edge[a, : len(lst)] = lst

    arrays = (coords, edge_u, edge_v, edge_obs, obs_bits, prob, weight, etype,
              nbr, nbr_edge, degree, boundary_edge, inc_edge)
    for arr in arrays:
        arr.setflags(write=False)
    return DetectorGraph3D(n, n_obs, *arrays, n_components=n_components, n_undetectable=n_undetectable)


@lru_cache(maxsize=64)
def detector_graph(tier: str, distance: int, p: float, basis: str = "z",
                   rounds: int | None = None) -> DetectorGraph3D:
    """Cached graph per (noise tier, d, p, basis, rounds)."""
    circuit = tier_circuit(tier, distance, p, rounds, basis)
    return build_detector_graph(circuit.detector_error_model(decompose_errors=True))
