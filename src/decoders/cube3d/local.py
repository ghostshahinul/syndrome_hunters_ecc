"""Cube3D-Local: the fast path. Mutual-isolation rules on 3D detection events.

Rules (any edge type, so measurement errors and hook errors use the same rule):
  R1  lit u whose ONLY lit neighbor is v, and v's ONLY lit neighbor is u
      -> flip the highest-probability (u, v) edge, clear u and v.
  R2  unlit c with exactly 2 lit neighbors a, b, where the only lit node within graph
      distance 2 of a is b and vice versa -> flip (a, c) and (c, b), clear a and b.
  R3  lit u with a boundary edge and no other lit node within graph distance 2
      -> flip u's boundary edge, clear u.
Graph distance counts detector-detector edges only; the boundary node is not a shortcut
(through it every boundary-adjacent node would be within distance 2 of every other).

Schedule: rule applications are centered on a node (R1: u, R2: c, R3: u) and touch only
its closed neighborhood, so two applications conflict iff their closed neighborhoods
intersect (graph distance <= 2). The conflict graph is greedy-colored; R1, R2 and R3 run
in that order, each processing one color at a time, vectorized over shots and centers.

Gate: a shot is ACCEPTED iff 0 <= lit <= K before the rules and the rules clear every
lit node. Rejected shots are routed whole to the deep tier; their partial work here is
discarded. Prediction = XOR of the per-edge observable bits of flipped edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.decoders.cube3d.graph import DetectorGraph3D

RULES = ("r1", "r2", "r3")


@dataclass(frozen=True)
class Cube3DLocalConfig:
    K: int = 6                 # accept only shots with at most K detection events
    backend: str = "reference"

    def __post_init__(self) -> None:
        if self.K < 0:
            raise ValueError(f"K must be >= 0, got {self.K}")
        if self.backend not in ("reference",):
            raise ValueError(f"unknown backend {self.backend!r}")


@dataclass(frozen=True, eq=False)
class LocalTables:
    ball2: np.ndarray          # (n, B) int32 nodes at graph distance 1..2, -1 padded
    color_groups: list[np.ndarray]

    @property
    def n_colors(self) -> int:
        return len(self.color_groups)


def build_tables(g: DetectorGraph3D) -> LocalTables:
    n = g.n_det
    nbrs = [set(g.nbr[u][g.nbr[u] >= 0].tolist()) for u in range(n)]
    balls = []
    for u in range(n):
        b = set(nbrs[u])
        for v in nbrs[u]:
            b |= nbrs[v]
        b.discard(u)
        balls.append(sorted(b))
    B = max(max((len(b) for b in balls), default=1), 1)
    ball2 = np.full((n, B), -1, dtype=np.int32)
    for u, b in enumerate(balls):
        ball2[u, : len(b)] = b

    # Greedy coloring in node order; conflict = graph distance <= 2 (closed neighborhoods meet).
    colors = np.full(n, -1, dtype=np.int32)
    for u in range(n):
        used = {int(colors[v]) for v in balls[u] if colors[v] >= 0}
        c = 0
        while c in used:
            c += 1
        colors[u] = c
    groups = [np.flatnonzero(colors == c).astype(np.int32) for c in range(int(colors.max(initial=-1)) + 1)]
    ball2.setflags(write=False)
    return LocalTables(ball2, groups)


@dataclass(eq=False)
class LocalResult:
    predictions: np.ndarray      # (shots, n_obs) bool; meaningful only where accepted
    accepted: np.ndarray         # (shots,) bool
    lit_count: np.ndarray        # (shots,) int32 detection events per shot
    cleared: dict[str, np.ndarray]  # rule -> (shots,) lit nodes cleared
    n_colors: int
    flip_shot: np.ndarray | None = field(default=None, repr=False)  # flipped edges, if recorded
    flip_edge: np.ndarray | None = field(default=None, repr=False)

    @property
    def accept_fraction(self) -> float:
        return float(self.accepted.mean()) if self.accepted.size else 0.0


def _gather(lit: np.ndarray, rows: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """lit[rows[i], idx[i, j]] for -1 padded idx (m, D) -> (m, D) bool, False on padding."""
    valid = idx >= 0
    return lit[rows[:, None], np.where(valid, idx, 0)] & valid


def run_local(g: DetectorGraph3D, tables: LocalTables, detection_events: np.ndarray,
              K: int = 6, record_flips: bool = False) -> LocalResult:
    dets = np.asarray(detection_events, dtype=bool)
    S = dets.shape[0]
    lit_count = dets.sum(axis=1, dtype=np.int32)
    obs_bits = np.zeros(S, dtype=np.uint64)
    cleared = {r: np.zeros(S, dtype=np.int32) for r in RULES}
    fs: list[np.ndarray] = []
    fe: list[np.ndarray] = []

    active = np.flatnonzero((lit_count > 0) & (lit_count <= K))
    lit = dets[active].copy()  # work only on shots the gate could accept

    def flip(rows: np.ndarray, edges: np.ndarray) -> None:
        np.bitwise_xor.at(obs_bits, active[rows], g.edge_obs_bits[edges])
        if record_flips:
            fs.append(active[rows])
            fe.append(edges.astype(np.int32))

    for rule in RULES:
        for centers in tables.color_groups:
            on = lit[:, centers]                                    # (SA, k)
            if rule == "r2":
                # candidate: unlit center with >= 1 lit neighbor
                any_nb = np.zeros_like(on)
                for j in range(g.nbr.shape[1]):
                    col = g.nbr[centers, j]
                    ok = col >= 0
                    any_nb[:, ok] |= lit[:, col[ok]]
                s, k = np.nonzero(~on & any_nb)
            else:
                s, k = np.nonzero(on)
            if s.size == 0:
                continue
            c = centers[k]
            L = _gather(lit, s, g.nbr[c])                            # (m, D) lit neighbors
            cnt = L.sum(axis=1)

            if rule == "r1":
                keep = cnt == 1
                s, c, L = s[keep], c[keep], L[keep]
                j = np.argmax(L, axis=1)
                v = g.nbr[c, j]
                keep = _gather(lit, s, g.nbr[v]).sum(axis=1) == 1    # v's only lit neighbor is u
                s, c, j, v = s[keep], c[keep], j[keep], v[keep]
                if s.size == 0:
                    continue
                lit[s, c] = False
                lit[s, v] = False
                flip(s, g.nbr_edge[c, j])
                np.add.at(cleared["r1"], active[s], 2)

            elif rule == "r2":
                keep = cnt == 2
                s, c, L = s[keep], c[keep], L[keep]
                if s.size == 0:
                    continue
                order = np.argsort(~L, axis=1, kind="stable")[:, :2]  # slots of the 2 lit neighbors
                ja, jb = order[:, 0], order[:, 1]
                a, b = g.nbr[c, ja], g.nbr[c, jb]
                iso_a = _gather(lit, s, tables.ball2[a]).sum(axis=1) == 1  # only b near a
                iso_b = _gather(lit, s, tables.ball2[b]).sum(axis=1) == 1  # only a near b
                keep = iso_a & iso_b
                s, c, ja, jb, a, b = s[keep], c[keep], ja[keep], jb[keep], a[keep], b[keep]
                if s.size == 0:
                    continue
                lit[s, a] = False
                lit[s, b] = False
                flip(s, g.nbr_edge[c, ja])
                flip(s, g.nbr_edge[c, jb])
                np.add.at(cleared["r2"], active[s], 2)

            else:  # r3
                keep = (g.boundary_edge[c] >= 0) & (_gather(lit, s, tables.ball2[c]).sum(axis=1) == 0)
                s, c = s[keep], c[keep]
                if s.size == 0:
                    continue
                lit[s, c] = False
                flip(s, g.boundary_edge[c])
                np.add.at(cleared["r3"], active[s], 1)

    accepted = lit_count == 0
    accepted[active] = ~lit.any(axis=1)
    shifts = np.arange(g.n_obs, dtype=np.uint64)
    predictions = ((obs_bits[:, None] >> shifts) & np.uint64(1)).astype(bool)
    predictions[~accepted] = False
    flip_shot = np.concatenate(fs) if record_flips and fs else (np.zeros(0, np.int64) if record_flips else None)
    flip_edge = np.concatenate(fe) if record_flips and fe else (np.zeros(0, np.int32) if record_flips else None)
    return LocalResult(predictions, accepted, lit_count, cleared, tables.n_colors, flip_shot, flip_edge)


class Cube3DLocal:
    """Fast path. Not a complete decoder: rejected shots must go to a deep tier."""

    name = "cube3d_local"

    def __init__(self, graph: DetectorGraph3D, config: Cube3DLocalConfig | None = None):
        self.graph = graph
        self.config = config or Cube3DLocalConfig()
        self.tables = build_tables(graph)

    @property
    def n_colors(self) -> int:
        return self.tables.n_colors

    def run(self, detection_events: np.ndarray, K: int | None = None,
            record_flips: bool = False) -> LocalResult:
        dets = np.asarray(detection_events, dtype=bool)
        if dets.ndim != 2 or dets.shape[1] != self.graph.n_det:
            raise ValueError(f"expected (shots, {self.graph.n_det}), got {dets.shape}")
        return run_local(self.graph, self.tables, dets,
                         self.config.K if K is None else K, record_flips)
