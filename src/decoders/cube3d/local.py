"""Cube3D-Local: the fast path. Mutual-isolation rules on 3D detection events.

Rules (any edge type, so measurement errors and hook errors use the same rule):
  R1  lit u whose ONLY lit neighbor is v, and v's ONLY lit neighbor is u
      -> flip the highest-probability (u, v) edge, clear u and v.
      With strict_r1=True the pair must also be isolated within graph distance 2
      (the only lit node within distance 2 of u is v and vice versa), as in R2.
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
With record_residual=True the uncleared detection events and the raw (not zeroed)
observable bits are kept, so a "partial" gate can hand the residual to the deep tier.

Backends: "reference" (NumPy, vectorized over shots and the centers of one color) and
"numba" (per shot, parallel over shots). Within a color both evaluate every center
against the state before that color and then apply all actions, so they are bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

from src.decoders.cube3d.graph import DetectorGraph3D
from src.decoders.cube3d.union_find import extract_lits

RULES = ("r1", "r2", "r3")
LOCAL_BACKENDS = ("reference", "numba")


@dataclass(frozen=True)
class Cube3DLocalConfig:
    K: int = 6                 # accept only shots with at most K detection events
    backend: str = "reference"
    strict_r1: bool = False    # R1 pairs must be isolated within distance 2

    def __post_init__(self) -> None:
        if self.K < 0:
            raise ValueError(f"K must be >= 0, got {self.K}")
        if self.backend not in LOCAL_BACKENDS:
            raise ValueError(f"unknown backend {self.backend!r}")


@dataclass(frozen=True, eq=False)
class LocalTables:
    ball2: np.ndarray          # (n, B) int32 nodes at graph distance 1..2, -1 padded
    color_groups: list[np.ndarray]
    color: np.ndarray          # (n,) int32 color of each node

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
    return LocalTables(ball2, groups, colors)


@dataclass(eq=False)
class LocalResult:
    predictions: np.ndarray      # (shots, n_obs) bool; meaningful only where accepted
    accepted: np.ndarray         # (shots,) bool
    lit_count: np.ndarray        # (shots,) int32 detection events per shot
    cleared: dict[str, np.ndarray]  # rule -> (shots,) lit nodes cleared
    n_colors: int
    flip_shot: np.ndarray | None = field(default=None, repr=False)  # flipped edges, if recorded
    flip_edge: np.ndarray | None = field(default=None, repr=False)
    obs_bits: np.ndarray | None = field(default=None, repr=False)   # (shots,) raw uint64 bits
    residual: np.ndarray | None = field(default=None, repr=False)   # (shots, n) uncleared events

    @property
    def accept_fraction(self) -> float:
        return float(self.accepted.mean()) if self.accepted.size else 0.0


def _gather(lit: np.ndarray, rows: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """lit[rows[i], idx[i, j]] for -1 padded idx (m, D) -> (m, D) bool, False on padding."""
    valid = idx >= 0
    return lit[rows[:, None], np.where(valid, idx, 0)] & valid


def run_local(g: DetectorGraph3D, tables: LocalTables, detection_events: np.ndarray,
              K: int = 6, record_flips: bool = False, record_residual: bool = False,
              strict_r1: bool = False) -> LocalResult:
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
                if strict_r1:
                    keep &= (_gather(lit, s, tables.ball2[c]).sum(axis=1) == 1) &                             (_gather(lit, s, tables.ball2[v]).sum(axis=1) == 1)
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
    residual = None
    if record_residual:
        residual = dets.copy()
        residual[active] = lit
    return LocalResult(predictions, accepted, lit_count, cleared, tables.n_colors, flip_shot, flip_edge,
                       obs_bits if record_residual else None, residual)


# ---- numba kernel ---------------------------------------------------------------------
# Same schedule as run_local: R1, R2, R3; colors in order; within a color every center is
# evaluated against the state before that color, then all actions are applied. Within a
# color the actions touch disjoint nodes (centers are >= 3 apart), so apply order is free.
# nlc[v] = number of lit neighbors of v, maintained incrementally; it prunes candidates
# before the distance-2 counts (a lit neighbor is always inside the distance-2 ball).
CHUNK = 64


@njit(cache=True)
def _count_lit(lit, idx):
    k = 0
    for j in range(idx.shape[0]):
        v = idx[j]
        if v >= 0 and lit[v]:
            k += 1
    return k


@njit(cache=True)
def local_work(n, n_colors):
    """Scratch workspace for local_shot (clean state; local_shot leaves it clean again)."""
    return (np.zeros(n, np.int32),             # 0 nlc: lit-neighbor count
            np.empty(n, np.int32),             # 1 lit list
            np.empty(n, np.int32),             # 2 candidates
            np.empty(n, np.int32),             # 3 candidates bucketed by color
            np.zeros(n, np.bool_),             # 4 seen (R2 candidates)
            np.zeros(n_colors + 1, np.int32),  # 5 bucket starts
            np.zeros(n_colors, np.int32),      # 6 bucket fill
            np.empty(n, np.int32),             # 7 action node a
            np.empty(n, np.int32),             # 8 action node b
            np.empty(n, np.int32),             # 9 action edge 1
            np.empty(n, np.int32),             # 10 action edge 2
            np.empty(n, np.int32))             # 11 initially lit nodes (for the reset)


@njit(cache=True)
def _clear(lit, nlc, nbr, x):
    lit[x] = 0
    for j in range(nbr.shape[1]):
        w = nbr[x, j]
        if w >= 0:
            nlc[w] -= 1


@njit(cache=True)
def local_shot(lit, lits0, nlit0, nbr, nbr_edge, ball2, boundary_edge, color, n_colors, edge_obs_bits,
               cleared, strict_r1, work):
    """Apply R1, R2, R3 in place on one shot. `lit` (uint8, n) is the dense shot, whose lit
    nodes are exactly lits0[:nlit0] (no O(n) scan here). Adds cleared counts per rule to
    cleared (3,). Returns (observable bits of the flipped edges, lit nodes remaining)."""
    (nlc, litlist, cand, by_color, seen, start, fill, act_a, act_b, act_e, act_f, init) = work
    n = lit.shape[0]
    D = nbr.shape[1]
    nlit = nlit0
    for i in range(nlit0):
        litlist[i] = lits0[i]
        init[i] = lits0[i]
    n_init = nlit
    m = nlit
    bits = np.uint64(0)
    if nlit == 0:
        return bits, 0
    for i in range(nlit):
        v = litlist[i]
        for j in range(D):
            w = nbr[v, j]
            if w >= 0:
                nlc[w] += 1

    for rule in range(3):
        if nlit == 0:
            break
        # compact the lit list
        k2 = 0
        for i in range(m):
            v = litlist[i]
            if lit[v]:
                litlist[k2] = v
                k2 += 1
        m = k2
        # Candidates at the start of the rule. Lit nodes and lit-neighbor counts only ever
        # decrease, so every center the reference could act on later in this rule is here.
        # (A node cleared by R2 has no lit neighbor left, so it never becomes an R2 center.)
        nc = 0
        if rule == 1:
            for i in range(m):
                v = litlist[i]
                for j in range(D):
                    c = nbr[v, j]
                    if c >= 0 and not lit[c] and not seen[c] and nlc[c] >= 2:
                        seen[c] = True
                        cand[nc] = c
                        nc += 1
            for i in range(nc):
                seen[cand[i]] = False
        elif rule == 0:
            for i in range(m):
                v = litlist[i]
                if nlc[v] >= 1:
                    cand[nc] = v
                    nc += 1
        else:
            for i in range(m):
                v = litlist[i]
                if boundary_edge[v] >= 0:
                    cand[nc] = v
                    nc += 1
        if nc == 0:
            continue
        for k in range(n_colors + 1):
            start[k] = 0
        for i in range(nc):
            start[color[cand[i]] + 1] += 1
        for k in range(n_colors):
            start[k + 1] += start[k]
            fill[k] = start[k]
        for i in range(nc):
            k = color[cand[i]]
            by_color[fill[k]] = cand[i]
            fill[k] += 1

        for col in range(n_colors):
            na = 0
            for i in range(start[col], start[col + 1]):
                c = by_color[i]
                if rule == 0:  # R1: mutually isolated adjacent pair
                    if not lit[c] or nlc[c] != 1:
                        continue
                    jj = -1
                    for j in range(D):
                        v = nbr[c, j]
                        if v >= 0 and lit[v]:
                            jj = j
                            break
                    v = nbr[c, jj]
                    if nlc[v] != 1:
                        continue
                    if strict_r1 and (_count_lit(lit, ball2[c]) != 1 or _count_lit(lit, ball2[v]) != 1):
                        continue
                    act_a[na] = c
                    act_b[na] = v
                    act_e[na] = nbr_edge[c, jj]
                    na += 1
                elif rule == 1:  # R2: isolated pair two steps apart through unlit c
                    if nlc[c] != 2:
                        continue
                    ja = -1
                    jb = -1
                    for j in range(D):
                        v = nbr[c, j]
                        if v >= 0 and lit[v]:
                            if ja < 0:
                                ja = j
                            else:
                                jb = j
                                break
                    a = nbr[c, ja]
                    b = nbr[c, jb]
                    if nlc[a] > 1 or nlc[b] > 1:
                        continue
                    if _count_lit(lit, ball2[a]) != 1 or _count_lit(lit, ball2[b]) != 1:
                        continue
                    act_a[na] = a
                    act_b[na] = b
                    act_e[na] = nbr_edge[c, ja]
                    act_f[na] = nbr_edge[c, jb]
                    na += 1
                else:  # R3: isolated node next to the boundary
                    if not lit[c] or nlc[c] != 0:
                        continue
                    if _count_lit(lit, ball2[c]) != 0:
                        continue
                    act_a[na] = c
                    act_e[na] = boundary_edge[c]
                    na += 1
            for i in range(na):  # apply after evaluating the whole color (snapshot semantics)
                _clear(lit, nlc, nbr, act_a[i])
                bits ^= edge_obs_bits[act_e[i]]
                if rule == 2:
                    cleared[2] += 1
                    nlit -= 1
                else:
                    _clear(lit, nlc, nbr, act_b[i])
                    if rule == 1:
                        bits ^= edge_obs_bits[act_f[i]]
                    cleared[rule] += 2
                    nlit -= 2

    # restore the workspace: nlc is nonzero only next to initially lit nodes
    for i in range(n_init):
        v = init[i]
        for j in range(D):
            w = nbr[v, j]
            if w >= 0:
                nlc[w] = 0
    return bits, nlit


@njit(parallel=True, cache=True)
def _local_batch(dets, K, strict_r1, nbr, nbr_edge, ball2, boundary_edge, color, n_colors, edge_obs_bits,
                 out_bits, out_rem, out_cleared, residual, out_lit):
    record = residual.shape[0] > 0
    S = dets.shape[0]
    n = dets.shape[1]
    nchunk = (S + CHUNK - 1) // CHUNK
    for ci in prange(nchunk):
        work = local_work(n, n_colors)
        lit = np.zeros(n, np.uint8)   # dense shot, kept all-zero between shots
        lits = np.empty(n, np.int32)
        for s in range(ci * CHUNK, min(S, (ci + 1) * CHUNK)):
            c = extract_lits(dets[s], lits)
            out_lit[s] = c
            for i in range(c):
                lit[lits[i]] = 1
            if c == 0 or c > K:
                out_rem[s] = c
            else:
                b, rem = local_shot(lit, lits, c, nbr, nbr_edge, ball2, boundary_edge, color, n_colors,
                                    edge_obs_bits, out_cleared[s], strict_r1, work)
                out_bits[s] = b
                out_rem[s] = rem
            for i in range(c):
                v = lits[i]
                if record:
                    residual[s, v] = lit[v]
                lit[v] = 0


def run_local_numba(g: DetectorGraph3D, tables: LocalTables, detection_events: np.ndarray,
                    K: int = 6, record_residual: bool = False, strict_r1: bool = False) -> LocalResult:
    dets = np.ascontiguousarray(detection_events, dtype=np.uint8)
    S = dets.shape[0]
    bits = np.zeros(S, np.uint64)
    rem = np.zeros(S, np.int32)
    cl = np.zeros((S, 3), np.int32)
    residual = np.zeros((S, g.n_det) if record_residual else (0, g.n_det), np.uint8)
    lit_count = np.zeros(S, np.int32)
    _local_batch(dets, min(int(K), g.n_det), bool(strict_r1), g.nbr, g.nbr_edge, tables.ball2, g.boundary_edge,
                 tables.color, tables.n_colors, g.edge_obs_bits, bits, rem, cl, residual, lit_count)
    accepted = rem == 0
    shifts = np.arange(g.n_obs, dtype=np.uint64)
    predictions = ((bits[:, None] >> shifts) & np.uint64(1)).astype(bool)
    predictions[~accepted] = False
    cleared = {r: cl[:, i].copy() for i, r in enumerate(RULES)}
    return LocalResult(predictions, accepted, lit_count, cleared,
                       tables.n_colors, None, None, bits if record_residual else None,
                       residual.astype(bool) if record_residual else None)


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
            record_flips: bool = False, record_residual: bool = False,
            backend: str | None = None) -> LocalResult:
        dets = np.asarray(detection_events, dtype=bool)
        if dets.ndim != 2 or dets.shape[1] != self.graph.n_det:
            raise ValueError(f"expected (shots, {self.graph.n_det}), got {dets.shape}")
        K = self.config.K if K is None else K
        backend = backend or self.config.backend
        if backend == "numba":
            if record_flips:
                raise ValueError("record_flips is only available on the reference backend")
            return run_local_numba(self.graph, self.tables, dets, K, record_residual,
                                   self.config.strict_r1)
        return run_local(self.graph, self.tables, dets, K, record_flips, record_residual,
                         self.config.strict_r1)
