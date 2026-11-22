"""UnionFind3D: our own weighted Delfosse-Nickerson Union-Find on the 3D detector graph.

Algorithm, per shot:
  1. Every lit detector starts as an odd cluster. Edge lengths are small integers,
     L_e = max(1, round(S * w_e / w_min)), so the lightest edge takes S half-edge steps.
     Default S = 4 (Milestone 1: S=2 -> 4 cut LER at d=9 from 1.142% to 0.876% for a
     1.19x single-thread slowdown; S=8 was within the CI of S=4 at 1.55x).
     Optional growth="smallest" grows only the minimum-size active clusters per round
     (Delfosse-Nickerson); it was within the CI of growth="all" at every d, so "all"
     stays the default.
  2. Growth rounds: every ACTIVE cluster (odd parity and not touching the boundary)
     adds 1 to the support of each not-yet-grown edge incident to its nodes (an edge
     between two active clusters gains 2 per round). All increments of a round are
     applied before any merge.
  3. Edges whose support reaches L_e are grown: their endpoints' clusters merge
     (union by rank, path compression; parity XOR, boundary flag OR). A grown boundary
     edge marks the cluster as touching the boundary. Clusters freeze when even or
     touching the boundary; growth stops when no cluster is active.
  4. Peeling on a spanning forest of the grown edges: BFS from the boundary node first,
     then from remaining cluster nodes in ascending id; neighbors in ascending edge id.
     Only ODD clusters may use grown boundary edges. (With weighted lengths a boundary
     edge can finish growing in the same round as the internal edge that makes the
     cluster even; peeling through the boundary there flips the logical: this broke
     every single internal fault at d=3 before the rule was added.)
     Leaves are peeled in reverse BFS order; a defective leaf flips its tree edge.
  5. Prediction = XOR of the observable bits of the peeled edges.

The result does not depend on union-find root choices or iteration order inside a
round, which is what makes the numba kernel bit-identical to the Python reference.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace

import numba
import numpy as np
import stim
from numba import njit, prange

from src.decoders.base import Decoder
from src.decoders.cube3d.graph import DetectorGraph3D, build_detector_graph

UF_BACKENDS = ("reference", "numba")
UF_GROWTH = ("all", "smallest")


@dataclass(frozen=True)
class UnionFind3DConfig:
    """backend: reference | numba.  scale: S in L_e = max(1, round(S * w_e / w_min)).
    growth: "all" grows every active cluster each round; "smallest" grows only the active
    clusters of minimum node count (Delfosse-Nickerson rule; ties all grow)."""
    backend: str = "numba"
    scale: int = 4
    growth: str = "all"
    correlated: bool = False  # two-pass: reweight partners of pass-1 edges, decode again
    skip_ahead: bool = False  # exact skip-ahead growth (same results; pays off only at large scale)
    table: bool = False       # answer 1-4-event shots that match a single DEM fault from a table
    float_score: bool = False  # E1: pass-selection score on fine weights (fixed point w * 2^24)
    weighted_forest: bool = False  # E2: peel on a min-length (Kruskal) spanning forest
    passes: int = 2           # E4: 3 = reweight again from pass 2's correction (if it was kept)

    def __post_init__(self) -> None:
        if self.backend not in UF_BACKENDS:
            raise ValueError(f"backend must be one of {UF_BACKENDS}, got {self.backend!r}")
        if self.growth not in UF_GROWTH:
            raise ValueError(f"growth must be one of {UF_GROWTH}, got {self.growth!r}")
        if self.scale < 1:
            raise ValueError(f"scale must be >= 1, got {self.scale}")
        if self.passes not in (2, 3):
            raise ValueError(f"passes must be 2 or 3, got {self.passes}")


@dataclass(frozen=True, eq=False)
class UFTables:
    length: np.ndarray   # (E,) int32 integer edge lengths
    b_inc: np.ndarray    # (B,) int32 boundary edges, ascending id
    smallest: bool = False  # growth rule: only minimum-size active clusters grow
    corr_ptr: np.ndarray | None = None  # (E+1,) int32 CSR over partner edges (see graph.corr_*)
    corr_idx: np.ndarray | None = None  # (P,) int32 partner f
    corr_len: np.ndarray | None = None  # (P,) int32 length of f given e is in the correction
    corr_credit: np.ndarray | None = None  # (P,) int32 max(0, L_e + L_f - L_joint): pair bonus
    skip: bool = False   # skip-ahead growth path in the numba kernel
    wforest: bool = False  # peel on a Kruskal min-length spanning forest
    passes: int = 2        # correlated passes (2 or 3)
    score_len: np.ndarray | None = None     # (E,) int64 edge cost in the pass-selection score
    score_credit: np.ndarray | None = None  # (P,) int64 pair credit in the pass-selection score


def length_of(w: np.ndarray, w_min: float, scale: int) -> np.ndarray:
    return np.maximum(1, np.rint(float(scale) * w / w_min)).astype(np.int32)


def build_uf_tables(g: DetectorGraph3D, scale: int = 4, growth: str = "all") -> UFTables:
    w = g.edge_weight
    w_min = float(w.min())
    length = length_of(w, w_min, scale)
    b_inc = np.flatnonzero(g.edge_v == g.boundary).astype(np.int32)
    ptr = g.corr_ptr if g.corr_ptr is not None else np.zeros(g.n_edges + 1, np.int32)
    idx = g.corr_idx if g.corr_idx is not None else np.zeros(0, np.int32)
    joint = g.corr_joint if g.corr_joint is not None else np.zeros(0)
    # P(f | e) ~ P(e and f) / P(e), clipped to [p_f, 0.5]; length on the same integer scale
    src = np.repeat(np.arange(g.n_edges), np.diff(ptr))
    cond = np.clip(joint / g.edge_prob[src], g.edge_prob[idx], 0.5) if len(idx) else np.zeros(0)
    cond = np.minimum(cond, 0.5 - 1e-12)
    corr_len = np.minimum(length_of(np.log((1 - cond) / cond), w_min, scale), length[idx]) \
        if len(idx) else np.zeros(0, np.int32)
    # Score bonus when a correction contains both e and f: the pair is likelier as one fault
    # (probability P(e and f)) than as two independent ones.
    pj = np.clip(joint, 1e-15, 0.5 - 1e-12) if len(idx) else np.zeros(0)
    credit = np.maximum(0, length[src] + length[idx] - length_of(np.log((1 - pj) / pj), w_min, scale)) \
        if len(idx) else np.zeros(0, np.int32)
    return UFTables(length, b_inc, growth == "smallest", np.ascontiguousarray(ptr, np.int32),
                    np.ascontiguousarray(idx, np.int32), corr_len.astype(np.int32),
                    credit.astype(np.int32), score_len=length.astype(np.int64),
                    score_credit=np.asarray(credit, np.int64))


FIXED_POINT = float(1 << 24)


def float_score_tables(g: DetectorGraph3D, t: UFTables) -> tuple[np.ndarray, np.ndarray]:
    """E1: the pass-selection score on fine weights: w_e and the pair credit
    max(0, w_e + w_f - w_joint) as fixed-point int64 (w * 2^24). Integer sums do not depend on
    the summation order, so numba and the reference stay bit-identical."""
    w = g.edge_weight
    src = np.repeat(np.arange(g.n_edges), np.diff(t.corr_ptr))
    joint = g.corr_joint if g.corr_joint is not None else np.zeros(0)
    if len(t.corr_idx):
        pj = np.clip(joint, 1e-15, 0.5 - 1e-12)
        cred = np.maximum(0.0, w[src] + w[t.corr_idx] - np.log((1 - pj) / pj))
    else:
        cred = np.zeros(0)
    return (np.rint(w * FIXED_POINT).astype(np.int64), np.rint(cred * FIXED_POINT).astype(np.int64))


# ---- reference (idiomatic Python) -------------------------------------------------------
def _kruskal_forest(g: DetectorGraph3D, t: UFTables, grown, allowed) -> set[int]:
    """E2: edges of a min-length spanning forest of the grown edges (ascending (length, id);
    boundary edges only where allowed(e))."""
    B = g.boundary
    par: dict[int, int] = {}

    def f(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    forest = set()
    for e in sorted((e for e in grown if allowed(e)), key=lambda e: (int(t.length[e]), e)):
        a, b = f(int(g.edge_u[e])), f(int(g.edge_v[e]) if int(g.edge_v[e]) != B else B)
        if a != b:
            par[a] = b
            forest.add(e)
    return forest


def uf_decode_shot_reference(g: DetectorGraph3D, t: UFTables, dets_row: np.ndarray,
                             stats: dict | None = None) -> tuple[int, list[int], bool]:
    """Decode one shot. Returns (observable bits, sorted flipped edge ids, ok).
    If `stats` is given it receives the per-shot instrumentation (the reference is the
    instrumented implementation; the numba kernel runs the same algorithm uninstrumented):
      max_cluster  node count of the largest final cluster
      rounds       executed growth rounds (each completes >= 1 edge; idle rounds are skipped)
      idle_rounds  rounds skipped because no edge would complete (rounds + idle = round-by-round)
      lit          detection events       clusters     final clusters (distinct roots)
      labels       {node: root} for every node in a cluster"""
    n, B = g.n_det, g.n_det
    lit = [int(v) for v in np.flatnonzero(dets_row)]
    parent: dict[int, int] = {}
    rank: dict[int, int] = {}
    parity: dict[int, int] = {}
    touches: dict[int, bool] = {}
    size: dict[int, int] = {}

    def add_node(v: int, odd: int) -> None:
        parent[v], rank[v], parity[v], touches[v], size[v] = v, 0, odd, False, 1

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        parity[ra] ^= parity[rb]
        touches[ra] = touches[ra] or touches[rb]
        size[ra] += size[rb]

    visited: list[int] = []
    for v in lit:
        add_node(v, 1)
        visited.append(v)
    support: dict[int, int] = defaultdict(int)
    grown: set[int] = set()
    ok = True

    def incident(u: int):
        return [int(e) for e in g.inc_edge[u] if e >= 0]

    rounds = idle = 0
    while True:
        active = {find(v) for v in visited if parity[find(v)] == 1 and not touches[find(v)]}
        if not active:
            break
        rounds += 1
        if t.smallest:
            m = min(size[r] for r in active)
            active = {r for r in active if size[r] == m}
        # Skip-ahead: every frontier edge grows by its rate r_e (number of active endpoints)
        # per round, and nothing changes until some edge completes, so apply
        # delta = min ceil((L_e - support_e) / r_e) rounds at once.
        rate: dict[int, int] = {}
        for u in visited:
            if find(u) not in active:
                continue
            for e in incident(u):
                if e in grown:
                    continue
                rate[e] = rate.get(e, 0) + 1
        if not rate:
            ok = False  # an odd cluster with nothing left to grow (no boundary path)
            break
        delta = min(-(-(int(t.length[e]) - support[e]) // r) for e, r in rate.items())
        idle += delta - 1
        newly: list[int] = []
        for e, r in rate.items():
            support[e] = min(int(t.length[e]), support[e] + r * delta)
            if support[e] >= t.length[e]:
                newly.append(e)
        for e in newly:
            grown.add(e)
            u, v = int(g.edge_u[e]), int(g.edge_v[e])
            if v == B:
                touches[find(u)] = True
                continue
            for x in (u, v):
                if x not in parent:
                    add_node(x, 0)
                    visited.append(x)
            union(u, v)

    # Peeling on a canonical spanning forest of the grown edges. Only odd clusters may
    # use their grown boundary edges; an even cluster is peeled on its internal edges.
    def allowed(e):
        return not (int(g.edge_v[e]) == B and parity[find(int(g.edge_u[e]))] == 0)

    forest_edges = _kruskal_forest(g, t, grown, allowed) if t.wforest else {e for e in grown if allowed(e)}
    adj: dict[int, list[int]] = defaultdict(list)
    for e in sorted(forest_edges):
        u, v = int(g.edge_u[e]), int(g.edge_v[e])
        adj[u].append(e)
        adj[v].append(e)
    marked = {B}
    order: list[int] = []
    tree_edge: dict[int, int] = {}
    tree_parent: dict[int, int] = {}
    for root in [B] + sorted(parent):
        if root != B and root in marked:
            continue
        marked.add(root)
        queue = deque([root])
        while queue:
            x = queue.popleft()
            for e in adj[x]:
                y = int(g.edge_v[e]) if int(g.edge_u[e]) == x else int(g.edge_u[e])
                if y in marked:
                    continue
                marked.add(y)
                tree_edge[y], tree_parent[y] = e, x
                order.append(y)
                queue.append(y)

    defect = set(lit)
    flipped: set[int] = set()
    for y in reversed(order):
        if y in defect:
            e, p = tree_edge[y], tree_parent[y]
            flipped ^= {e}
            defect.discard(y)
            if p != B:
                defect ^= {p}
    if defect:
        ok = False
    if stats is not None:
        stats["max_cluster"] = max((size[find(v)] for v in parent), default=0)
        stats["rounds"], stats["idle_rounds"], stats["lit"] = rounds, idle, len(lit)
        stats["labels"] = {v: find(v) for v in parent}
        stats["clusters"] = len(set(stats["labels"].values()))
    bits = 0
    for e in flipped:
        bits ^= int(g.edge_obs_bits[e])
    return bits, sorted(flipped), ok


def correlated_lengths(t: UFTables, flipped) -> np.ndarray:
    """Per-shot lengths for pass 2: every partner f of a pass-1 edge e gets
    min(current length, length(P(f | e)))."""
    len2 = t.length.copy()
    for e in flipped:
        for k in range(t.corr_ptr[e], t.corr_ptr[e + 1]):
            f = t.corr_idx[k]
            len2[f] = min(len2[f], t.corr_len[k])
    return len2


def correlated_score(t: UFTables, flipped) -> int:
    """Correlated cost of a correction: sum of edge costs minus the pair credit of every
    unordered DEM partner pair (e, f) that it contains. Costs are t.score_len/score_credit
    (integer lengths by default; fixed-point fine weights with float_score)."""
    slen = t.score_len if t.score_len is not None else t.length
    scred = t.score_credit if t.score_credit is not None else t.corr_credit
    inside = set(int(e) for e in flipped)
    score = sum(int(slen[e]) for e in inside)
    for e in inside:
        for k in range(t.corr_ptr[e], t.corr_ptr[e + 1]):
            f = int(t.corr_idx[k])
            if f > e and f in inside:
                score -= int(scred[k])
    return score


def uf_correlated_shot_reference(g: DetectorGraph3D, t: UFTables, dets_row: np.ndarray,
                                 stats: dict | None = None) -> tuple[int, list[int], bool]:
    """Two-pass correlated UF on one shot (reference). Pass 2 reweights the DEM partners of
    pass-1 edges; its correction is kept only if its correlated score is strictly lower."""
    st1: dict = {}
    bits, flipped, ok = uf_decode_shot_reference(g, t, dets_row, st1)
    result = (bits, flipped, ok, st1)
    len2 = correlated_lengths(t, flipped)
    ran = kept = ran3 = kept3 = False
    rounds2 = 0
    if not np.array_equal(len2, t.length):
        ran = True
        st2: dict = {}
        b2, f2, ok2 = uf_decode_shot_reference(g, replace(t, length=len2), dets_row, st2)
        rounds2 = st2["rounds"]
        sc2 = correlated_score(t, f2)
        if sc2 < correlated_score(t, flipped):
            result = (b2, f2, ok2, st2)
            kept = True
            if t.passes >= 3:  # E4: reweight again from pass 2's correction
                len3 = correlated_lengths(t, f2)
                if not np.array_equal(len3, t.length):
                    ran3 = True
                    st3: dict = {}
                    b3, f3, ok3 = uf_decode_shot_reference(g, replace(t, length=len3), dets_row, st3)
                    if correlated_score(t, f3) < sc2:
                        result = (b3, f3, ok3, st3)
                        kept3 = True
    if stats is not None:
        stats.update(pass3_ran=ran3, pass3_kept=kept3, pass2_changed_obs=int(result[0]) != int(bits))
        changed_e = np.flatnonzero(len2 != t.length)
        lab1 = st1["labels"]
        stats["pass2_needed"] = any(int(x) in lab1 for e in changed_e for x in (g.edge_u[e], g.edge_v[e])
                                    if int(x) != g.boundary)
        stats.update(result[3])
        # pass-1 clusters that contain an endpoint of a reweighted edge
        changed = np.flatnonzero(len2 != t.length)
        labels = st1["labels"]
        touched = {labels[int(x)] for e in changed for x in (g.edge_u[e], g.edge_v[e]) if int(x) in labels}
        stats.update(pass2_ran=ran, pass2_kept=kept, touched_clusters=len(touched),
                     clusters_pass1=st1["clusters"], rounds_pass1=st1["rounds"], rounds_pass2=rounds2,
                     idle_rounds_pass1=st1["idle_rounds"], reweighted_edges=len(changed))
    return result[:3]


# ---- numba kernel ---------------------------------------------------------------------
# Shots are processed in chunks of CHUNK; each chunk allocates one scratch workspace and
# resets only the entries a shot touched, so a shot costs O(cluster size), not O(n + E).
CHUNK = 64


@njit(cache=True)
def _find(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        nxt = parent[x]
        parent[x] = root
        x = nxt
    return root


@njit(cache=True)
def uf_work(n, E):
    """Scratch workspace for _uf_shot (clean state; _uf_shot leaves it clean again)."""
    return (np.arange(n + 1).astype(np.int32),  # 0 parent
            np.zeros(n + 1, np.int32),           # 1 rank
            np.zeros(n + 1, np.uint8),           # 2 parity
            np.zeros(n + 1, np.bool_),           # 3 touches boundary
            np.ones(n + 1, np.int32),            # 4 cluster size (nodes)
            np.zeros(n + 1, np.bool_),           # 5 in_cluster
            np.empty(n, np.int32),               # 6 visited nodes
            np.zeros(E, np.int32),               # 7 support
            np.zeros(E, np.bool_),               # 8 grown
            np.empty(E, np.int32),               # 9 newly grown this round
            np.zeros(n + 1, np.bool_),           # 10 active
            np.empty(E, np.int32),               # 11 edges with support > 0
            np.zeros(n + 1, np.bool_),           # 12 marked (peeling BFS)
            np.empty(n + 1, np.int32),           # 13 tree_edge
            np.empty(n + 1, np.int32),           # 14 tree_parent
            np.empty(n, np.int32),               # 15 BFS order
            np.empty(n + 1, np.int32),           # 16 queue
            np.zeros(n + 1, np.bool_),           # 17 defect
            np.zeros(1, np.int32),               # 18 out: node count of the largest cluster
            np.empty(n + 1, np.int32),           # 19 out: flipped edges (peeling order)
            np.zeros(1, np.int32),               # 20 out: number of flipped edges
            np.zeros(E, np.int32),               # 21 rate: active endpoints of a frontier edge (round)
            np.empty(E, np.int32),               # 22 frontier edges of the round
            np.zeros(n + 1, np.int32),           # 23 (unused; kept for a stable workspace layout)
            np.empty(E, np.int32),               # 24 (unused; kept for a stable workspace layout)
            np.empty(n, np.int32),               # 25 (unused; kept for a stable workspace layout)
            np.empty(n, np.int32),               # 26 lit-index buffer for callers
            np.empty(n + 1, np.int32),           # 27 Kruskal union-find parents (weighted forest)
            np.empty(E, np.int64),               # 28 Kruskal sort keys (length * E + id)
            np.zeros(E, np.bool_))               # 29 forest edge flags


@njit(cache=True)
def extract_lits(row, buf):
    """Indices of the nonzero entries of a dense detection row -> buf[:k]; returns k.
    The only O(n) pass over a shot; everything downstream works on the k lit indices."""
    k = 0
    for v in range(row.shape[0]):
        if row[v]:
            buf[k] = v
            k += 1
    return k


@njit(cache=True)
def _uf_shot(lits, nlit, s, edge_u, edge_v, length, inc_edge, b_inc, obs_bits, corr, smallest, work,
             skip=False, wforest=False):
    """UF on the detection events lits[:nlit] (ascending order not required).
    wforest=True (E2): peel on a min-length spanning forest of the grown edges (Kruskal,
    ascending (length, id); boundary edges only for odd clusters) instead of the BFS forest.

    Two exact growth paths (same grown edges and clusters, so bit-identical results):
      skip=False  round by round, one pass per round (fastest at the default S = 4);
      skip=True   skip-ahead: while no edge completes the active set, and so each frontier
                  edge's rate r_e (number of active endpoints, 1 or 2), cannot change, so the
                  idle rounds before the next completion, delta = min ceil((L_e - s_e) / r_e),
                  are applied at once. The executed rounds no longer grow with the weight scale
                  S, but each costs three passes over the frontier (Phase B: 25-43% slower per
                  round), so this pays off only at large S."""
    (parent, rank, parity, touches, size, in_cluster, visited, support, grown, newly,
     active, touched, marked, tree_edge, tree_parent, order, queue, defect, stat, flips, nflips,
     rate, frontier, rem, gb, sortbuf, _litbuf, kpar, kkeys, forest) = work
    n = in_cluster.shape[0] - 1
    B = n
    D = inc_edge.shape[1]
    if nlit == 0:
        stat[0] = 0
        nflips[0] = 0
        return np.uint64(0), True
    nvis = 0
    for i in range(nlit):
        v = lits[i]
        parity[v] = 1
        in_cluster[v] = True
        visited[nvis] = v
        nvis += 1
    ntouched = 0

    ok = True
    while True:
        any_active = False
        for i in range(nvis):
            r = _find(parent, visited[i])
            act = parity[r] == 1 and not touches[r]
            active[visited[i]] = act
            any_active = any_active or act
        if not any_active:
            break
        if smallest:
            m = n + 1
            for i in range(nvis):
                if active[visited[i]]:
                    sz = size[_find(parent, visited[i])]
                    if sz < m:
                        m = sz
            for i in range(nvis):
                if active[visited[i]] and size[_find(parent, visited[i])] != m:
                    active[visited[i]] = False
        nnew = 0
        if skip:
            # frontier edges of the active clusters, with their rates
            nfr = 0
            for i in range(nvis):
                u = visited[i]
                if not active[u]:
                    continue
                for j in range(D):
                    e = inc_edge[u, j]
                    if e < 0:
                        break
                    if grown[e]:
                        continue
                    if rate[e] == 0:
                        frontier[nfr] = e
                        nfr += 1
                    rate[e] += 1
            if nfr == 0:
                ok = False  # an odd cluster with nothing left to grow (no boundary path)
                break
            delta = 1 << 30
            for k in range(nfr):
                e = frontier[k]
                need = length[e] - support[e]
                if rate[e] == 2:  # r is 1 or 2 (an edge has two endpoints): no integer division
                    need = (need + 1) >> 1
                if need < delta:
                    delta = need
            for k in range(nfr):
                e = frontier[k]
                if support[e] == 0:
                    touched[ntouched] = e
                    ntouched += 1
                s2 = support[e] + rate[e] * delta
                rate[e] = 0
                if s2 >= length[e]:
                    support[e] = length[e]
                    newly[nnew] = e
                    nnew += 1
                else:
                    support[e] = s2
        else:
            growable = False
            for i in range(nvis):
                u = visited[i]
                if not active[u]:
                    continue
                for j in range(D):
                    e = inc_edge[u, j]
                    if e < 0:
                        break
                    if grown[e] or support[e] >= length[e]:
                        continue
                    growable = True
                    if support[e] == 0:
                        touched[ntouched] = e
                        ntouched += 1
                    support[e] += 1
                    if support[e] >= length[e]:
                        newly[nnew] = e
                        nnew += 1
            if not growable:
                ok = False
                break
        for k in range(nnew):
            e = newly[k]
            grown[e] = True
            u = edge_u[e]
            v = edge_v[e]
            if v == B:
                touches[_find(parent, u)] = True
                continue
            if not in_cluster[v]:
                in_cluster[v] = True
                visited[nvis] = v
                nvis += 1
            if not in_cluster[u]:
                in_cluster[u] = True
                visited[nvis] = u
                nvis += 1
            ra = _find(parent, u)
            rb = _find(parent, v)
            if ra != rb:
                if rank[ra] < rank[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                if rank[ra] == rank[rb]:
                    rank[ra] += 1
                parity[ra] ^= parity[rb]
                touches[ra] = touches[ra] or touches[rb]
                size[ra] += size[rb]

    # Peeling forest: BFS from the boundary (grown boundary edges, ascending id), then from
    # the cluster nodes in ascending id; neighbors in ascending edge id. (Phase B tried sorted
    # lists of the visited nodes and grown boundary edges instead of these scans; both were
    # slower at d <= 9, so the scans stay.)
    nk = 0
    if wforest:
        E = edge_u.shape[0]
        for i in range(ntouched):
            e = touched[i]
            if not grown[e]:
                continue
            if edge_v[e] == B and parity[_find(parent, edge_u[e])] == 0:
                continue
            kkeys[nk] = np.int64(length[e]) * E + e
            nk += 1
        ks = kkeys[:nk]
        ks.sort()
        for i in range(nvis):
            kpar[visited[i]] = visited[i]
        kpar[B] = B
        for i in range(nk):
            e = np.int32(ks[i] % E)
            a = _find(kpar, edge_u[e])
            b = _find(kpar, edge_v[e])
            if a != b:
                kpar[a] = b
                forest[e] = True
    norder = 0
    for idx in range(-1, n):
        root = B if idx < 0 else idx
        if idx >= 0 and (not in_cluster[root] or marked[root]):
            continue
        marked[root] = True
        qh = 0
        qt = 0
        queue[qt] = root
        qt += 1
        while qh < qt:
            x = queue[qh]
            qh += 1
            if x == B:
                for j in range(b_inc.shape[0]):
                    e = b_inc[j]
                    if not (forest[e] if wforest else grown[e]):
                        continue
                    y = edge_u[e]
                    if marked[y] or parity[_find(parent, y)] == 0:
                        continue
                    marked[y] = True
                    tree_edge[y] = e
                    tree_parent[y] = x
                    order[norder] = y
                    norder += 1
                    queue[qt] = y
                    qt += 1
            else:
                for j in range(D):
                    e = inc_edge[x, j]
                    if e < 0:
                        break
                    if not (forest[e] if wforest else grown[e]):
                        continue
                    y = edge_v[e] if edge_u[e] == x else edge_u[e]
                    if y == B and parity[_find(parent, x)] == 0:
                        continue
                    if marked[y]:
                        continue
                    marked[y] = True
                    tree_edge[y] = e
                    tree_parent[y] = x
                    order[norder] = y
                    norder += 1
                    queue[qt] = y
                    qt += 1

    for i in range(nlit):
        defect[lits[i]] = True
    bits = np.uint64(0)
    record = corr.shape[0] > 0
    nf = 0
    for i in range(norder - 1, -1, -1):
        y = order[i]
        if defect[y]:
            e = tree_edge[y]
            p = tree_parent[y]
            bits ^= obs_bits[e]
            flips[nf] = e
            nf += 1
            if record:
                corr[s, e] ^= 1
            defect[y] = False
            if p != B:
                defect[p] = not defect[p]

    mx = 0
    for i in range(nvis):
        sz = size[_find(parent, visited[i])]
        if sz > mx:
            mx = sz
    stat[0] = mx
    nflips[0] = nf
    # check and restore the workspace (only touched entries)
    for i in range(nvis):
        v = visited[i]
        if defect[v]:
            ok = False
        parent[v] = v
        rank[v] = 0
        parity[v] = 0
        touches[v] = False
        size[v] = 1
        in_cluster[v] = False
        active[v] = False
        marked[v] = False
        defect[v] = False
    marked[B] = False
    for i in range(ntouched):
        e = touched[i]
        support[e] = 0
        grown[e] = False
        forest[e] = False
    return bits, ok


@njit(parallel=True, cache=True)
def _uf_batch(dets, edge_u, edge_v, length, inc_edge, b_inc, obs_bits, corr, out_bits, out_ok,
              smallest, out_size, skip, tkeys, tvals, out_table, wforest):
    S = dets.shape[0]
    n = dets.shape[1]
    E = edge_u.shape[0]
    nchunk = (S + CHUNK - 1) // CHUNK
    for ci in prange(nchunk):
        work = uf_work(n, E)
        lits = work[26]
        for s in range(ci * CHUNK, min(S, (ci + 1) * CHUNK)):
            nlit = extract_lits(dets[s], lits)
            hit, tb = table_lookup(lits, nlit, tkeys, tvals)
            if hit:
                out_bits[s] = tb
                out_table[s] = True
                continue
            b, ok = _uf_shot(lits, nlit, s, edge_u, edge_v, length, inc_edge, b_inc, obs_bits, corr,
                             smallest, work, skip, wforest)
            out_bits[s] = b
            out_ok[s] = ok
            out_size[s] = work[18][0]


@njit(cache=True)
def _corr_score(flips, nf, length, cptr, cidx, ccredit, inside):
    """Correlated score of a correction (see correlated_score); `inside` is all-False on entry
    and restored before returning."""
    score = 0
    for i in range(nf):
        e = flips[i]
        inside[e] = True
        score += length[e]
    for i in range(nf):
        e = flips[i]
        for k in range(cptr[e], cptr[e + 1]):
            f = cidx[k]
            if f > e and inside[f]:
                score -= ccredit[k]
    for i in range(nf):
        inside[flips[i]] = False
    return score


@njit(cache=True)
def _uf_correlated_shot(lits, nlit, s, edge_u, edge_v, length, inc_edge, b_inc, obs_bits, corr, smallest,
                        work, cptr, cidx, clen, ccredit, len2, mod, inside, keep, stat_keep, skip, wforest,
                        slen, scred, passes):
    """Pass 1 with `length`; pass 2 with the partners of pass-1 edges shortened; keep pass 2
    only if its correlated score is strictly lower. `len2` must equal `length` on entry and
    is restored. On return work[19..20] (flips) and work[18] (max cluster) describe the
    kept pass."""
    b, ok = _uf_shot(lits, nlit, s, edge_u, edge_v, length, inc_edge, b_inc, obs_bits, corr, smallest, work,
                     skip, wforest)
    flips = work[19]
    nf1 = work[20][0]
    nm = 0
    for i in range(nf1):
        e = flips[i]
        for k in range(cptr[e], cptr[e + 1]):
            f = cidx[k]
            L = clen[k]
            if L < len2[f]:
                if len2[f] == length[f]:
                    mod[nm] = f
                    nm += 1
                len2[f] = L
    if nm == 0:
        return b, ok
    score1 = _corr_score(flips, nf1, slen, cptr, cidx, scred, inside)
    for i in range(nf1):
        keep[i] = flips[i]
    stat_keep[0] = work[18][0]
    b2, ok2 = _uf_shot(lits, nlit, s, edge_u, edge_v, len2, inc_edge, b_inc, obs_bits, corr, smallest, work,
                       skip, wforest)
    for i in range(nm):
        len2[mod[i]] = length[mod[i]]
    score2 = _corr_score(flips, work[20][0], slen, cptr, cidx, scred, inside)
    if not score2 < score1:
        for i in range(nf1):  # restore pass 1 as the reported correction
            flips[i] = keep[i]
        work[20][0] = nf1
        work[18][0] = stat_keep[0]
        return b, ok
    if passes < 3:
        return b2, ok2
    # E4, pass 3: reweight from pass 2's correction and decode once more
    nf2 = work[20][0]
    for i in range(nf2):
        keep[i] = flips[i]
    stat_keep[0] = work[18][0]
    nm = 0
    for i in range(nf2):
        e = keep[i]
        for k in range(cptr[e], cptr[e + 1]):
            f = cidx[k]
            L = clen[k]
            if L < len2[f]:
                if len2[f] == length[f]:
                    mod[nm] = f
                    nm += 1
                len2[f] = L
    if nm == 0:
        return b2, ok2
    b3, ok3 = _uf_shot(lits, nlit, s, edge_u, edge_v, len2, inc_edge, b_inc, obs_bits, corr, smallest, work,
                       skip, wforest)
    for i in range(nm):
        len2[mod[i]] = length[mod[i]]
    score3 = _corr_score(flips, work[20][0], slen, cptr, cidx, scred, inside)
    if score3 < score2:
        return b3, ok3
    for i in range(nf2):  # restore pass 2
        flips[i] = keep[i]
    work[20][0] = nf2
    work[18][0] = stat_keep[0]
    return b2, ok2


@njit(parallel=True, cache=True)
def _uf_correlated_batch(dets, edge_u, edge_v, length, inc_edge, b_inc, obs_bits, corr, out_bits,
                         out_ok, smallest, out_size, cptr, cidx, clen, ccredit, skip, tkeys, tvals,
                         out_table, wforest, slen, scred, passes):
    S = dets.shape[0]
    n = dets.shape[1]
    E = edge_u.shape[0]
    nchunk = (S + CHUNK - 1) // CHUNK
    record = corr.shape[0] > 0
    for ci in prange(nchunk):
        work = uf_work(n, E)
        lits = work[26]
        len2 = length.copy()
        mod = np.empty(E, np.int32)
        inside = np.zeros(E, np.bool_)
        keep = np.empty(n + 1, np.int32)
        stat_keep = np.zeros(1, np.int32)
        none = np.zeros((0, 0), np.uint8)
        for s in range(ci * CHUNK, min(S, (ci + 1) * CHUNK)):
            nlit = extract_lits(dets[s], lits)
            hit, tb = table_lookup(lits, nlit, tkeys, tvals)
            if hit:
                out_bits[s] = tb
                out_table[s] = True
                continue
            b, ok = _uf_correlated_shot(lits, nlit, s, edge_u, edge_v, length, inc_edge, b_inc, obs_bits,
                                        none, smallest, work, cptr, cidx, clen, ccredit, len2, mod,
                                        inside, keep, stat_keep, skip, wforest, slen, scred, passes)
            out_bits[s] = b
            out_ok[s] = ok
            out_size[s] = work[18][0]
            if record:  # correction of the final pass (each tree edge is flipped at most once)
                for i in range(work[20][0]):
                    corr[s, work[19][i]] = 1


@njit(cache=True)
def table_lookup(lits, nlit, keys, vals):
    """(found, observable bits) for the events lits[:nlit] in the single-fault table
    (graph.fault_keys / fault_obs; keys are 4 x 16-bit sorted ids, unused slots 0xFFFF)."""
    if nlit < 1 or nlit > 4 or keys.shape[0] == 0:
        return False, np.uint64(0)
    a0 = lits[0]
    a1 = lits[1] if nlit > 1 else 0xFFFF
    a2 = lits[2] if nlit > 2 else 0xFFFF
    a3 = lits[3] if nlit > 3 else 0xFFFF
    # sort the (at most 4) ids; padding 0xFFFF sorts last
    if a0 > a1:
        a0, a1 = a1, a0
    if a2 > a3:
        a2, a3 = a3, a2
    if a0 > a2:
        a0, a2 = a2, a0
    if a1 > a3:
        a1, a3 = a3, a1
    if a1 > a2:
        a1, a2 = a2, a1
    key = np.uint64(a0) | (np.uint64(a1) << np.uint64(16)) | (np.uint64(a2) << np.uint64(32)) \
        | (np.uint64(a3) << np.uint64(48))
    i = np.searchsorted(keys, key)
    if i < keys.shape[0] and keys[i] == key:
        return True, vals[i]
    return False, np.uint64(0)


def table_lookup_reference(g: DetectorGraph3D, dets_row: np.ndarray):
    """Reference lookup: (found, observable bits)."""
    from src.decoders.cube3d.graph import TABLE_MAX_LIT, pack_syndrome
    lit = np.flatnonzero(dets_row)
    if not 1 <= len(lit) <= TABLE_MAX_LIT or g.fault_keys is None or len(g.fault_keys) == 0:
        return False, 0
    key = np.uint64(pack_syndrome(lit))
    i = int(np.searchsorted(g.fault_keys, key))
    if i < len(g.fault_keys) and g.fault_keys[i] == key:
        return True, int(g.fault_obs[i])
    return False, 0


def bits_to_bool(bits: np.ndarray, n_obs: int) -> np.ndarray:
    return ((bits[:, None] >> np.arange(n_obs, dtype=np.uint64)) & np.uint64(1)).astype(bool)


@dataclass(eq=False)
class UFResult:
    predictions: np.ndarray       # (shots, n_obs) bool
    ok: np.ndarray                # (shots,) bool: every defect peeled
    correction: np.ndarray | None  # (shots, E) uint8 if requested
    max_cluster: np.ndarray | None = None  # (shots,) int32 nodes in the largest final cluster
    table: np.ndarray | None = None  # (shots,) bool: answered by the single-fault table


class UnionFind3D:
    """UF engine on a DetectorGraph3D (usable on any detection-event set, e.g. residuals)."""

    def __init__(self, graph: DetectorGraph3D, config: UnionFind3DConfig | None = None):
        self.graph = graph
        self.config = config or UnionFind3DConfig()
        c = self.config
        t = replace(build_uf_tables(graph, c.scale, c.growth), skip=c.skip_ahead, wforest=c.weighted_forest,
                    passes=c.passes)
        if c.float_score:
            sl, sc = float_score_tables(graph, t)
            t = replace(t, score_len=sl, score_credit=sc)
        self.tables = t

    def run(self, detection_events: np.ndarray, record_correction: bool = False,
            backend: str | None = None) -> UFResult:
        g, t = self.graph, self.tables
        dets = np.ascontiguousarray(detection_events, dtype=np.uint8)
        if dets.ndim != 2 or dets.shape[1] != g.n_det:
            raise ValueError(f"expected (shots, {g.n_det}), got {dets.shape}")
        S = dets.shape[0]
        backend = backend or self.config.backend
        corr = np.zeros((S, g.n_edges) if record_correction else (0, 0), dtype=np.uint8)
        bits = np.zeros(S, dtype=np.uint64)
        ok = np.ones(S, dtype=bool)
        size = np.zeros(S, dtype=np.int32)
        table = np.zeros(S, dtype=bool)
        use_table = self.config.table and g.fault_keys is not None
        tkeys = g.fault_keys if use_table else np.zeros(0, np.uint64)
        tvals = g.fault_obs if use_table else np.zeros(0, np.uint64)
        correlated = self.config.correlated
        if backend == "numba" and correlated:
            _uf_correlated_batch(dets, g.edge_u, g.edge_v, t.length, g.inc_edge, t.b_inc,
                                 g.edge_obs_bits, corr, bits, ok, t.smallest, size,
                                 t.corr_ptr, t.corr_idx, t.corr_len, t.corr_credit, t.skip, tkeys, tvals,
                                 table, t.wforest, t.score_len, t.score_credit, t.passes)
        elif backend == "numba":
            _uf_batch(dets, g.edge_u, g.edge_v, t.length, g.inc_edge, t.b_inc,
                      g.edge_obs_bits, corr, bits, ok, t.smallest, size, t.skip, tkeys, tvals, table,
                      t.wforest)
        else:
            shot_fn = uf_correlated_shot_reference if correlated else uf_decode_shot_reference
            for s in range(S):
                if use_table:
                    hit, tb = table_lookup_reference(g, dets[s])
                    if hit:
                        bits[s], table[s] = tb, True
                        continue
                st: dict = {}
                b, flipped, good = shot_fn(g, t, dets[s], st)
                bits[s], ok[s], size[s] = b, good, st["max_cluster"]
                if record_correction:
                    corr[s, flipped] = 1
        return UFResult(bits_to_bool(bits, g.n_obs), ok, corr if record_correction else None, size, table)


class UnionFind3DDecoder(Decoder):
    name = "union_find_3d"

    def __init__(self, dem: stim.DetectorErrorModel, config: UnionFind3DConfig | None = None):
        super().__init__(dem)
        self.engine = UnionFind3D(build_detector_graph(dem), config)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        return self.engine.run(self._check_input(detection_events)).predictions


def set_threads(n: int) -> None:
    """Thread count for the numba backend."""
    numba.set_num_threads(n)


class UnionFind3DCorrelatedDecoder(UnionFind3DDecoder):
    """Two-pass correlated UF: pass 2 reweights the DEM partners of pass-1 edges."""

    name = "uf3d_correlated"
    # With the single-fault table every single DEM fault decodes exactly (without it, two
    # three-component hook instructions at circuit-level d=3 were lost to a pairing of two
    # likelier correlated faults, as correlated PyMatching loses 1 of 286).
    single_fault_allowance = 0

    def __init__(self, dem: stim.DetectorErrorModel, config: UnionFind3DConfig | None = None):
        super().__init__(dem, config or UnionFind3DConfig(correlated=True, table=True))
