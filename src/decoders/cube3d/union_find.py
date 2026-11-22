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
from dataclasses import dataclass

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

    def __post_init__(self) -> None:
        if self.backend not in UF_BACKENDS:
            raise ValueError(f"backend must be one of {UF_BACKENDS}, got {self.backend!r}")
        if self.growth not in UF_GROWTH:
            raise ValueError(f"growth must be one of {UF_GROWTH}, got {self.growth!r}")
        if self.scale < 1:
            raise ValueError(f"scale must be >= 1, got {self.scale}")


@dataclass(frozen=True, eq=False)
class UFTables:
    length: np.ndarray   # (E,) int32 integer edge lengths
    b_inc: np.ndarray    # (B,) int32 boundary edges, ascending id
    smallest: bool = False  # growth rule: only minimum-size active clusters grow


def build_uf_tables(g: DetectorGraph3D, scale: int = 4, growth: str = "all") -> UFTables:
    w = g.edge_weight
    length = np.maximum(1, np.rint(float(scale) * w / w.min())).astype(np.int32)
    b_inc = np.flatnonzero(g.edge_v == g.boundary).astype(np.int32)
    return UFTables(length, b_inc, growth == "smallest")


# ---- reference (idiomatic Python) -------------------------------------------------------
def uf_decode_shot_reference(g: DetectorGraph3D, t: UFTables, dets_row: np.ndarray
                             ) -> tuple[int, list[int], bool]:
    """Decode one shot. Returns (observable bits, sorted flipped edge ids, ok)."""
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

    while True:
        active = {find(v) for v in visited if parity[find(v)] == 1 and not touches[find(v)]}
        if not active:
            break
        if t.smallest:
            m = min(size[r] for r in active)
            active = {r for r in active if size[r] == m}
        newly: list[int] = []
        grew = False
        for u in visited:
            if find(u) not in active:
                continue
            for e in incident(u):
                if e in grown or support[e] >= t.length[e]:
                    continue
                grew = True
                support[e] += 1
                if support[e] >= t.length[e]:
                    newly.append(e)
        if not grew:
            ok = False  # an odd cluster with nothing left to grow (no boundary path)
            break
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
    adj: dict[int, list[int]] = defaultdict(list)
    for e in sorted(grown):
        u, v = int(g.edge_u[e]), int(g.edge_v[e])
        if v == B and parity[find(u)] == 0:
            continue
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
    bits = 0
    for e in flipped:
        bits ^= int(g.edge_obs_bits[e])
    return bits, sorted(flipped), ok


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
            np.zeros(n + 1, np.bool_))           # 17 defect


@njit(cache=True)
def _uf_shot(row, s, edge_u, edge_v, length, inc_edge, b_inc, obs_bits, corr, smallest, work):
    (parent, rank, parity, touches, size, in_cluster, visited, support, grown, newly,
     active, touched, marked, tree_edge, tree_parent, order, queue, defect) = work
    n = row.shape[0]
    B = n
    D = inc_edge.shape[1]
    nvis = 0
    for v in range(n):
        if row[v]:
            parity[v] = 1
            in_cluster[v] = True
            visited[nvis] = v
            nvis += 1
    if nvis == 0:
        return np.uint64(0), True
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

    # Peeling forest: BFS from boundary, then from cluster nodes in ascending id.
    norder = 0
    for step in range(n + 1):
        root = B if step == 0 else step - 1
        if step > 0 and (not in_cluster[root] or marked[root]):
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
                    if not grown[e]:
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
                    if not grown[e]:
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

    for i in range(nvis):
        v = visited[i]
        defect[v] = row[v] != 0
    bits = np.uint64(0)
    record = corr.shape[0] > 0
    for i in range(norder - 1, -1, -1):
        y = order[i]
        if defect[y]:
            e = tree_edge[y]
            p = tree_parent[y]
            bits ^= obs_bits[e]
            if record:
                corr[s, e] ^= 1
            defect[y] = False
            if p != B:
                defect[p] = not defect[p]

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
    return bits, ok


@njit(parallel=True, cache=True)
def _uf_batch(dets, edge_u, edge_v, length, inc_edge, b_inc, obs_bits, corr, out_bits, out_ok,
              smallest):
    S = dets.shape[0]
    n = dets.shape[1]
    E = edge_u.shape[0]
    nchunk = (S + CHUNK - 1) // CHUNK
    for ci in prange(nchunk):
        work = uf_work(n, E)
        for s in range(ci * CHUNK, min(S, (ci + 1) * CHUNK)):
            b, ok = _uf_shot(dets[s], s, edge_u, edge_v, length, inc_edge, b_inc, obs_bits, corr,
                             smallest, work)
            out_bits[s] = b
            out_ok[s] = ok


def bits_to_bool(bits: np.ndarray, n_obs: int) -> np.ndarray:
    return ((bits[:, None] >> np.arange(n_obs, dtype=np.uint64)) & np.uint64(1)).astype(bool)


@dataclass(eq=False)
class UFResult:
    predictions: np.ndarray       # (shots, n_obs) bool
    ok: np.ndarray                # (shots,) bool: every defect peeled
    correction: np.ndarray | None  # (shots, E) uint8 if requested


class UnionFind3D:
    """UF engine on a DetectorGraph3D (usable on any detection-event set, e.g. residuals)."""

    def __init__(self, graph: DetectorGraph3D, config: UnionFind3DConfig | None = None):
        self.graph = graph
        self.config = config or UnionFind3DConfig()
        self.tables = build_uf_tables(graph, self.config.scale, self.config.growth)

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
        if backend == "numba":
            _uf_batch(dets, g.edge_u, g.edge_v, t.length, g.inc_edge, t.b_inc,
                      g.edge_obs_bits, corr, bits, ok, t.smallest)
        else:
            for s in range(S):
                b, flipped, good = uf_decode_shot_reference(g, t, dets[s])
                bits[s], ok[s] = b, good
                if record_correction:
                    corr[s, flipped] = 1
        return UFResult(bits_to_bool(bits, g.n_obs), ok, corr if record_correction else None)


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
