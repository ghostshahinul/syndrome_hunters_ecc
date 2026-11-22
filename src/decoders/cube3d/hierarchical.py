"""Hierarchical: five stages on the shared 3D detector graph, cheapest first.

Per shot (lit = number of detection events), with the defaults (gate="whole_shot", K=6,
table=True, deep tier = correlated UF with the auto pass-2 gate):
  1. Empty shot           lit == 0 -> predict no flip.
  2. Single-fault table   1 <= lit <= 4 and the syndrome is exactly one DEM fault's ->
                          that fault's observable flip (exact lookup; no match -> go on).
  3. Cube3D-Local         lit <= K: apply the local rules R1-R3. Every event cleared ->
     fast path            fast-path prediction, UF never runs. Otherwise the whole-shot
                          gate discards the local work and the shot goes to stage 4
                          (lit > K skips the rules and goes straight to stage 4).
  4. Correlated UF,       weighted union-find on the whole shot with the base edge lengths.
     pass 1
  5. Correlated UF,       reweight the partners of pass 1's correction from the DEM's
     pass 2              joint probabilities and run UF again, keeping pass 2 only if its
                          correlated score is strictly lower. It runs only if pass 1's
                          largest cluster has more than floor(rounds^2 / 6.5) nodes, and is
                          skipped exactly (same output) when no shortened edge touches a
                          node pass 1 visited.

gate="partial" (registry name hierarchical_partial) instead sends only the events the local
rules left to UF and XORs the local and UF bits; there K defaults to "no cap".

Backends:
  numba      one fused prange kernel over shots (local rules, gate and UF in one pass).
  reference  NumPy Cube3D-Local + the Python UF reference. Bit-identical to numba (tested).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import stim
from numba import njit, prange

from src.decoders.base import Decoder
from src.decoders.cube3d.graph import DetectorGraph3D, build_detector_graph
from src.decoders.cube3d.local import Cube3DLocal, Cube3DLocalConfig, local_shot, local_work
from src.decoders.cube3d.union_find import (CHUNK, UnionFind3D, UnionFind3DConfig, _uf_correlated_shot,
                                            _uf_shot, bits_to_bool, extract_lits, table_lookup,
                                            table_lookup_reference, uf_correlated_shot_reference,
                                            uf_decode_shot_reference, uf_work)

GATES = ("whole_shot", "partial")
BACKENDS = ("reference", "numba")
DEFAULT_K = {"whole_shot": 6, "partial": None}


@dataclass(frozen=True)
class HierarchicalConfig:
    gate: str = "whole_shot"
    K: int | None = None          # None -> DEFAULT_K[gate] (None there = no cap)
    backend: str = "numba"
    strict_r1: bool = False       # fast-path R1 pairs must be isolated within distance 2
    table: bool = True            # 1-4-event shots matching a single DEM fault: answer from a table
    # Deep tier: correlated two-pass UF by default (item 4: it beats plain PyMatching's accuracy
    # at d = 7/9 at about 2x its single-thread time). UnionFind3DConfig() gives plain UF.
    # Final round: pass 2 only when pass 1's largest cluster exceeds floor(rounds^2 / 6.5) nodes
    # (LER within the 95% CI of always running pass 2 at p = 0.5%, d = 3-9; 14-23% faster).
    uf: UnionFind3DConfig = field(default_factory=lambda: UnionFind3DConfig(correlated=True, pass2_gate=-1))

    def __post_init__(self) -> None:
        if self.gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}, got {self.gate!r}")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.K is not None and self.K < 0:
            raise ValueError(f"K must be >= 0, got {self.K}")


@dataclass(eq=False)
class HierarchicalResult:
    predictions: np.ndarray   # (shots, n_obs) bool
    fast: np.ndarray          # (shots,) bool: decoded by the fast path alone (incl. lit == 0)
    uf_called: np.ndarray     # (shots,) bool: UnionFind3D ran on this shot
    lit_count: np.ndarray     # (shots,) int32
    cleared: np.ndarray       # (shots,) int32 events cleared by the local rules
    ok: np.ndarray            # (shots,) bool: UF peeled every defect (True where UF not called)
    max_cluster: np.ndarray | None = None  # (shots,) int32 largest UF cluster (0 if UF not called)
    table: np.ndarray | None = None        # (shots,) bool: answered by the single-fault table

    @property
    def table_fraction(self) -> float:
        return float(self.table.mean()) if self.table is not None and self.table.size else 0.0

    @property
    def fast_fraction(self) -> float:
        """Fraction of shots the fast path decoded alone (lit == 0 shots included)."""
        return float(self.fast.mean()) if self.fast.size else 0.0

    @property
    def fast_fraction_nontrivial(self) -> float:
        """Same, over shots with at least one detection event."""
        m = self.lit_count > 0
        return float(self.fast[m].mean()) if m.any() else 0.0

    @property
    def cleared_fraction(self) -> float:
        """Fraction of all detection events removed by the local rules."""
        tot = int(self.lit_count.sum())
        return float(self.cleared.sum()) / tot if tot else 0.0


@njit(cache=True)
def _deep(lits, nlit, s, edge_u, edge_v, length, inc_edge, b_inc, obs_bits, corr, smallest, uw, correlated,
          cptr, cidx, clen, ccredit, len2, mod, inside, keep, stat_keep, wforest, slen, scred, passes, p2skip,
          p2gate):
    """The deep tier on the detection events lits[:nlit]: plain or correlated two-pass UF."""
    if correlated:
        return _uf_correlated_shot(lits, nlit, s, edge_u, edge_v, length, inc_edge, b_inc, obs_bits, corr,
                                   smallest, uw, cptr, cidx, clen, ccredit, len2, mod, inside, keep,
                                   stat_keep, False, wforest, slen, scred, passes, p2skip, p2gate)
    return _uf_shot(lits, nlit, s, edge_u, edge_v, length, inc_edge, b_inc, obs_bits, corr, smallest, uw,
                    False, wforest)


@njit(parallel=True, cache=True)
def _hier_batch(dets, K, partial, strict_r1,
                nbr, nbr_edge, ball2, boundary_edge, color, n_colors,
                edge_u, edge_v, length, inc_edge, b_inc, smallest, obs_bits,
                out_bits, out_fast, out_uf, out_cleared, out_ok, out_lit, out_size,
                correlated, cptr, cidx, clen, ccredit, tkeys, tvals, out_table, wforest, slen, scred, passes,
                p2skip, p2gate):
    S = dets.shape[0]
    n = dets.shape[1]
    E = edge_u.shape[0]
    corr = np.zeros((0, 0), np.uint8)
    nchunk = (S + CHUNK - 1) // CHUNK
    for ci in prange(nchunk):
        uw = uf_work(n, E)
        len2 = length.copy()
        mod = np.empty(E, np.int32)
        inside = np.zeros(E, np.bool_)
        keep = np.empty(n + 1, np.int32)
        stat_keep = np.zeros(1, np.int32)
        lw = local_work(n, n_colors)
        lit = np.zeros(n, np.uint8)   # dense shot for the local rules, all-zero between shots
        lits = np.empty(n, np.int32)  # lit indices of the shot (extracted once)
        rlits = np.empty(n, np.int32)  # residual lit indices (partial gate)
        cl = np.zeros(3, np.int32)
        for s in range(ci * CHUNK, min(S, (ci + 1) * CHUNK)):
            c = extract_lits(dets[s], lits)
            out_lit[s] = c
            if c == 0:
                out_fast[s] = True
                continue
            hit, tb = table_lookup(lits, c, tkeys, tvals)
            if hit:  # exactly the syndrome of one DEM fault: its observable flip
                out_bits[s] = tb
                out_fast[s] = True
                out_table[s] = True
                continue
            if c > K:
                b, ok = _deep(lits, c, s, edge_u, edge_v, length, inc_edge, b_inc, obs_bits, corr, smallest,
                              uw, correlated, cptr, cidx, clen, ccredit, len2, mod, inside, keep, stat_keep,
                              wforest, slen, scred, passes, p2skip, p2gate)
                out_bits[s] = b
                out_uf[s] = True
                out_ok[s] = ok
                out_size[s] = uw[18][0]
                continue
            cl[0] = 0
            cl[1] = 0
            cl[2] = 0
            for i in range(c):
                lit[lits[i]] = 1
            bl, rem = local_shot(lit, lits, c, nbr, nbr_edge, ball2, boundary_edge, color, n_colors,
                                 obs_bits, cl, strict_r1, lw)
            out_cleared[s] = cl[0] + cl[1] + cl[2]
            nr = 0
            for i in range(c):  # residual events, then restore the all-zero dense buffer
                v = lits[i]
                if lit[v]:
                    rlits[nr] = v
                    nr += 1
                    lit[v] = 0
            if rem == 0:
                out_bits[s] = bl
                out_fast[s] = True
                continue
            out_uf[s] = True
            if partial:
                b, ok = _deep(rlits, nr, s, edge_u, edge_v, length, inc_edge, b_inc, obs_bits, corr, smallest,
                              uw, correlated, cptr, cidx, clen, ccredit, len2, mod, inside, keep, stat_keep,
                              wforest, slen, scred, passes, p2skip, p2gate)
                out_bits[s] = bl ^ b
            else:
                b, ok = _deep(lits, c, s, edge_u, edge_v, length, inc_edge, b_inc, obs_bits, corr, smallest,
                              uw, correlated, cptr, cidx, clen, ccredit, len2, mod, inside, keep, stat_keep,
                              wforest, slen, scred, passes, p2skip, p2gate)
                out_bits[s] = b
            out_ok[s] = ok
            out_size[s] = uw[18][0]


class Hierarchical:
    """Two-tier engine on a DetectorGraph3D."""

    def __init__(self, graph: DetectorGraph3D, config: HierarchicalConfig | None = None):
        self.graph = graph
        self.config = config or HierarchicalConfig()
        self.local = Cube3DLocal(graph, Cube3DLocalConfig(strict_r1=self.config.strict_r1))
        self.uf = UnionFind3D(graph, self.config.uf)

    def effective_K(self, gate: str, K: int | None) -> int:
        if K is None:
            K = self.config.K if gate == self.config.gate else None
        if K is None:
            K = DEFAULT_K[gate]
        return self.graph.n_det if K is None else min(int(K), self.graph.n_det)

    def run(self, detection_events: np.ndarray, K: int | None = None, gate: str | None = None,
            backend: str | None = None) -> HierarchicalResult:
        g = self.graph
        dets = np.ascontiguousarray(detection_events, dtype=np.uint8)
        if dets.ndim != 2 or dets.shape[1] != g.n_det:
            raise ValueError(f"expected (shots, {g.n_det}), got {dets.shape}")
        gate = gate or self.config.gate
        if gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
        K = self.effective_K(gate, K)
        backend = backend or self.config.backend
        S = dets.shape[0]
        bits = np.zeros(S, np.uint64)
        fast = np.zeros(S, bool)
        uf_called = np.zeros(S, bool)
        cleared = np.zeros(S, np.int32)
        ok = np.ones(S, bool)
        lit_count = np.zeros(S, np.int32)
        size = np.zeros(S, np.int32)
        table = np.zeros(S, bool)
        use_table = self.config.table and g.fault_keys is not None
        tkeys = g.fault_keys if use_table else np.zeros(0, np.uint64)
        tvals = g.fault_obs if use_table else np.zeros(0, np.uint64)
        if backend == "numba":
            lt, ut = self.local.tables, self.uf.tables
            _hier_batch(dets, K, gate == "partial", self.config.strict_r1,
                        g.nbr, g.nbr_edge, lt.ball2, g.boundary_edge, lt.color, lt.n_colors,
                        g.edge_u, g.edge_v, ut.length, g.inc_edge, ut.b_inc, ut.smallest,
                        g.edge_obs_bits, bits, fast, uf_called, cleared, ok, lit_count, size,
                        self.config.uf.correlated, ut.corr_ptr, ut.corr_idx, ut.corr_len, ut.corr_credit,
                        tkeys, tvals, table, ut.wforest, ut.score_len, ut.score_credit, ut.passes, ut.p2skip,
                        ut.p2gate)
        else:
            bits, fast, uf_called, cleared, ok, size, table = self._run_reference(dets, K, gate, use_table)
            lit_count = dets.sum(axis=1, dtype=np.int32)
        return HierarchicalResult(bits_to_bool(bits, g.n_obs), fast, uf_called, lit_count, cleared, ok,
                                  size, table)

    def _run_reference(self, dets, K, gate, use_table=False):
        g = self.graph
        loc = self.local.run(dets, K=K, record_residual=True, backend="reference")
        S = dets.shape[0]
        bits = np.zeros(S, np.uint64)
        fast = loc.accepted.copy()
        bits[fast] = loc.obs_bits[fast]
        uf_called = ~fast
        cleared = sum(loc.cleared.values()).astype(np.int32)
        ok = np.ones(S, bool)
        size = np.zeros(S, np.int32)
        table = np.zeros(S, bool)
        if use_table:  # the table is consulted before the local rules
            for s in range(S):
                hit, tb = table_lookup_reference(g, dets[s])
                if hit:
                    table[s] = fast[s] = True
                    uf_called[s] = False
                    bits[s] = np.uint64(tb)
                    cleared[s] = 0
        for s in np.flatnonzero(uf_called):
            row = loc.residual[s] if gate == "partial" else dets[s]
            st: dict = {}
            deep = uf_correlated_shot_reference if self.config.uf.correlated else uf_decode_shot_reference
            b, _, good = deep(g, self.uf.tables, np.asarray(row, np.uint8), st)
            bits[s] = (loc.obs_bits[s] ^ np.uint64(b)) if gate == "partial" else np.uint64(b)
            ok[s] = good
            size[s] = st["max_cluster"]
        return bits, fast, uf_called, cleared, ok, size, table


class HierarchicalDecoder(Decoder):
    """Registry entry: whole-shot gate (the spec's design)."""

    name = "hierarchical"
    gate = "whole_shot"

    def __init__(self, dem: stim.DetectorErrorModel, config: HierarchicalConfig | None = None):
        super().__init__(dem)
        self.engine = Hierarchical(build_detector_graph(dem), config or HierarchicalConfig(gate=self.gate))

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        return self.engine.run(self._check_input(detection_events)).predictions


class HierarchicalPartialDecoder(HierarchicalDecoder):
    """Registry entry: partial gate (local rules strip isolated groups, UF decodes the rest)."""

    name = "hierarchical_partial"
    gate = "partial"
