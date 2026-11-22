"""Ensemble UF: M UnionFind3D variants on the same shot, keep the cheapest correction.

Variant v (v = 0 .. M-1):
  - integer edge lengths from weights w_e * exp(sigma * z_{v,e}), z ~ N(0, 1) drawn with the
    fixed seed `seed + v`; variant 0 is unperturbed (sigma = 0);
  - growth rule "smallest" for odd v when mix_growth=True, otherwise "all";
  - optionally correlated two-pass UF (item 1), using that variant's lengths.
Every variant's correction is scored under the ORIGINAL lengths: the sum of lengths, minus the
DEM pair credits when correlated (see union_find.correlated_score). The lowest score wins, and
ties go to the lowest v. So variant 0 alone reproduces UnionFind3D (or UF3D correlated), and
more variants can only find corrections that are cheaper under the original weights.

Parallelism:
  parallel="variants": shots in sequence, numba prange over the M variants of a shot, so one
    shot uses up to M cores ("cores used per shot" = min(M, threads)).
  parallel="shots":    numba prange over chunks of shots; the M variants of a shot run in
    sequence on one core (1 core per shot).
Both give identical results, and both are bit-identical to the Python reference.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import stim
from numba import njit, prange

from src.decoders.base import Decoder
from src.decoders.cube3d.graph import DetectorGraph3D, build_detector_graph
from src.decoders.cube3d.union_find import (_corr_score, _uf_correlated_shot, _uf_shot, bits_to_bool, extract_lits,
                                            build_uf_tables, correlated_score, length_of,
                                            uf_correlated_shot_reference, uf_decode_shot_reference)

PARALLEL_MODES = ("variants", "shots")


@dataclass(frozen=True)
class EnsembleConfig:
    M: int = 8
    sigma: float = 0.2   # chosen on seed-7 data (d=7: 147 -> 119 failures at M=8); 0.3+ is worse
    correlated: bool = True
    mix_growth: bool = True
    seed: int = 1000
    scale: int = 4
    parallel: str = "shots"
    backend: str = "numba"

    def __post_init__(self) -> None:
        if self.M < 1:
            raise ValueError(f"M must be >= 1, got {self.M}")
        if self.sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {self.sigma}")
        if self.parallel not in PARALLEL_MODES:
            raise ValueError(f"parallel must be one of {PARALLEL_MODES}, got {self.parallel!r}")
        if self.backend not in ("numba", "reference"):
            raise ValueError(f"unknown backend {self.backend!r}")


@dataclass(eq=False)
class EnsembleResult:
    predictions: np.ndarray        # (shots, n_obs) bool
    ok: np.ndarray                 # (shots,) bool
    chosen: np.ndarray             # (shots,) int32 winning variant
    correction: np.ndarray | None  # (shots, E) uint8 if requested


def variant_lengths(g: DetectorGraph3D, M: int, sigma: float, seed: int, scale: int) -> np.ndarray:
    w = g.edge_weight
    w_min = float(w.min())
    out = np.empty((M, g.n_edges), np.int32)
    out[0] = length_of(w, w_min, scale)
    for v in range(1, M):
        z = np.random.default_rng(seed + v).standard_normal(g.n_edges)
        out[v] = length_of(w * np.exp(sigma * z), w_min, scale)
    return out


# ---- numba -------------------------------------------------------------------------------
@njit(cache=True)
def _multi_work(M, n, E):
    parent = np.empty((M, n + 1), np.int32)
    for v in range(M):
        for i in range(n + 1):
            parent[v, i] = i
    return (parent, np.zeros((M, n + 1), np.int32), np.zeros((M, n + 1), np.uint8),
            np.zeros((M, n + 1), np.bool_), np.ones((M, n + 1), np.int32), np.zeros((M, n + 1), np.bool_),
            np.empty((M, n), np.int32), np.zeros((M, E), np.int32), np.zeros((M, E), np.bool_),
            np.empty((M, E), np.int32), np.zeros((M, n + 1), np.bool_), np.empty((M, E), np.int32),
            np.zeros((M, n + 1), np.bool_), np.empty((M, n + 1), np.int32), np.empty((M, n + 1), np.int32),
            np.empty((M, n), np.int32), np.empty((M, n + 1), np.int32), np.zeros((M, n + 1), np.bool_),
            np.zeros((M, 1), np.int32), np.empty((M, n + 1), np.int32), np.zeros((M, 1), np.int32),
            np.zeros((M, E), np.int32), np.empty((M, E), np.int32), np.zeros((M, n + 1), np.int32),
            np.empty((M, E), np.int32), np.zeros((M, n), np.int32), np.empty((M, n), np.int32),
            np.empty((M, n + 1), np.int32), np.empty((M, E), np.int64), np.zeros((M, E), np.bool_))


@njit(cache=True)
def _view(W, v):
    return (W[0][v], W[1][v], W[2][v], W[3][v], W[4][v], W[5][v], W[6][v], W[7][v], W[8][v], W[9][v],
            W[10][v], W[11][v], W[12][v], W[13][v], W[14][v], W[15][v], W[16][v], W[17][v], W[18][v],
            W[19][v], W[20][v], W[21][v], W[22][v], W[23][v], W[24][v], W[25][v], W[26][v],
            W[27][v], W[28][v], W[29][v])


@njit(cache=True)
def _variant(lits, nlit, v, edge_u, edge_v, lengths, smallest, inc_edge, b_inc, obs_bits, correlated,
             cptr, cidx, clen, ccredit, W, len2, mod, inside, keep, stat_keep, none):
    """Decode the events lits[:nlit] with variant v; returns (bits, ok, score under lengths[0])."""
    work = _view(W, v)
    if correlated:
        b, ok = _uf_correlated_shot(lits, nlit, 0, edge_u, edge_v, lengths[v], inc_edge, b_inc, obs_bits,
                                    none, smallest[v], work, cptr, cidx, clen, ccredit, len2[v], mod[v],
                                    inside[v], keep[v], stat_keep[v], False, False, lengths[v], ccredit, 2,
                                    True)
    else:
        b, ok = _uf_shot(lits, nlit, 0, edge_u, edge_v, lengths[v], inc_edge, b_inc, obs_bits, none,
                         smallest[v], work)
    flips = work[19]
    nf = work[20][0]
    if correlated:
        sc = _corr_score(flips, nf, lengths[0], cptr, cidx, ccredit, inside[v])
    else:
        sc = 0
        for i in range(nf):
            sc += lengths[0][flips[i]]
    return b, ok, sc


@njit(cache=True)
def _pick(M, vb, vok, vsc):
    best = 0
    for v in range(1, M):
        if vsc[v] < vsc[best]:
            best = v
    return best


@njit(cache=True)
def _scratch(M, n, E, lengths):
    return (lengths.copy(), np.empty((M, E), np.int32), np.zeros((M, E), np.bool_),
            np.empty((M, n + 1), np.int32), np.zeros((M, 1), np.int32))


@njit(parallel=True, cache=True)
def _variants_of_shot(lits, nlit, M, edge_u, edge_v, lengths, smallest, inc_edge, b_inc, obs_bits, correlated,
                      cptr, cidx, clen, ccredit, W, len2, mod, inside, keep, stat_keep, none, vb, vok, vsc):
    for v in prange(M):
        b, ok, sc = _variant(lits, nlit, v, edge_u, edge_v, lengths, smallest, inc_edge, b_inc, obs_bits,
                             correlated, cptr, cidx, clen, ccredit, W, len2, mod, inside, keep,
                             stat_keep, none)
        vb[v] = b
        vok[v] = ok
        vsc[v] = sc


@njit(cache=True)
def _ens_by_variants(dets, edge_u, edge_v, lengths, smallest, inc_edge, b_inc, obs_bits, correlated,
                     cptr, cidx, clen, ccredit, out_bits, out_ok, out_v, corr):
    S, n = dets.shape
    M, E = lengths.shape
    W = _multi_work(M, n, E)
    len2, mod, inside, keep, stat_keep = _scratch(M, n, E, lengths)
    none = np.zeros((0, 0), np.uint8)
    vb = np.zeros(M, np.uint64)
    vok = np.zeros(M, np.bool_)
    vsc = np.zeros(M, np.int64)
    record = corr.shape[0] > 0
    lits = np.empty(n, np.int32)
    for s in range(S):
        nlit = extract_lits(dets[s], lits)
        _variants_of_shot(lits, nlit, M, edge_u, edge_v, lengths, smallest, inc_edge, b_inc, obs_bits,
                          correlated, cptr, cidx, clen, ccredit, W, len2, mod, inside, keep, stat_keep,
                          none, vb, vok, vsc)
        best = _pick(M, vb, vok, vsc)
        out_bits[s] = vb[best]
        out_ok[s] = vok[best]
        out_v[s] = best
        if record:
            for i in range(W[20][best][0]):
                corr[s, W[19][best][i]] = 1


@njit(parallel=True, cache=True)
def _ens_by_shots(dets, edge_u, edge_v, lengths, smallest, inc_edge, b_inc, obs_bits, correlated,
                  cptr, cidx, clen, ccredit, out_bits, out_ok, out_v, corr, chunk):
    S, n = dets.shape
    M, E = lengths.shape
    record = corr.shape[0] > 0
    nchunk = (S + chunk - 1) // chunk
    for ci in prange(nchunk):
        W = _multi_work(M, n, E)
        len2, mod, inside, keep, stat_keep = _scratch(M, n, E, lengths)
        none = np.zeros((0, 0), np.uint8)
        vb = np.zeros(M, np.uint64)
        vok = np.zeros(M, np.bool_)
        vsc = np.zeros(M, np.int64)
        lits = np.empty(n, np.int32)
        for s in range(ci * chunk, min(S, (ci + 1) * chunk)):
            nlit = extract_lits(dets[s], lits)
            for v in range(M):
                b, ok, sc = _variant(lits, nlit, v, edge_u, edge_v, lengths, smallest, inc_edge, b_inc,
                                     obs_bits, correlated, cptr, cidx, clen, ccredit, W, len2, mod,
                                     inside, keep, stat_keep, none)
                vb[v] = b
                vok[v] = ok
                vsc[v] = sc
            best = _pick(M, vb, vok, vsc)
            out_bits[s] = vb[best]
            out_ok[s] = vok[best]
            out_v[s] = best
            if record:
                for i in range(W[20][best][0]):
                    corr[s, W[19][best][i]] = 1


# ---- engine ------------------------------------------------------------------------------
class UFEnsemble:
    def __init__(self, graph: DetectorGraph3D, config: EnsembleConfig | None = None):
        self.graph = g = graph
        self.config = c = config or EnsembleConfig()
        self.tables = build_uf_tables(g, c.scale)
        self.lengths = variant_lengths(g, c.M, c.sigma, c.seed, c.scale)
        self.smallest = np.array([c.mix_growth and v % 2 == 1 for v in range(c.M)], dtype=np.bool_)

    def run(self, detection_events: np.ndarray, record_correction: bool = False,
            backend: str | None = None, parallel: str | None = None) -> EnsembleResult:
        g, t, c = self.graph, self.tables, self.config
        dets = np.ascontiguousarray(detection_events, dtype=np.uint8)
        if dets.ndim != 2 or dets.shape[1] != g.n_det:
            raise ValueError(f"expected (shots, {g.n_det}), got {dets.shape}")
        S = dets.shape[0]
        bits = np.zeros(S, np.uint64)
        ok = np.ones(S, bool)
        chosen = np.zeros(S, np.int32)
        corr = np.zeros((S, g.n_edges) if record_correction else (0, 0), np.uint8)
        backend = backend or c.backend
        if backend == "numba":
            args = (dets, g.edge_u, g.edge_v, self.lengths, self.smallest, g.inc_edge, t.b_inc,
                    g.edge_obs_bits, c.correlated, t.corr_ptr, t.corr_idx, t.corr_len, t.corr_credit,
                    bits, ok, chosen, corr)
            if (parallel or c.parallel) == "variants":
                _ens_by_variants(*args)
            else:
                _ens_by_shots(*args, 16)
        else:
            for s in range(S):
                bits[s], ok[s], chosen[s], flipped = self._shot_reference(dets[s])
                if record_correction:
                    corr[s, flipped] = 1
        return EnsembleResult(bits_to_bool(bits, g.n_obs), ok, chosen, corr if record_correction else None)

    def _shot_reference(self, row):
        g, t, c = self.graph, self.tables, self.config
        best = None
        for v in range(c.M):
            tv = replace(t, length=self.lengths[v], smallest=bool(self.smallest[v]),
                         score_len=self.lengths[v].astype(np.int64))
            fn = uf_correlated_shot_reference if c.correlated else uf_decode_shot_reference
            b, flipped, ok = fn(g, tv, row)
            score = correlated_score(t, flipped) if c.correlated else int(sum(int(t.length[e]) for e in flipped))
            if best is None or score < best[0]:
                best = (score, b, ok, v, flipped)
        return best[1], best[2], best[3], best[4]


class UFEnsembleDecoder(Decoder):
    name = "uf3d_ensemble"
    single_fault_allowance = 2  # inherits correlated UF's two circuit d=3 cases (see union_find)

    def __init__(self, dem: stim.DetectorErrorModel, config: EnsembleConfig | None = None):
        super().__init__(dem)
        self.engine = UFEnsemble(build_detector_graph(dem), config)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        return self.engine.run(self._check_input(detection_events)).predictions
