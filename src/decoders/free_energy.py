"""Free energy of a matching correction under the full hyperedge fault model.

A correction is a set of graph edges whose boundary is the syndrome. Its
energy is the negative log-likelihood ratio of the most probable fault set that
produces it. Each edge costs the weight of its independent graphlike
mechanisms, and a decomposed hyperedge fault (for example a Y error, which is
one X edge and one Z edge) can explain several of its edges at once for less.

Matching finds one low-energy correction. The maximum-likelihood class compares
sums over every correction in a class. The short cycles of the matching graph
(triangles and squares, the local stabilizer moves) generate the corrections
near a given one. Toggling cycle m changes the energy by dE_m. A first-order
cluster expansion of the class sum around a locally minimal correction S gives

    -log Z(S) ~ E(S) - sum_m [log(1 + exp(-dE_m(S))) - log(1 + exp(-E(m)))],

where m runs over cycles touching S. The subtracted term removes the vacuum
fluctuations that every correction shares. Each correction is first pushed
downhill by single cycle moves, so every dE_m in the sum is non-negative.

The cycle and hyperedge tables are built once per detector error model.
Scoring a correction touches only the cycles adjacent to it, so the cost grows
with the size of the correction rather than with the code distance.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import stim
from numba import njit

MISSING = 40.0  # weight of an edge that only occurs inside hyperedges


def _merge(a, b):
    return a + b - 2 * a * b


def _csr(lists):
    ptr = np.zeros(len(lists) + 1, np.int64)
    for i, items in enumerate(lists):
        ptr[i + 1] = ptr[i] + len(items)
    flat = np.fromiter((x for items in lists for x in items), np.int64, count=int(ptr[-1]))
    return ptr, flat


class CorrectionScorer:
    """Static tables for scoring corrections on one matching graph.

    Detector indices follow the given DEM. Single-detector parts attach to one
    extra node, index num_detectors, which stands for the free boundary.
    """

    def __init__(self, dem: stim.DetectorErrorModel, descent_steps: int = 8, cycle_length: int = 4):
        if cycle_length not in (3, 4):
            raise ValueError("cycle_length must be 3 or 4")
        self.descent_steps = descent_steps
        free = dem.num_detectors
        self.num_nodes = free + 1
        key_of = {}
        single = defaultdict(float)
        hyper = defaultdict(float)
        for ins in dem.flattened():
            if ins.type != "error":
                continue
            p = ins.args_copy()[0]
            if not 0 < p < 0.5:
                continue
            ids = set()
            for part in ins.target_groups():
                sites = sorted(t.val for t in part if t.is_relative_detector_id())
                if not sites:
                    continue
                if len(sites) > 2:
                    raise ValueError("expected a graphlike decomposition (decompose_errors=True)")
                key = (sites[0], sites[1] if len(sites) == 2 else free)
                ids ^= {key_of.setdefault(key, len(key_of))}
            ids = tuple(sorted(ids))
            if len(ids) == 1:
                single[ids[0]] = _merge(single[ids[0]], p)
            elif ids:
                hyper[ids] = _merge(hyper[ids], p)
        num_edges = len(key_of)
        self.num_edges = num_edges
        packed = {u * self.num_nodes + v: e for (u, v), e in key_of.items()}
        self.keys = np.array(sorted(packed), np.int64)
        self.key_ids = np.array([packed[k] for k in self.keys], np.int64)
        self.w = np.full(num_edges, MISSING)
        for e, p in single.items():
            self.w[e] = math.log((1 - p) / p)

        hlist = sorted(hyper)
        self.hw = np.array([math.log((1 - hyper[h]) / hyper[h]) for h in hlist], np.float64)
        self.hptr, self.hedges = _csr(hlist)
        by_hyper = [[] for _ in range(num_edges)]
        for i, h in enumerate(hlist):
            for e in h:
                by_hyper[e].append(i)
        self.ehptr, self.eh = _csr(by_hyper)

        adj = defaultdict(dict)
        for (u, v), e in key_of.items():
            adj[u][v] = e
            adj[v][u] = e
        cycles = set()
        for (a, b), e in key_of.items():
            for x, e2 in adj[b].items():
                if x == a:
                    continue
                if x in adj[a]:
                    cycles.add(frozenset((e, e2, adj[a][x])))
                for y, e3 in (adj[x].items() if cycle_length == 4 else ()):
                    if y != a and y != b and y in adj[a]:
                        c = frozenset((e, e2, e3, adj[a][y]))
                        if len(c) == 4:
                            cycles.add(c)
        clist = sorted(tuple(sorted(c)) for c in cycles)
        self.cptr, self.cedges = _csr(clist)
        by_edge = [[] for _ in range(num_edges)]
        for i, c in enumerate(clist):
            for e in c:
                by_edge[e].append(i)
        self.ecptr, self.ec = _csr(by_edge)
        # For each cycle, the hyperedges meeting it, which cycle positions they
        # cover (a bit mask over at most four edges), and an upper bound on the
        # hyperedge savings a toggle of the cycle can gain.
        saving = np.array([max(0.0, sum(self.w[e] for e in h) - self.hw[i])
                           for i, h in enumerate(hlist)])
        meet, masks, bound = [], [], np.zeros(len(clist))
        for c in clist:
            found = defaultdict(int)
            for pos, e in enumerate(c):
                for h in by_hyper[e]:
                    found[h] |= 1 << pos
            meet.append(sorted(found))
            masks.append([found[h] for h in sorted(found)])
            bound[len(meet) - 1] = sum(saving[h] for h in found)
        self.mptr, self.mh = _csr(meet)
        self.mmask = np.fromiter((m for ms in masks for m in ms), np.int64, count=len(self.mh))
        self.msize = np.diff(self.hptr)[self.mh] if len(self.mh) else np.zeros(0, np.int64)
        self.bound = bound
        self.state = _State(num_edges, len(hlist), len(clist))
        self.vac = _vacuum(*self._graph(np.zeros(len(clist))), *self.state.arrays)

    def _graph(self, vac):
        return (self.keys, self.key_ids, self.num_nodes, self.w, self.hptr, self.hedges, self.hw,
                self.ehptr, self.eh, self.cptr, self.cedges, self.ecptr, self.ec, vac,
                self.mptr, self.mh, self.mmask, self.msize, self.bound)

    def score(self, pairs: np.ndarray, offsets: np.ndarray):
        """Energy, entropy and an identity hash for each correction.

        Correction i is the edge list pairs[offsets[i]:offsets[i+1]] in the
        node-pair format returned by pymatching (-1 for the boundary). A
        correction that uses an edge missing from the model gets energy inf.
        """
        return _score_many(np.ascontiguousarray(pairs, np.int64), np.asarray(offsets, np.int64),
                           *self._graph(self.vac), *self.state.arrays, self.descent_steps)


class _State:
    """Per-scorer scratch arrays, clean between calls."""

    def __init__(self, num_edges, num_hyper, num_cycles):
        self.arrays = (
            np.zeros(num_edges, np.bool_),      # in_s
            np.zeros(num_edges, np.bool_),      # claimed
            np.zeros(num_hyper, np.int32),      # count of hyperedge edges in S
            np.full(num_hyper, -1, np.int64),   # hyperedge stamps
            np.full(num_cycles, -1, np.int64),  # cycle stamps
            np.zeros(1, np.int64),              # stamp counter
        )


@njit(cache=True, inline="always")
def _packing(cand, ncand, hptr, hedges, hw, w, claimed):
    """Greedy disjoint hyperedge savings among cand[:ncand], reordering cand."""
    total = 0.0
    live = ncand
    while live > 0:
        best, best_i = 0.0, -1
        for i in range(live):
            h = cand[i]
            s = -hw[h]
            for j in range(hptr[h], hptr[h + 1]):
                e = hedges[j]
                if claimed[e]:
                    s = -1.0
                    break
                s += w[e]
            if s > best:
                best, best_i = s, i
        if best_i < 0:
            break
        h = cand[best_i]
        for j in range(hptr[h], hptr[h + 1]):
            claimed[hedges[j]] = True
        total += best
        live -= 1
        cand[best_i], cand[live] = cand[live], cand[best_i]
    for k in range(ncand):
        h = cand[k]
        for j in range(hptr[h], hptr[h + 1]):
            claimed[hedges[j]] = False
    return total


SKIP = 12.0  # moves whose energy change provably exceeds this are ignored

POPCOUNT = np.array([bin(i).count("1") for i in range(16)], np.int64)


@njit(cache=True, inline="always")
def _move(m, cptr, cedges, mptr, mh, mmask, msize, bound, hptr, hedges, hw, w, in_s, claimed,
          count, before, after):
    """Energy change from toggling cycle m in S, leaving S unchanged.

    Only hyperedges meeting m can change status. Packing is recomputed among
    those, treating hyperedges elsewhere in S as fixed. Returns a lower bound
    above SKIP when the move cannot matter.
    """
    base = 0.0
    held = 0
    pos = 0
    for j in range(cptr[m], cptr[m + 1]):
        e = cedges[j]
        if in_s[e]:
            base -= w[e]
            held |= 1 << pos
        else:
            base += w[e]
        pos += 1
    if base - bound[m] > SKIP:
        return base - bound[m]
    nb = na = 0
    for k in range(mptr[m], mptr[m + 1]):
        h = mh[k]
        size = msize[k]
        c = count[h]
        mask = mmask[k]
        moved = c - POPCOUNT[mask & held] + POPCOUNT[mask & ~held & 15]
        if c == size and nb < len(before):
            before[nb] = h
            nb += 1
        if moved == size and na < len(after):
            after[na] = h
            na += 1
    if nb == 0 and na == 0:
        return base
    return base - (_packing(after, na, hptr, hedges, hw, w, claimed)
                   - _packing(before, nb, hptr, hedges, hw, w, claimed))


@njit(cache=True, inline="always")
def _toggle(e, add, in_s, count, ehptr, eh):
    in_s[e] = add
    step = 1 if add else -1
    for k in range(ehptr[e], ehptr[e + 1]):
        count[eh[k]] += step


@njit(cache=True)
def _vacuum(keys, key_ids, num_nodes, w, hptr, hedges, hw, ehptr, eh, cptr, cedges, ecptr, ec,
            vac, mptr, mh, mmask, msize, bound, in_s, claimed, count, hstamp, cstamp, clock):
    """log(1 + exp(-E(m))) for every short cycle m in isolation."""
    before = np.empty(64, np.int64)
    after = np.empty(64, np.int64)
    out = np.empty(len(cptr) - 1)
    for m in range(len(cptr) - 1):
        energy = _move(m, cptr, cedges, mptr, mh, mmask, msize, bound, hptr, hedges, hw, w, in_s,
                       claimed, count, before, after)
        out[m] = math.log1p(math.exp(-energy))
    return out


@njit(cache=True)
def _score_many(pairs, offsets, keys, key_ids, num_nodes, w, hptr, hedges, hw, ehptr, eh, cptr,
                cedges, ecptr, ec, vac, mptr, mh, mmask, msize, bound, in_s, claimed, count, hstamp,
                cstamp, clock, max_steps):
    n = len(offsets) - 1
    energy = np.empty(n)
    entropy = np.zeros(n)
    ident = np.zeros(n, np.uint64)
    free = num_nodes - 1
    touched = np.empty(256, np.int64)
    before = np.empty(64, np.int64)
    after = np.empty(64, np.int64)
    edges = np.empty(16, np.int64)
    for c in range(n):
        lo, hi = offsets[c], offsets[c + 1]
        need = (hi - lo) + 4 * max_steps + 8
        if len(edges) < need:
            edges = np.empty(2 * need, np.int64)
        ne = 0
        ok = True
        for i in range(lo, hi):
            u, v = pairs[i, 0], pairs[i, 1]
            u = free if u < 0 else u
            v = free if v < 0 else v
            if u > v:
                u, v = v, u
            key = u * num_nodes + v
            pos = np.searchsorted(keys, key)
            if pos >= len(keys) or keys[pos] != key:
                ok = False
                break
            e = key_ids[pos]
            _toggle(e, not in_s[e], in_s, count, ehptr, eh)
            edges[ne] = e
            ne += 1
        # Keep each edge that ended up in S once.
        k = 0
        for j in range(ne):
            e = edges[j]
            if in_s[e]:
                dup = False
                for q in range(k):
                    if edges[q] == e:
                        dup = True
                        break
                if not dup:
                    edges[k] = e
                    k += 1
        ne = k
        if ok:
            # Each sweep evaluates every adjacent cycle move. A sweep that finds
            # no downhill move is final, and its move energies give the entropy.
            s = 0.0
            for step in range(max_steps + 1):
                clock[0] += 1
                sweep = clock[0]
                best, best_m = -1e-9, -1
                s = 0.0
                for j in range(ne):
                    e = edges[j]
                    for q in range(ecptr[e], ecptr[e + 1]):
                        m = ec[q]
                        if cstamp[m] == sweep:
                            continue
                        cstamp[m] = sweep
                        d = _move(m, cptr, cedges, mptr, mh, mmask, msize, bound, hptr, hedges, hw, w,
                                  in_s, claimed, count, before, after)
                        if d < best:
                            best, best_m = d, m
                        s += math.log1p(math.exp(-max(d, 0.0))) - vac[m]
                if best_m < 0 or step == max_steps:
                    break
                for q in range(cptr[best_m], cptr[best_m + 1]):
                    e = cedges[q]
                    _toggle(e, not in_s[e], in_s, count, ehptr, eh)
                    if in_s[e]:
                        edges[ne] = e
                        ne += 1
                k = 0
                for j in range(ne):
                    if in_s[edges[j]]:
                        edges[k] = edges[j]
                        k += 1
                ne = k
            entropy[c] = s
            # Energy of the locally minimal correction.
            clock[0] += 1
            stamp = clock[0]
            nc = 0
            base = 0.0
            for j in range(ne):
                e = edges[j]
                base += w[e]
                for q in range(ehptr[e], ehptr[e + 1]):
                    h = eh[q]
                    if hstamp[h] != stamp:
                        hstamp[h] = stamp
                        if count[h] == hptr[h + 1] - hptr[h]:
                            if nc == len(touched):
                                touched = np.concatenate((touched, touched))
                            touched[nc] = h
                            nc += 1
            energy[c] = base - _packing(touched, nc, hptr, hedges, hw, w, claimed)
            h64 = np.uint64(0)
            for j in range(ne):
                x = np.uint64(edges[j] + 1) * np.uint64(0x9E3779B97F4A7C15)
                h64 += x ^ (x >> np.uint64(31))
            ident[c] = h64
        else:
            energy[c] = np.inf
        for j in range(ne):
            if in_s[edges[j]]:
                _toggle(edges[j], False, in_s, count, ehptr, eh)
    return energy, entropy, ident
