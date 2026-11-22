"""Prepare detector error models for matching.

Two transformations are used by the free-energy decoder.

* regroup: Stim's graphlike decomposition sometimes splits one fault into
  several components inside the same detector sector (the detectors linked by
  graphlike edges, for example the X checks of a surface code). When their
  XOR is itself graphlike it is one ordinary edge, so the parts are merged and
  only genuine cross-sector correlations remain hyperedges. The odds of each
  fault can also be multiplied by odds_scale per component. That changes the
  weights matching sees, never the circuit that is sampled.

* LogicalSplit: for one logical observable, replace the free boundary of the
  sector that carries the observable by two explicit nodes. Node A collects
  boundary edges that flip the observable and node B the rest. Every correction
  then flips the observable exactly when it uses an odd number of A edges, so
  setting the detection values of A and B forces matching into a chosen
  logical class. If the observable also sits on bulk edges, a gauge set G of
  detectors is folded into it first: L' = L xor parity(detectors in G), chosen
  by two-colouring the bulk graph so that every bulk edge has L' = 0. This
  works whenever every cycle of bulk edges is logically trivial, which holds
  for planar codes.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import stim


def _sectors(dem: stim.DetectorErrorModel) -> np.ndarray:
    parent = list(range(dem.num_detectors))

    def root(a):
        while a != parent[a]:
            parent[a] = a = parent[parent[a]]
        return a

    for ins in dem.flattened():
        if ins.type == "error":
            for part in ins.target_groups():
                sites = [t.val for t in part if t.is_relative_detector_id()]
                if len(sites) == 2:
                    parent[root(sites[1])] = root(sites[0])
    return np.array([root(i) for i in range(dem.num_detectors)], dtype=np.int64)


def _parts(ins):
    """(detector set, logical mask) of each separator-delimited part."""
    out = []
    for part in ins.target_groups():
        sites, logical = set(), 0
        for t in part:
            if t.is_relative_detector_id():
                sites ^= {t.val}
            elif t.is_logical_observable_id():
                logical ^= 1 << t.val
        out.append((sites, logical))
    return out


def _targets(parts):
    targets = []
    for sites, logical in parts:
        if targets:
            targets.append(stim.target_separator())
        targets.extend(stim.target_relative_detector_id(s) for s in sorted(sites))
        targets.extend(stim.target_logical_observable_id(b) for b in range(64) if logical >> b & 1)
    return targets


def center(dem: stim.DetectorErrorModel):
    """Replace each fault with p > 1/2 by its complement.

    Such a fault is more likely present than absent, so its presence becomes
    the reference: the returned model has probability 1 - p, and the caller
    XORs detector_offset into shots and logical_offset into predictions.
    Returns (model, detector_offset, logical_offset).
    """
    offsets = np.zeros(dem.num_detectors, dtype=bool)
    logical = np.zeros(dem.num_observables, dtype=bool)
    out = stim.DetectorErrorModel()
    for ins in dem.flattened():
        if ins.type != "error" or ins.args_copy()[0] <= 0.5:
            out.append(ins)
            continue
        for t in ins.targets_copy():
            if t.is_relative_detector_id():
                offsets[t.val] ^= True
            elif t.is_logical_observable_id():
                logical[t.val] ^= True
        if ins.args_copy()[0] < 1:
            out.append("error", 1 - ins.args_copy()[0], ins.targets_copy())
    if out.num_detectors < dem.num_detectors:
        out.append("detector", [], [stim.target_relative_detector_id(dem.num_detectors - 1)])
    if out.num_observables < dem.num_observables:
        out.append("logical_observable", [], [stim.target_logical_observable_id(dem.num_observables - 1)])
    return out, offsets, logical


def regroup(dem: stim.DetectorErrorModel, odds_scale: float = 1.0) -> stim.DetectorErrorModel:
    """Merge same-sector components of each fault and rescale fault odds.

    At odds_scale=1 the joint law of detectors and observables is unchanged.
    Faults with probability above 1/2 are rejected; callers check for them.
    """
    labels = _sectors(dem)
    out = stim.DetectorErrorModel()
    for ins in dem.flattened():
        if ins.type != "error":
            out.append(ins)
            continue
        p = ins.args_copy()[0]
        if p > 0.5:
            raise ValueError("faults with probability above 1/2 are not supported")
        parts = _parts(ins)
        by_sector = defaultdict(lambda: [set(), 0])
        frame = 0
        for sites, logical in parts:
            if not sites:
                frame ^= logical
                continue
            slot = by_sector[labels[min(sites)]]
            slot[0] ^= sites
            slot[1] ^= logical
        merged = [(s, l) for s, l in by_sector.values() if s or l]
        if merged and all(len(s) <= 2 and s for s, _ in merged):
            parts = merged
        else:
            parts = [(s, l) for s, l in parts if s]
        if not parts:
            if frame:
                raise ValueError("a fault flips an observable without flipping any detector")
            continue
        parts[0] = (parts[0][0], parts[0][1] ^ frame)
        if p == 0:
            continue
        if odds_scale != 1 and p < 0.5:
            odds = p / (1 - p) * odds_scale ** len(parts)
            p = odds / (1 + odds)
        out.append("error", p, _targets(parts))
    # Keep the detector and observable counts even if trailing ones lost all faults.
    if out.num_detectors < dem.num_detectors:
        out.append("detector", [], [stim.target_relative_detector_id(dem.num_detectors - 1)])
    if out.num_observables < dem.num_observables:
        out.append("logical_observable", [], [stim.target_logical_observable_id(dem.num_observables - 1)])
    return out


def restrict(dem: stim.DetectorErrorModel, keep: np.ndarray) -> stim.DetectorErrorModel:
    """Drop every graphlike part with a detector outside keep.

    Plain matching factorizes over detector sectors, so on syndromes supported
    inside keep this graph gives the same matching as the full one.
    """
    out = stim.DetectorErrorModel()
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        parts = [(s, l) for s, l in _parts(ins) if s and all(keep[x] for x in s)]
        if parts and ins.args_copy()[0] > 0:
            out.append("error", ins.args_copy()[0], _targets(parts))
    out.append("detector", [], [stim.target_relative_detector_id(dem.num_detectors - 1)])
    if dem.num_observables:
        out.append("logical_observable", [], [stim.target_logical_observable_id(dem.num_observables - 1)])
    return out


class LogicalSplit:
    """Class-forcing transform of a one-observable, graphlike DEM.

    Attributes:
        dem: the split model with detectors A = n and B = n + 1 appended.
        sector: boolean mask of the detectors in the logical sector.
        gauge: boolean mask G; the observable equals L' xor parity(G).
    """

    def __init__(self, dem: stim.DetectorErrorModel, sector: np.ndarray, gauge: np.ndarray,
                 split: stim.DetectorErrorModel):
        self.source = dem
        self.sector = sector
        self.gauge = gauge
        self.dem = split
        self.num_detectors = dem.num_detectors

    @classmethod
    def build(cls, dem: stim.DetectorErrorModel) -> "LogicalSplit | None":
        """Return None when the model has no such split (several observables,
        several sectors carrying the observable, or a logically odd bulk cycle)."""
        if dem.num_observables != 1:
            return None
        n = dem.num_detectors
        labels = _sectors(dem)
        bulk = defaultdict(list)
        carriers = set()
        for ins in dem.flattened():
            if ins.type != "error":
                continue
            for sites, logical in _parts(ins):
                if len(sites) > 2:
                    return None
                if logical and not sites:
                    return None
                if logical:
                    carriers.add(labels[min(sites)])
                if len(sites) == 2:
                    u, v = sorted(sites)
                    bulk[u].append((v, logical))
                    bulk[v].append((u, logical))
        if len(carriers) != 1:
            return None
        home, = carriers
        sector = labels == home
        # Two-colour the logical sector so bulk edges carry no observable.
        colour = np.full(n, -1, dtype=np.int64)
        start = int(np.flatnonzero(sector)[0])
        colour[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v, logical in bulk[u]:
                want = colour[u] ^ logical
                if colour[v] < 0:
                    colour[v] = want
                    queue.append(v)
                elif colour[v] != want:
                    return None
        gauge = colour == 1
        a = stim.target_relative_detector_id(n)
        b = stim.target_relative_detector_id(n + 1)
        out = stim.DetectorErrorModel()
        for ins in dem.flattened():
            if ins.type != "error":
                out.append(ins)
                continue
            targets = []
            for sites, logical in _parts(ins):
                if targets:
                    targets.append(stim.target_separator())
                targets.extend(stim.target_relative_detector_id(s) for s in sorted(sites))
                if logical:
                    targets.append(stim.target_logical_observable_id(0))
                if len(sites) == 1 and sector[min(sites)]:
                    flipped = logical ^ int(gauge[min(sites)])
                    targets.append(a if flipped else b)
            out.append("error", ins.args_copy()[0], targets)
        return cls(dem, sector, gauge, out)

    def shift(self, dets: np.ndarray) -> np.ndarray:
        """parity(G) per shot: the observable equals L' xor this."""
        return np.bitwise_xor.reduce(dets[:, self.gauge], axis=1) if self.gauge.any() else \
            np.zeros(len(dets), dtype=bool)

    def force(self, dets: np.ndarray, frame_class: np.ndarray) -> np.ndarray:
        """Append A and B so that matching must return L' = frame_class."""
        rows = np.zeros((len(dets), self.num_detectors + 2), dtype=bool)
        rows[:, :self.num_detectors] = dets
        parity = np.bitwise_xor.reduce(dets[:, self.sector], axis=1)
        rows[:, -2] = frame_class
        rows[:, -1] = parity ^ frame_class
        return rows
