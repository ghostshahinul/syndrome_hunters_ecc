"""Read both interpretations of a decomposed Stim detector error model.

An error instruction is one Bernoulli variable. Its separator-delimited parts
give graph edges for matching, while their XOR gives the complete circuit fault.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import NamedTuple

import numpy as np
import stim


class Fault(NamedTuple):
    detectors: tuple[int, ...]
    logical: int
    probability: float

    @property
    def weight(self) -> float:
        return math.log1p(-self.probability) - math.log(self.probability)


@dataclass
class FaultModel:
    faults: list[Fault]
    edges: list[Fault]
    detector_offset: np.ndarray
    logical_offset: int
    couplings: set
    probabilities: list[float]
    detector_events: list[set[int]]
    edge_events: dict


def _toggle(table, key, probability):
    # Parity of independent Bernoulli mechanisms with identical signatures.
    old = table.get(key, 0.0)
    table[key] = old + probability - 2 * old * probability


def read_model(dem: stim.DetectorErrorModel) -> FaultModel:
    if dem.num_observables > 64:
        raise ValueError("the handwritten decoders support up to 64 logical observables")
    whole, edges, couplings = {}, {}, set()
    probabilities = []
    detector_events = [set() for _ in range(dem.num_detectors)]
    edge_events = defaultdict(set)
    offset = np.zeros(dem.num_detectors, dtype=bool)
    logical_offset = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        parts = []
        sites, logical = set(), 0
        for target in inst.targets_copy():
            if target.is_separator():
                parts.append((tuple(sorted(sites)), logical))
                sites, logical = set(), 0
            elif target.is_relative_detector_id():
                sites.symmetric_difference_update((target.val,))
            elif target.is_logical_observable_id():
                logical ^= 1 << target.val
        parts.append((tuple(sorted(sites)), logical))

        total, frame = set(), 0
        odd_parts = set()
        for part, mask in parts:
            total.symmetric_difference_update(part)
            frame ^= mask
            odd_parts.symmetric_difference_update(((part, mask),))
        if p > 0.5:
            # Use the most likely occurrence as a known change of reference.
            for i in total:
                offset[i] ^= True
            logical_offset ^= frame
            p = 1 - p
        if p == 0:
            continue
        if total or frame:
            _toggle(whole, (tuple(sorted(total)), frame), p)
        event = len(probabilities)
        probabilities.append(p)
        for d in total:
            detector_events[d].add(event)
        couplings.update(combinations(sorted(part for part in odd_parts if part[0]), 2))
        for part, mask in odd_parts:
            if len(part) > 2:
                raise ValueError("expected a graphlike decomposition (decompose_errors=True)")
            if part:
                _toggle(edges, (part, mask), p)
                edge_events[part, mask].add(event)

    def unpack(table):
        return [Fault(sites, mask, p) for (sites, mask), p in sorted(table.items())]

    return FaultModel(unpack(whole), unpack(edges), offset, logical_offset, couplings,
                      probabilities, detector_events, dict(edge_events))


def edge_arrays(model: FaultModel, boundary: int):
    """Parallel edges use the cheapest representative, retaining its logical mask."""
    best = {}
    for edge in model.edges:
        u = edge.detectors[0]
        v = edge.detectors[1] if len(edge.detectors) == 2 else boundary
        candidate = (edge.weight, edge.logical)
        if (u, v) not in best or candidate < best[u, v]:
            best[u, v] = candidate
    ordered = sorted(best, key=lambda ends: (best[ends][0], ends))
    return (
        np.array([u for u, _ in ordered], dtype=np.int32),
        np.array([v for _, v in ordered], dtype=np.int32),
        np.array([best[e][0] for e in ordered], dtype=np.float64),
        np.array([best[e][1] for e in ordered], dtype=np.uint64),
    )


def logical_bits(masks: np.ndarray, count: int) -> np.ndarray:
    shifts = np.arange(count, dtype=np.uint64)
    return ((masks[:, None] >> shifts) & 1).astype(bool)
