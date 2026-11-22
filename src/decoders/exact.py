"""Exact logical MAP lookup for small DEM syndrome spaces.

For a single observable the inverse Walsh transform of E[(-1)^(z·D+L)]
gives P(D=s,L=0)-P(D=s,L=1), up to the common transform length. For
multiple observables the transform of the joint characteristic function
gives all their joint probabilities. Independent Bernoulli faults multiply
in Fourier space. One extra Walsh transform evaluates their log product for
every frequency simultaneously. Only the online lookup consumes shots.
"""
import math
from dataclasses import dataclass

import numpy as np
from numba import njit


@njit(cache=True)
def _walsh(values):
    width = 1
    while width < len(values):
        for block in range(0, len(values), 2 * width):
            for j in range(block, block + width):
                a, b = values[j], values[j + width]
                values[j], values[j + width] = a + b, a - b
        width *= 2


@njit(cache=True, inline="always")
def _parity(word):
    word ^= word >> 16
    word ^= word >> 8
    word ^= word >> 4
    return (0x6996 >> (word & 15)) & 1


@njit(cache=True)
def _characteristics(work, total, fair_masks, fair_bits):
    magnitude = 0.0
    for z in range(len(work)):
        value = math.exp(min(0.0, (total - work[z]) * 0.5))
        for j in range(len(fair_masks)):
            if _parity(z & fair_masks[j]) != fair_bits[j]:
                value = 0.0
                break
        work[z] = value
        magnitude += value
    return magnitude


@njit(cache=True)
def _choose(work, table, width, classes, bound, unknown):
    for key in range(width):
        if classes == 2 and len(work) == width:
            delta = work[key]
            table[key] = int(delta < 0) if abs(delta) > bound else unknown
        else:
            best, second, winner = -np.inf, -np.inf, 0
            for k in range(classes):
                value = work[k * width + key]
                if value > best:
                    best, second, winner = value, best, k
                elif value > second:
                    second = value
            table[key] = winner if best - second > 2 * bound else unknown


@njit(cache=True)
def _lookup(dets, pivots, deps, relations, key_offset, offsets, frame, nobs, table, unknown):
    out = np.zeros((len(dets), nobs), dtype=np.bool_)
    pending = np.zeros(len(dets), dtype=np.bool_)
    for i in range(len(dets)):
        key = key_offset
        for j in range(len(pivots)):
            if dets[i, pivots[j]]:
                key ^= 1 << j
        for j in range(len(deps)):
            col = deps[j]
            if int(dets[i, col] ^ offsets[col]) != _parity(key & relations[j]):
                pending[i] = True
                break
        result = int(table[key])
        if result == unknown:
            pending[i] = True
        if not pending[i]:
            for j in range(nobs):
                out[i, j] = ((result ^ frame) >> j) & 1
    return out, pending


def _basis(signatures, limit):
    """Reduced GF(2) basis with one pivot detector coordinate per row."""
    rows = {}
    for word in signatures:
        while word:
            bit = word.bit_length() - 1
            if bit not in rows:
                rows[bit] = word
                if len(rows) > limit:
                    return None
                break
            word ^= rows[bit]
    pivots = sorted(rows)
    for i, bit in enumerate(pivots):
        for other in pivots[i + 1:]:
            if rows[other] & (1 << bit):
                rows[other] ^= rows[bit]
    return pivots, rows


@dataclass
class ExactFaultTable:
    pivots: np.ndarray
    dependent: np.ndarray
    relations: np.ndarray
    key_offset: int
    detector_offset: np.ndarray
    frame: int
    num_observables: int
    decisions: np.ndarray
    unknown: int

    @classmethod
    def from_model(cls, model, num_detectors, num_observables, max_rank=24):
        if not 1 <= num_observables <= max_rank:
            return None
        faults = model.faults
        signatures = [sum(1 << site for site in f.detectors) for f in faults]
        extra = 0 if num_observables == 1 else num_observables
        basis = _basis(signatures, max_rank - extra)
        if basis is None:
            return None
        pivots, rows = basis
        rank = len(pivots)

        def project(word):
            return sum(((word >> bit) & 1) << i for i, bit in enumerate(pivots))

        dependent = [i for i in range(num_detectors) if i not in rows]
        relations = [sum(((rows[bit] >> j) & 1) << i for i, bit in enumerate(pivots))
                     for j in dependent]
        key_offset = sum(int(model.detector_offset[bit]) << i for i, bit in enumerate(pivots))
        width = 1 << rank
        work = np.zeros(width << extra, dtype=np.float64)
        logs = []
        fair = set()
        for fault, signature in zip(faults, signatures):
            mask = project(signature) | ((fault.logical << rank) if extra else 0)
            sign = 0 if extra else fault.logical
            if fault.probability == 0.5:
                fair.add((mask, sign))
                continue
            value = math.log1p(-2 * fault.probability)
            logs.append(value)
            work[mask] += -value if sign else value
        _walsh(work)
        total = math.fsum(logs)
        masks = np.asarray([mask for mask, _ in fair], dtype=np.int64)
        bits = np.asarray([bit for _, bit in fair], dtype=np.int64)
        magnitude = _characteristics(work, total, masks, bits)
        _walsh(work)
        # Conservative numerical guard for cancellation in the log products and
        # the unnormalised inverse transform. Unresolved cells use the fallback.
        bound = (4 * len(faults) + 12 * (rank + extra) + 64) * (
            np.finfo(np.float64).eps * (1 + abs(total)) * magnitude)
        classes = 1 << num_observables
        dtype = np.min_scalar_type(classes)
        unknown = int(np.iinfo(dtype).max)
        table = np.empty(width, dtype=dtype)
        _choose(work, table, width, classes, bound, unknown)
        return cls(np.array(pivots, np.int32), np.array(dependent, np.int32),
                   np.array(relations, np.int64), key_offset, model.detector_offset.copy(),
                   model.logical_offset, num_observables, table, unknown)

    def lookup(self, dets):
        array = np.asarray(dets, dtype=np.bool_)
        if array.ndim != 2 or array.shape[1] != len(self.detector_offset):
            raise ValueError("invalid detector array shape")
        return _lookup(array, self.pivots, self.dependent, self.relations, self.key_offset,
                       self.detector_offset, self.frame, self.num_observables, self.decisions, self.unknown)
