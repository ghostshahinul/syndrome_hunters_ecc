"""Weighted union–find on the projected detector graph, written for this project."""

from __future__ import annotations

import numpy as np
import stim
from numba import njit

from src.decoders.base import Decoder
from src.decoders.faults import edge_arrays, logical_bits, read_model


@njit(cache=True)
def _find(node, parent, frame):
    """Return the root and XOR of logical edge labels along the forest path."""
    label = np.uint64(0)
    while parent[node] != node:
        up = parent[node]
        frame[node] ^= frame[up]
        parent[node] = parent[up]
        label ^= frame[node]
        node = parent[node]
    return node, label


@njit(cache=True)
def _grow_batch(dets, left, right, length, labels, context=None):
    shots, boundary = dets.shape
    output = np.zeros(shots, dtype=np.uint64)
    vertices = boundary + 1
    parent = np.empty(vertices, np.int32)
    size = np.empty(vertices, np.int32)
    odd = np.empty(vertices, np.bool_)
    frame = np.empty(vertices, np.uint64)
    roots = np.empty(vertices, np.int32)
    growth = np.empty(len(left), np.int32)
    ready = np.empty(len(left), np.int32)
    limits = length if context is None else length.copy()
    for shot in range(shots):
        if not dets[shot].any():
            continue
        if context is not None:
            # A co-firing partner can lower the cost of an edge in this shot.
            evidence, starts, other, edge, ticks = context
            limits[:] = length
            for d in range(evidence.shape[1]):
                if evidence[shot, d]:
                    for j in range(starts[d], starts[d + 1]):
                        if other[j] < 0 or evidence[shot, other[j]]:
                            e = edge[j]
                            limits[e] = min(limits[e], ticks[j])
        for i in range(vertices):
            parent[i] = i
        size[:] = 1
        odd[:boundary] = dets[shot]
        odd[boundary] = False
        frame[:] = 0
        growth[:] = 0
        active = np.count_nonzero(dets[shot])
        while active:
            for i in range(vertices):
                roots[i] = _find(i, parent, frame)[0]
            count = 0
            progressed = False
            # Every odd, non-boundary cluster grows a half-edge in this tick.
            # Collect contacts before merging so enumeration order cannot give
            # a newly formed cluster an extra growth step.
            for e in range(len(left)):
                a, b = roots[left[e]], roots[right[e]]
                if a == b:
                    continue
                speed = int(odd[a]) + int(odd[b])
                if speed:
                    progressed = True
                    growth[e] += speed
                    if growth[e] >= limits[e]:
                        ready[count] = e
                        count += 1
            if not progressed:
                raise ValueError("syndrome has odd parity in a component without a boundary")
            for j in range(count):
                e = ready[j]
                a, fa = _find(left[e], parent, frame)
                b, fb = _find(right[e], parent, frame)
                if a == b:
                    continue
                if b == boundary or (a != boundary and size[a] < size[b]):
                    a, b = b, a
                active -= int(odd[a]) + int(odd[b])
                parent[b] = a
                frame[b] = fa ^ labels[e] ^ fb
                size[a] += size[b]
                odd[a] = (odd[a] != odd[b]) and a != boundary
                active += int(odd[a])
        # XOR paths from defects to roots. Shared branches cancel, exactly as
        # in leaf peeling. The virtual boundary is always its own root.
        for i in range(boundary):
            if dets[shot, i]:
                output[shot] ^= _find(i, parent, frame)[1]
    return output


class UnionFindDecoder(Decoder):
    name = "union_find"

    def __init__(self, dem: stim.DetectorErrorModel):
        super().__init__(dem)
        model = read_model(dem)
        self.graph = growth_graph(model, self.num_detectors)
        self.offset = model.detector_offset if model.detector_offset.any() else None
        self.frame = np.uint64(model.logical_offset)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        dets = self._check_input(detection_events)
        if self.offset is not None:
            dets = dets ^ self.offset
        masks = _grow_batch(dets, *self.graph) ^ self.frame
        return logical_bits(masks, self.num_observables)


def growth_graph(model, num_detectors):
    left, right, weight, labels = edge_arrays(model, num_detectors)
    unit = growth_unit(weight)
    lengths = np.maximum(1, np.rint(weight / unit)).astype(np.int32)
    return left, right, lengths, labels


def growth_unit(weight):
    positive = weight[weight > 0]
    return positive.min() / 4 if positive.size else 1.0
