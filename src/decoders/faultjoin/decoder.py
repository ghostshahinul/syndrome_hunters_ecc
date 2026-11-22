"""FaultJoin: two-fault logical voting with syndrome-conditioned union–find."""

from __future__ import annotations

import numpy as np
import stim
from numba import njit

from src.decoders.base import Decoder
from src.decoders.faultjoin.condition import conditioned_graph
from src.decoders.faultjoin.faults import logical_bits, read_model
from src.decoders.faultjoin.join import join_decision, make_join
from src.decoders.faultjoin.union_find import _grow_batch


@njit(cache=True)
def _one_vote(row, index, hits):
    count = 0
    for d in range(len(row)):
        if row[d]:
            hits[count] = d
            count += 1
            if count > index.max_hits:
                break
    return join_decision(hits, count, index)


@njit(cache=True)
def _vote_batch(dets, index):
    shots, width = dets.shape
    masks = np.zeros(shots, np.uint64)
    pending = np.zeros(shots, np.bool_)
    hits = np.empty(min(width, index.max_hits + 1), np.int32)
    for shot in range(shots):
        found, masks[shot] = _one_vote(dets[shot], index, hits)
        pending[shot] = not found
    return masks, pending


@njit(cache=True)
def _single(row, index, keep, graph, rules, logical_offset, num_observables):
    hits = np.empty(min(len(row), index.max_hits + 1), np.int32)
    found, frame = _one_vote(row, index, hits)
    if not found and len(keep):
        active = row[keep].reshape(1, -1)
        context = (row.reshape(1, -1), rules[0], rules[1], rules[2], rules[3])
        frame = _grow_batch(active, graph[0], graph[1], graph[2], graph[3], context)[0]
    frame ^= logical_offset
    prediction = np.empty(num_observables, np.bool_)
    for i in range(num_observables):
        prediction[i] = (frame >> np.uint64(i)) & np.uint64(1)
    return prediction


class FaultJoinDecoder(Decoder):
    name = "fault_join"

    def _check_single(self, syndrome: np.ndarray) -> np.ndarray:
        # Kept here (shahi_2 added it to the shared base.py; main's base.py is unchanged).
        row = np.asarray(syndrome, dtype=bool)
        if row.shape != (self.num_detectors,):
            raise ValueError(f"expected one syndrome of length {self.num_detectors}, got {row.shape}")
        return row

    def __init__(self, dem: stim.DetectorErrorModel):
        super().__init__(dem)
        model = read_model(dem)
        self.index = make_join(model.faults, self.num_detectors)
        self.keep, self.graph, self.rules = conditioned_graph(model, self.num_detectors)
        self.offset = model.detector_offset if model.detector_offset.any() else None
        self.frame = np.uint64(model.logical_offset)
        # A single small-code shot can bypass batch allocation and JIT dispatch.
        width = (self.num_detectors + 7) // 8
        frames = np.unique(self.index.decisions)
        answers = dict(zip(frames, logical_bits(frames ^ self.frame, self.num_observables)))
        self.single_table = {int(key).to_bytes(width, "little"): answers[decision]
                             for key, decision in zip(self.index.keys, self.index.decisions)}

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        row = self._check_single(syndrome)
        if self.offset is not None:
            row = row ^ self.offset
        if self.single_table:
            answer = self.single_table.get(np.packbits(row, bitorder="little").tobytes())
            if answer is not None:
                return answer.copy()
        return _single(np.ascontiguousarray(row), self.index, self.keep, self.graph, self.rules,
                       self.frame, self.num_observables)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        dets = self._check_input(detection_events)
        if self.offset is not None:
            dets = dets ^ self.offset
        masks, pending = _vote_batch(dets, self.index)
        if pending.any() and self.keep.size:
            evidence = dets if pending.all() else dets[pending]
            active = np.ascontiguousarray(evidence[:, self.keep])
            masks[pending] = _grow_batch(active, *self.graph, (evidence, *self.rules))
        return logical_bits(masks ^ self.frame, self.num_observables)

