"""Compile the boundary-restored Ising decision on small, fully enumerable DEMs."""

import time

import numpy as np
from numba import njit

from src.decoders.base import Decoder
from src.decoders.faults import read_model
from src.decoders.ising import BOUNDARY_HYBRID, IsingHybrid
from src.ising_data import DEFAULT_CHECKPOINT, DEFAULT_REPO

TABLE_NAME = "ising_boundary_table"


@njit(cache=True, inline="always")
def _syndrome_key(row, weights):
    key = 0
    for col in range(len(row)):
        if row[col]:
            key |= weights[col]
    return key


@njit(cache=True)
def _lookup_batch(dets, weights, predictions, counts):
    output = np.empty((len(dets), 1), dtype=np.bool_)
    outside = np.empty(len(dets), dtype=np.intp)
    n_outside = 0
    events = 0
    for i in range(len(dets)):
        key = _syndrome_key(dets[i], weights)
        if key < len(predictions):
            output[i, 0] = predictions[key]
            events += int(counts[key])
        else:
            outside[n_outside] = i
            n_outside += 1
    return output, events, outside[:n_outside]


def active_detectors(dem):
    model = read_model(dem)
    active = sorted({d for fault in model.faults for d in fault.detectors})
    if len(active) > 20 or model.detector_offset.any():
        raise ValueError("Exhaustive Ising lookup supports up to 20 active detectors and a zero reference offset")
    # Full rank guarantees every enumerated word is an attainable syndrome.
    pivots = {}
    for fault in model.faults:
        mask = sum(1 << d for d in fault.detectors)
        while mask:
            bit = mask.bit_length() - 1
            if bit not in pivots:
                pivots[bit] = mask
                break
            mask ^= pivots[bit]
    if len(pivots) != len(active):
        raise ValueError("Active detector columns must have independent syndrome coordinates for exhaustive lookup")
    return np.asarray(active, dtype=int)


class CompiledIsingHybrid(IsingHybrid):
    """A CPU decision table populated from the pretrained hybrid without shot labels."""

    @classmethod
    def from_experiment(cls, name, exp, repo=DEFAULT_REPO, checkpoint=DEFAULT_CHECKPOINT,
                        device="auto", precision="auto"):
        if name != TABLE_NAME:
            raise ValueError(f"Unknown compiled hybrid {name!r}")
        active = active_detectors(exp.dem)
        reference = IsingHybrid.from_experiment(BOUNDARY_HYBRID, exp, repo, checkpoint, device, precision)
        return cls(reference, active)

    def __init__(self, reference, active):
        Decoder.__init__(self, reference.dem)
        self.name = TABLE_NAME
        self.reference = reference
        self.predecoder = reference.predecoder
        bits = len(active)
        size = 1 << bits
        # The extra high bit sends any fired inactive column to the streaming decoder.
        self.key_weights = np.full(self.num_detectors, size, dtype=np.int32)
        self.key_weights[active] = 1 << np.arange(bits, dtype=np.int32)
        self.predictions = np.empty(size, dtype=bool)
        self.counts = np.empty(size, dtype=np.min_scalar_type(self.num_detectors))
        powers = np.arange(bits, dtype=np.uint32)
        self.predecoder.reset_peak()
        start = time.perf_counter()
        # Fixed-size compilation batches keep GPU memory bounded. The table is
        # constructed before encountering any Monte Carlo data, including warm-up.
        for begin in range(0, size, 64):
            end = min(begin + 64, size)
            keys = np.arange(begin, end, dtype=np.uint32)
            dets = np.zeros((len(keys), self.num_detectors), dtype=bool)
            dets[:, active] = (keys[:, None] >> powers) & 1
            self.predictions[begin:end] = reference.decode_batch(dets)[:, 0]
            self.counts[begin:end] = np.count_nonzero(reference.last_residual, axis=1)
        self.table_info = dict(
            device="cpu_lookup", table_lookup_engine="numba", compile_device=self.predecoder.info["device"],
            table_build_s=time.perf_counter() - start, table_entries=size,
            table_build_gpu_peak_mib=self.predecoder.peak_mib(),
            table_bytes=self.predictions.nbytes + self.counts.nbytes,
        )

    def decode_batch(self, detection_events):
        dets = self._check_input(detection_events)
        result, self.last_event_count, outside = _lookup_batch(
            dets, self.key_weights, self.predictions, self.counts)
        self.last_ai_s = self.last_global_s = 0.
        # Preserve the reference's handling of externally supplied off-support
        # shots. The audited Stim generator cannot fire an inactive detector.
        if len(outside):
            result[outside] = self.reference.decode_batch(dets[outside])
            self.last_event_count += int(np.count_nonzero(self.reference.last_residual))
            self.last_ai_s = self.reference.last_ai_s
            self.last_global_s = self.reference.last_global_s
        return result

    def decode(self, syndrome):
        row = self._check_single(syndrome)
        key = _syndrome_key(row, self.key_weights)
        if key >= len(self.predictions):
            return self.reference.decode(row)
        return self.predictions[key:key+1].copy()

    def residual_events(self):
        return self.last_event_count
