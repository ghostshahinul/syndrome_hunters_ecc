"""PyMatching (MWPM) wrapper: the reference decoder everything is compared against."""

from __future__ import annotations

import numpy as np
import pymatching
import stim

from src.decoders.base import Decoder
from src.decoders.model import center, regroup


class PyMatchingDecoder(Decoder):
    name = "pymatching"
    correlations = False

    def __init__(self, dem: stim.DetectorErrorModel):
        super().__init__(dem)
        self.matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=self.correlations)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        dets = self._check_input(detection_events)
        return self.matching.decode_batch(dets, enable_correlations=self.correlations).astype(bool)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        return self.matching.decode(self._check_single(syndrome), enable_correlations=self.correlations).astype(bool)


class CorrelatedMatchingDecoder(PyMatchingDecoder):
    """Two-pass correlated matching, a stronger reference for circuit faults."""

    name = "pymatching_correlated"
    correlations = True


class RegroupedCorrelatedMatchingDecoder(CorrelatedMatchingDecoder):
    """Correlated matching on the regrouped model with fault odds scaled by 1/2
    per component. This is stage 1 of free-energy matching on its own, and
    isolates what the logical-class vote adds."""

    name = "pymatching_correlated_regrouped"

    def __init__(self, dem: stim.DetectorErrorModel, odds_scale: float = 0.5):
        Decoder.__init__(self, dem)
        source, offset, self.frame = center(dem)
        self.offset = offset if offset.any() else None
        try:
            model = regroup(source, odds_scale)
        except ValueError:
            model = source
        self.matching = pymatching.Matching.from_detector_error_model(model, enable_correlations=True)
        self.matching.ensure_num_fault_ids(self.num_observables)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        dets = self._check_input(detection_events)
        if not self.num_observables:
            return np.zeros((len(dets), 0), dtype=bool)
        if self.offset is not None:
            dets = dets ^ self.offset
        return super().decode_batch(dets) ^ self.frame

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        return self.decode_batch(self._check_single(syndrome)[None, :])[0]
