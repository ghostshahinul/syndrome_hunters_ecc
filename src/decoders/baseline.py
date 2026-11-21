"""PyMatching (MWPM) wrapper: the reference decoder everything is compared against."""

from __future__ import annotations

import numpy as np
import pymatching
import stim

from src.decoders.base import Decoder


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
