"""PyMatching (MWPM) wrapper: the reference decoder everything is compared against."""

from __future__ import annotations

import numpy as np
import pymatching
import stim

from src.decoders.base import Decoder


class PyMatchingDecoder(Decoder):
    name = "pymatching"

    def __init__(self, dem: stim.DetectorErrorModel):
        super().__init__(dem)
        self.matching = pymatching.Matching.from_detector_error_model(dem)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        dets = self._check_input(detection_events)
        return self.matching.decode_batch(dets).astype(bool)
