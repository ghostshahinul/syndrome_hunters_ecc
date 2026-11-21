"""Our novel decoder idea. TODO: implement.

Keep it behind the same Decoder interface so evaluate.py and the tests pick it up
automatically.
"""

from __future__ import annotations

import numpy as np
import stim

from src.decoders.base import Decoder


class ExperimentalDecoder(Decoder):
    name = "experimental"

    def __init__(self, dem: stim.DetectorErrorModel):
        super().__init__(dem)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ExperimentalDecoder is not implemented yet")
