"""The Decoder interface every decoder must implement.

Contract:
  * Constructed from a stim.DetectorErrorModel (all setup/precomputation happens here,
    so it is excluded from per-shot latency).
  * decode_batch(dets) takes a bool array of shape (shots, num_detectors) and returns a
    bool array of shape (shots, num_observables): the predicted observable flips.
  * Decoding must be deterministic and must not mutate its input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import stim


class Decoder(ABC):
    name: str = "abstract"

    def __init__(self, dem: stim.DetectorErrorModel):
        self.dem = dem
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables

    @abstractmethod
    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        """(shots, num_detectors) bool -> (shots, num_observables) bool."""

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Single shot: (num_detectors,) bool -> (num_observables,) bool."""
        return self.decode_batch(self._check_single(syndrome)[None, :])[0]

    def _check_single(self, syndrome: np.ndarray) -> np.ndarray:
        row = np.asarray(syndrome, dtype=bool)
        if row.shape != (self.num_detectors,):
            raise ValueError(f"expected one syndrome of length {self.num_detectors}, got {row.shape}")
        return row

    def _check_input(self, detection_events: np.ndarray) -> np.ndarray:
        dets = np.asarray(detection_events, dtype=bool)
        if dets.ndim != 2 or dets.shape[1] != self.num_detectors:
            raise ValueError(
                f"expected shape (shots, {self.num_detectors}), got {dets.shape}"
            )
        return dets
