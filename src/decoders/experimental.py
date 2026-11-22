"""FaultJoin: exact logical posteriors, sparse path sums, and correlated matching."""
import numpy as np
import pymatching

from src.decoders.base import Decoder
from src.decoders.exact import ExactFaultTable
from src.decoders.soft_paths import SoftPathModel
from src.decoders.faults import read_model
from src.decoders.frontier import frontier_screen, path_model
from src.decoders.sectors import sector_model


class FaultJoinDecoder(Decoder):
    name = "fault_join"

    def __init__(self, dem):
        super().__init__(dem)
        model = read_model(dem)
        self.exact = ExactFaultTable.from_model(model, self.num_detectors, self.num_observables)
        self.offset = model.detector_offset if model.detector_offset.any() else None
        self.frame = np.array([model.logical_offset & (1 << bit) for bit in range(self.num_observables)], dtype=bool)
        self.paths = None
        self.soft = None
        if self.exact is None and self.num_observables == 1:
            projected = read_model(sector_model(dem, center=True))
            self.paths = path_model(projected, self.num_detectors)
            if self.paths is not None:
                self.soft = SoftPathModel(projected, self.num_detectors)
        self.matching = pymatching.Matching.from_detector_error_model(
            sector_model(dem, 0.5, center=True), enable_correlations=True)
        self.matching.ensure_num_fault_ids(self.num_observables)

    def decode_batch(self, detection_events):
        rows = self._check_input(detection_events)
        if not len(rows):
            return np.empty((0, self.num_observables), dtype=bool)
        if self.exact is not None:
            answer, pending = self.exact.lookup(rows)
            if not pending.any():
                return answer
            answer[pending] = self._complete(rows[pending])
            return answer
        return self._complete(rows)

    def _complete(self, rows):
        centered = rows if self.offset is None else rows ^ self.offset
        if self.paths is None:
            answer = self.matching.decode_batch(centered, enable_correlations=True).astype(bool)
        else:
            decision, pending = frontier_screen(centered, *self.paths, 7, .005, 100.)
            answer = decision[:, None]
            if pending.any():
                unresolved = np.flatnonzero(pending)
                subset = centered[unresolved]
                soft, defer = self.soft.predict(subset, min_margin=.65)
                answer[unresolved, 0] = soft
                if defer.any():
                    answer[unresolved[defer]] = self.matching.decode_batch(subset[defer], enable_correlations=True)
        return answer ^ self.frame


ExperimentalDecoder = FaultJoinDecoder
