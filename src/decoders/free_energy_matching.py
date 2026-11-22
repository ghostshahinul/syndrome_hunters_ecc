"""Free-energy matching: correlated matching with a logical-class free-energy vote.

Correlated matching returns the lowest-weight correction it finds, then
reports that correction's logical class. The optimal decoder instead compares
the total probability of the two classes. They disagree when the best
corrections of the two classes have similar weight, so this decoder:

1. Decodes every shot with correlated matching on the regrouped model.
2. Measures the logical gap: the weight difference between the best plain
   matching correction of the opposite class (forced through LogicalSplit)
   and the best correction overall. Shots with a gap of at least
   `gap_threshold`, on which plain and correlated matching also agree, keep
   the correlated answer. Almost all logical failures lie below the threshold.
3. On the remaining shots, asks each member matcher for its best correction
   inside each class, scores every distinct correction by its free energy
   (src/decoders/free_energy.py: hyperedge-aware energy minus local cycle
   entropy) and picks the class with the larger summed weight exp(-F).

Steps 1 and 2 are batch matching calls. Step 3 runs on a small fraction of
shots, and its cost depends on correction size rather than the code size, so
the method scales with code distance like matching does.
"""

from __future__ import annotations

import numpy as np
import pymatching
import stim

from src.decoders.base import Decoder
from src.decoders.free_energy import CorrectionScorer
from src.decoders.model import LogicalSplit, center, regroup, restrict


class FreeEnergyMatchingDecoder(Decoder):
    name = "free_energy_matching"

    def __init__(self, dem: stim.DetectorErrorModel, *, odds_scale: float = 0.5,
                 member_scales: tuple[float, ...] = (0.5,), score_scale: float = 0.5,
                 gap_threshold: float = 4.0, descent_steps: int = 8, cycle_length: int = 4):
        super().__init__(dem)
        self.gap_threshold = gap_threshold
        self.split = None
        # Faults above probability 1/2 become a known reference flip.
        source, offset, self.frame = center(dem)
        self.offset = offset if offset.any() else None
        try:
            base = regroup(source, odds_scale)
        except ValueError:
            # Unsupported model (a fault that flips an observable and no
            # detector): correlated matching on the centered model as given.
            base = None
        self.matching = pymatching.Matching.from_detector_error_model(
            source if base is None else base, enable_correlations=True)
        self.matching.ensure_num_fault_ids(self.num_observables)
        if base is None:
            return
        self.split = LogicalSplit.build(base)
        if self.split is None:
            return
        # Plain matching factorizes over sectors, so the gap needs only the
        # sector that carries the observable (plus the two split nodes).
        sector = self.split.sector
        self.plain = pymatching.Matching.from_detector_error_model(restrict(base, sector))
        self.gate = pymatching.Matching.from_detector_error_model(
            restrict(self.split.dem, np.concatenate([sector, [True, True]])))
        self.members = []
        for scale in member_scales:
            split = self.split if scale == odds_scale else LogicalSplit.build(regroup(source, scale))
            self.members.append(pymatching.Matching.from_detector_error_model(
                split.dem, enable_correlations=True))
        scored = self.split if score_scale == odds_scale else LogicalSplit.build(regroup(source, score_scale))
        self.scorer = CorrectionScorer(scored.dem, descent_steps=descent_steps, cycle_length=cycle_length)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        dets = self._check_input(detection_events)
        if not len(dets) or not self.num_observables:
            return np.zeros((len(dets), self.num_observables), dtype=bool)
        if self.offset is not None:
            dets = dets ^ self.offset
        answer = self.matching.decode_batch(dets, enable_correlations=True).astype(bool)
        if self.split is not None:
            hard = self.hard_shots(dets, answer[:, 0])
            if len(hard):
                answer[hard, 0] = self.vote(dets[hard], answer[hard, 0])
        return answer ^ self.frame

    def hard_shots(self, dets: np.ndarray, correlated: np.ndarray) -> np.ndarray:
        """Indices of shots whose logical gap is below the threshold."""
        local = dets & self.split.sector
        plain, best = self.plain.decode_batch(local, return_weights=True)
        plain = plain[:, 0].astype(bool)
        shift = self.split.shift(dets)
        _, other = self.gate.decode_batch(self.split.force(local, ~(plain ^ shift)), return_weights=True)
        return np.flatnonzero((other - best < self.gap_threshold) | (plain != correlated))

    def vote(self, dets: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        """Free-energy class decision for each shot; fallback where undecided."""
        shots = len(dets)
        forced = [self.split.force(dets, np.full(shots, c, dtype=bool)) for c in (False, True)]
        pairs, offsets, cls = [], [0], []
        for i in range(shots):
            for c in (0, 1):
                row = forced[c][i]
                for member in self.members:
                    edges = member.decode_to_edges_array(row, enable_correlations=True)
                    pairs.append(edges)
                    offsets.append(offsets[-1] + len(edges))
                    cls.append(c)
        energy, entropy, ident = self.scorer.score(
            np.concatenate(pairs) if offsets[-1] else np.zeros((0, 2), np.int64), np.array(offsets))
        cls = np.asarray(cls)
        free = energy - entropy
        out = fallback.copy()
        shift = self.split.shift(dets)
        block = 2 * len(self.members)
        for i in range(shots):
            mass = [-np.inf, -np.inf]
            seen = set()
            for k in range(i * block, (i + 1) * block):
                if not np.isfinite(free[k]) or (cls[k], ident[k]) in seen:
                    continue
                seen.add((cls[k], ident[k]))
                mass[cls[k]] = np.logaddexp(mass[cls[k]], -free[k])
            if np.isfinite(mass[0]) or np.isfinite(mass[1]):
                out[i] = bool(mass[1] > mass[0]) ^ shift[i]
        return out
