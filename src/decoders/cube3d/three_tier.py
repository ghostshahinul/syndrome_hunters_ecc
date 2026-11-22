"""Three tiers: Cube3D-Local -> UnionFind3D -> MWPM3D (our own exact matching).

Per shot:
  1. Hierarchical (fast path + UnionFind3D, whole-shot gate) decodes every shot and reports
     the node count of UF's largest final cluster (0 when the fast path finished the shot).
  2. Shots whose largest cluster has more than `threshold` nodes are decoded again by MWPM3D,
     and the MWPM answer replaces UF's. Large clusters are where UF's greedy growth and
     peeling are most likely to differ from minimum-weight matching.
threshold=None -> 2 * (number of time layers - 1), i.e. 2 * rounds (2d for rounds = d).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import stim

from src.decoders.base import Decoder
from src.decoders.cube3d.graph import DetectorGraph3D, build_detector_graph
from src.decoders.cube3d.hierarchical import Hierarchical, HierarchicalConfig
from src.decoders.cube3d.mwpm import MWPM3D


@dataclass(frozen=True)
class ThreeTierConfig:
    threshold: int | None = None  # route to MWPM when UF's largest cluster has > threshold nodes
    hier: HierarchicalConfig = field(default_factory=HierarchicalConfig)

    def __post_init__(self) -> None:
        if self.threshold is not None and self.threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {self.threshold}")


@dataclass(eq=False)
class ThreeTierResult:
    predictions: np.ndarray   # (shots, n_obs) bool
    routed: np.ndarray        # (shots,) bool: decoded by MWPM3D
    fast: np.ndarray          # (shots,) bool: finished by the fast path
    max_cluster: np.ndarray   # (shots,) int32

    @property
    def routed_fraction(self) -> float:
        return float(self.routed.mean()) if self.routed.size else 0.0


class ThreeTier:
    def __init__(self, graph: DetectorGraph3D, config: ThreeTierConfig | None = None):
        self.graph = graph
        self.config = config or ThreeTierConfig()
        self.hier = Hierarchical(graph, self.config.hier)
        self.mwpm = MWPM3D(graph)
        layers = len(np.unique(graph.coords[:, 2]))
        self.default_threshold = 2 * max(layers - 1, 1)

    @property
    def threshold(self) -> int:
        t = self.config.threshold
        return self.default_threshold if t is None else t

    def run(self, detection_events: np.ndarray, threshold: int | None = None) -> ThreeTierResult:
        T = self.threshold if threshold is None else threshold
        dets = np.ascontiguousarray(detection_events, dtype=np.uint8)
        h = self.hier.run(dets)
        pred = h.predictions.copy()
        routed = h.max_cluster > T
        if routed.any():
            pred[routed] = self.mwpm.run(dets[routed])
        return ThreeTierResult(pred, routed, h.fast, h.max_cluster)


class ThreeTierDecoder(Decoder):
    name = "hierarchical_mwpm"
    slow = True  # pure-Python blossom (networkx)

    def __init__(self, dem: stim.DetectorErrorModel, config: ThreeTierConfig | None = None):
        super().__init__(dem)
        self.engine = ThreeTier(build_detector_graph(dem), config)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        return self.engine.run(self._check_input(detection_events)).predictions
