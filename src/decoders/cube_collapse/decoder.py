"""Cube-Collapse decoder: cube -> temporal filter -> 2D Clique pass -> greedy completion.

Every shot is fully resolved: syndrome(correction) == flat is guaranteed (and asserted
in tests). No unresolved flags, no hand-off to a global decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import stim

from src.decoders.base import Decoder
from src.decoders.cube_collapse.clique import color_cliques, stage_s1, stage_s2, stage_s3
from src.decoders.cube_collapse.completion import stage_s4
from src.decoders.cube_collapse.config import CubeCollapseConfig
from src.decoders.cube_collapse.filter import CONF_SURE, FilterResult, temporal_filter
from src.decoders.cube_collapse.graph import CheckGraph, build_check_graph
from src.decoders.cube_collapse.layout import CubeLayout, build_cube

STAGES = ("s1", "s2", "s3", "s4")


@dataclass(eq=False)
class CubeCollapseResult:
    predictions: np.ndarray   # (shots, n_obs) bool
    correction: np.ndarray    # (shots, E) uint8 edge flips
    flat: np.ndarray          # (shots, n) uint8 syndrome that was decoded
    confidence: np.ndarray    # (shots, n) uint8
    cleared: dict[str, np.ndarray]  # stage -> (shots,) lit nodes cleared
    radius: np.ndarray        # (shots,) max Stage 4 radius used (0 = Stage 4 not needed)
    n_colors: int
    filter: FilterResult | None = field(default=None, repr=False)

    @property
    def stats(self) -> dict:
        shots = self.flat.shape[0]
        lit = int(self.flat.sum())
        out = {"shots": shots, "lit_initial": lit, "n_colors": self.n_colors}
        for st in STAGES:
            c = int(self.cleared[st].sum())
            out[f"cleared_{st}"] = c
            out[f"share_{st}"] = c / lit if lit else 0.0
        out["frac_shots_s4"] = float((self.cleared["s4"] > 0).mean()) if shots else 0.0
        out["max_radius"] = int(self.radius.max(initial=0))
        return out


class CubeCollapseDecoder(Decoder):
    name = "cube_collapse"

    def __init__(self, dem: stim.DetectorErrorModel, config: CubeCollapseConfig | None = None):
        super().__init__(dem)
        self.config = config or CubeCollapseConfig()
        self.layout = CubeLayout.from_dem(dem)
        self.graph: CheckGraph = build_check_graph(dem, self.layout)
        self.color_groups = color_cliques(self.graph)

    @property
    def n_colors(self) -> int:
        return len(self.color_groups)

    # ---- Decoder interface -------------------------------------------------------
    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        return self.decode_full(detection_events).predictions

    # ---- full pipeline -------------------------------------------------------------
    def decode_full(self, detection_events: np.ndarray) -> CubeCollapseResult:
        dets = self._check_input(detection_events)
        raw = build_cube(dets, self.layout, self.config.include_final_layer)
        return self.decode_cube(raw)

    def decode_cube(self, raw: np.ndarray) -> CubeCollapseResult:
        """Window-ready entry point: raw ancilla cube (shots, T, n_sites), any T >= 1."""
        raw = np.asarray(raw, dtype=np.uint8)
        if raw.ndim != 3 or raw.shape[2] != self.graph.n_nodes:
            raise ValueError(f"expected (shots, T, {self.graph.n_nodes}), got {raw.shape}")
        filt = temporal_filter(raw, self.config.late_k)
        result = self.decode_flat(filt.flat, filt.confidence)
        result.filter = filt
        return result

    def decode_flat(self, flat: np.ndarray, confidence: np.ndarray | None = None) -> CubeCollapseResult:
        """Stages 3-4 on a 2D syndrome (shots, n_sites)."""
        flat = np.ascontiguousarray(flat, dtype=np.uint8)
        if flat.ndim != 2 or flat.shape[1] != self.graph.n_nodes:
            raise ValueError(f"expected (shots, {self.graph.n_nodes}), got {flat.shape}")
        if confidence is None:
            confidence = np.full(flat.shape, CONF_SURE, dtype=np.uint8)

        lit = flat.copy()
        flips = np.zeros((flat.shape[0], self.graph.n_edges), dtype=np.uint8)
        cleared = {
            "s1": stage_s1(lit, flips, self.graph, self.color_groups),
            "s2": stage_s2(lit, flips, self.graph, self.color_groups),
            "s3": stage_s3(lit, flips, self.graph, self.color_groups),
        }
        cleared["s4"], radius = stage_s4(lit, flips, confidence, self.graph)
        return CubeCollapseResult(
            predictions=self.graph.observables(flips),
            correction=flips,
            flat=flat,
            confidence=np.asarray(confidence, dtype=np.uint8),
            cleared=cleared,
            radius=radius,
            n_colors=self.n_colors,
        )
