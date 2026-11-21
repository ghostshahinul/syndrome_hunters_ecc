"""Configuration for the Cube-Collapse decoder."""

from __future__ import annotations

from dataclasses import dataclass

BACKENDS = ("reference",)


@dataclass(frozen=True)
class CubeCollapseConfig:
    """
    Attributes:
        late_k: a node whose filtered value has held for fewer than `late_k` final
            rounds is classed "late" (lower confidence -> processed later in Stage 4).
        include_final_layer: append Stim's final data-readout detector layer as the
            last time slice of the cube (recommended; it makes `flat` the data parity).
        backend: implementation of the hot loops. "reference" is pure NumPy/Python
            and is the source of truth for every other backend.
    """

    late_k: int = 2
    include_final_layer: bool = True
    backend: str = "reference"

    def __post_init__(self) -> None:
        if self.late_k < 1:
            raise ValueError(f"late_k must be >= 1, got {self.late_k}")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
