"""Build Stim surface-code memory circuits under circuit-level depolarizing noise."""

from __future__ import annotations

import stim


def surface_code_circuit(
    distance: int,
    p: float,
    rounds: int | None = None,
    basis: str = "z",
) -> stim.Circuit:
    """Rotated surface-code memory experiment.

    Args:
        distance: code distance d (odd, >= 3).
        p: physical error rate applied uniformly to every noise channel.
        rounds: syndrome-extraction rounds; defaults to d.
        basis: "z" or "x" memory experiment.
    """
    if basis not in ("x", "z"):
        raise ValueError(f"basis must be 'x' or 'z', got {basis!r}")
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds if rounds is not None else distance,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


def detector_error_model(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    """Graph-like DEM (hyperedges decomposed) suitable for matching decoders."""
    return circuit.detector_error_model(decompose_errors=True)
