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


def code_capacity_circuit(distance: int, p: float, basis: str = "z") -> stim.Circuit:
    """Code-capacity noise: one round, data depolarization only, perfect measurements."""
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=1,
        before_round_data_depolarization=p,
    )


def phenomenological_circuit(
    distance: int, p: float, rounds: int | None = None, basis: str = "z"
) -> stim.Circuit:
    """Phenomenological noise: data depolarization each round + measurement flips."""
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds if rounds is not None else distance,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
    )


NOISE_TIERS = ("code_capacity", "phenomenological", "circuit")


def tier_circuit(
    tier: str, distance: int, p: float, rounds: int | None = None, basis: str = "z"
) -> stim.Circuit:
    """Dispatch on noise tier name: code_capacity | phenomenological | circuit."""
    if tier == "code_capacity":
        return code_capacity_circuit(distance, p, basis)
    if tier == "phenomenological":
        return phenomenological_circuit(distance, p, rounds, basis)
    if tier == "circuit":
        return surface_code_circuit(distance, p, rounds, basis)
    raise ValueError(f"unknown noise tier {tier!r}; choose from {NOISE_TIERS}")


def detector_error_model(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    """Graph-like DEM (hyperedges decomposed) suitable for matching decoders."""
    return circuit.detector_error_model(decompose_errors=True)
