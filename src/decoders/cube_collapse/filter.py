"""Stage 1: temporal filter, vectorized over shots and sites.

3-point majority over time (left pad 0 = known initial state, right pad = replicate the
last value), flat = filtered value at the last round, plus per-node diagnostics.

Note: with right-edge replication, filtered[-1] = maj(r[-2], r[-1], r[-1]) = r[-1], so
`flat` always equals the raw last value. The filter therefore changes `flat` only via
which slice is last (final data layer or not); its main role is the diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONF_SURE, CONF_LATE, CONF_MESSY = 0, 1, 2
CONF_NAMES = {CONF_SURE: "sure", CONF_LATE: "late", CONF_MESSY: "messy"}


@dataclass(frozen=True, eq=False)
class FilterResult:
    filtered: np.ndarray     # (shots, T, n) uint8
    flat: np.ndarray         # (shots, n) uint8: filtered value at the last round
    hold: np.ndarray         # (shots, n) int32: number of final rounds holding that value
    transitions: np.ndarray  # (shots, n) int32: changes in filtered series (from initial 0)
    confidence: np.ndarray   # (shots, n) uint8: CONF_SURE / CONF_LATE / CONF_MESSY


def majority3(raw: np.ndarray) -> np.ndarray:
    """3-point running majority along axis 1 of a (shots, T, n) 0/1 array."""
    raw = np.asarray(raw, dtype=np.uint8)
    left = np.concatenate([np.zeros_like(raw[:, :1]), raw[:, :-1]], axis=1)
    right = np.concatenate([raw[:, 1:], raw[:, -1:]], axis=1)
    return ((left + raw + right) >= 2).astype(np.uint8)


def temporal_filter(raw: np.ndarray, late_k: int = 2) -> FilterResult:
    """Apply the Stage 1 filter to a raw cube (shots, T, n), T >= 1.

    Confidence precedence: messy (transitions >= 2) > late (hold < late_k) > sure.
    """
    raw = np.asarray(raw, dtype=np.uint8)
    if raw.ndim != 3 or raw.shape[1] < 1:
        raise ValueError(f"raw cube must have shape (shots, T>=1, n), got {raw.shape}")
    f = majority3(raw)
    flat = f[:, -1, :]

    prev = np.concatenate([np.zeros_like(f[:, :1]), f[:, :-1]], axis=1)
    transitions = (f != prev).sum(axis=1, dtype=np.int32)

    same_as_last = (f == flat[:, None, :])[:, ::-1, :]
    hold = np.cumprod(same_as_last, axis=1, dtype=np.int32).sum(axis=1, dtype=np.int32)

    conf = np.full(flat.shape, CONF_SURE, dtype=np.uint8)
    conf[hold < late_k] = CONF_LATE
    conf[transitions >= 2] = CONF_MESSY
    return FilterResult(f, flat, hold, transitions, conf)


def majority_flat(raw: np.ndarray) -> np.ndarray:
    """Baseline only: whole-window majority vote per node (ties -> last raw value)."""
    raw = np.asarray(raw, dtype=np.uint8)
    T = raw.shape[1]
    ones = raw.sum(axis=1, dtype=np.int32)
    out = (2 * ones > T).astype(np.uint8)
    tie = 2 * ones == T
    out[tie] = raw[:, -1, :][tie]
    return out
