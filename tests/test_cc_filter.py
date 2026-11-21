"""Cube-Collapse Stage 0 (cube) and Stage 1 (temporal filter) on hand-built series."""

import numpy as np
import pytest

from src.circuits import surface_code_circuit
from src.decoders.cube_collapse.filter import (
    CONF_LATE, CONF_MESSY, CONF_SURE, majority_flat, temporal_filter,
)
from src.decoders.cube_collapse.layout import CubeLayout, build_cube


def run(series, late_k=2):
    raw = np.array(series, dtype=np.uint8)[None, :, None]
    r = temporal_filter(raw, late_k)
    return (r.filtered[0, :, 0].tolist(), int(r.flat[0, 0]), int(r.hold[0, 0]),
            int(r.transitions[0, 0]), int(r.confidence[0, 0]))


# series, filtered, flat, hold, transitions, confidence
CASES = {
    "flat":         ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], 0, 5, 0, CONF_SURE),
    "flat_one":     ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 1, 5, 1, CONF_SURE),
    "early_step":   ([0, 1, 1, 1, 1], [0, 1, 1, 1, 1], 1, 4, 1, CONF_SURE),
    "mid_step":     ([0, 0, 1, 1, 1], [0, 0, 1, 1, 1], 1, 3, 1, CONF_SURE),
    "single_blip":  ([0, 0, 1, 0, 0], [0, 0, 0, 0, 0], 0, 5, 0, CONF_SURE),
    "late_step":    ([0, 0, 0, 0, 1], [0, 0, 0, 0, 1], 1, 1, 1, CONF_LATE),
    "double_blip":  ([0, 1, 1, 0, 0], [0, 1, 1, 0, 0], 0, 2, 2, CONF_MESSY),
    "messy":        ([1, 1, 0, 0, 1, 1, 1], [1, 1, 0, 0, 1, 1, 1], 1, 3, 3, CONF_MESSY),
    "late_messy":   ([0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1], 1, 1, 3, CONF_MESSY),
}


@pytest.mark.parametrize("name", CASES)
def test_filter_hand_built(name):
    series, filtered, flat, hold, transitions, conf = CASES[name]
    assert run(series) == (filtered, flat, hold, transitions, conf)


def test_flat_is_last_raw_value():
    """Right-edge replication makes flat == raw[-1] for every series (documented)."""
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 2, size=(500, 6, 7), dtype=np.uint8)
    np.testing.assert_array_equal(temporal_filter(raw).flat, raw[:, -1, :])


def test_late_k_changes_only_confidence():
    a, b = run([0, 0, 0, 1, 1], late_k=2), run([0, 0, 0, 1, 1], late_k=3)
    assert a[:4] == b[:4]
    assert (a[4], b[4]) == (CONF_SURE, CONF_LATE)


def test_whole_window_majority_baseline_misses_late_steps():
    raw = np.array([[0, 0, 0, 1, 1]], dtype=np.uint8)[:, :, None]
    assert majority_flat(raw)[0, 0] == 0          # misses the data error
    assert temporal_filter(raw).flat[0, 0] == 1   # end-state filter catches it
    tie = np.array([[0, 0, 1, 1]], dtype=np.uint8)[:, :, None]
    assert majority_flat(tie)[0, 0] == 1          # tie -> last raw value


@pytest.fixture(scope="module")
def layout_d3():
    return CubeLayout.from_circuit(surface_code_circuit(3, 0.001))


@pytest.mark.parametrize("basis", ["z", "x"])
@pytest.mark.parametrize("d", [3, 5, 7])
def test_layout_shape(d, basis):
    lay = CubeLayout.from_circuit(surface_code_circuit(d, 0.001, basis=basis))
    assert lay.n_layers == d + 1
    assert lay.n_sites == (d * d - 1) // 2


def test_cube_cumulative_xor(layout_d3):
    """A data error before round k fires one detection at t=k; raw stays 1 from k on."""
    lay = layout_d3
    dets = np.zeros((1, lay.det_index.max() + 1), dtype=bool)
    dets[0, lay.det_index[2, 1]] = True
    raw = build_cube(dets, lay)
    assert raw.shape == (1, lay.n_layers, lay.n_sites)
    assert raw[0, :, 1].tolist() == [0, 0, 1, 1]
    assert raw[0, :, [0, 2, 3]].sum() == 0


def test_final_round_measurement_blip(layout_d3):
    """Measurement error in the last round (t=d-1): detections at t=d-1 and t=d."""
    lay = layout_d3
    d = lay.n_layers - 1
    dets = np.zeros((1, lay.det_index.max() + 1), dtype=bool)
    dets[0, lay.det_index[d - 1, 0]] = True
    dets[0, lay.det_index[d, 0]] = True

    with_final = temporal_filter(build_cube(dets, lay, include_final_layer=True))
    assert with_final.flat[0, 0] == 0                     # blip removed: data parity is clean

    without = temporal_filter(build_cube(dets, lay, include_final_layer=False))
    assert without.flat[0, 0] == 1                        # blip is the last slice: undetectable
    assert without.confidence[0, 0] == CONF_LATE          # ... but flagged as low confidence
