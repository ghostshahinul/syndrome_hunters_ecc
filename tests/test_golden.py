"""Speedups must not change results: compare every engine with the saved golden outputs
(tests/golden/golden_outputs.npz, written by scripts/make_golden.py before Phase B)."""

from pathlib import Path

import numpy as np
import pytest

from scripts.make_golden import CASES, TIER_P, engines, run_engine
from src.decoders.cube3d.graph import detector_graph

GOLDEN = Path(__file__).parent / "golden" / "golden_outputs.npz"


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip("no golden file; run python -m scripts.make_golden")
    return np.load(GOLDEN)


@pytest.mark.parametrize("tier,d", CASES)
def test_outputs_match_golden(golden, tier, d):
    key = f"{tier}_{d}"
    n = int(golden[f"{key}__ndet"])
    dets = np.unpackbits(golden[f"{key}__dets"], axis=1)[:, :n].astype(np.uint8)
    g = detector_graph(tier, d, TIER_P[tier])
    for name, eng in engines(g):
        out = run_engine(name, eng, dets)
        for field, arr in out.items():
            np.testing.assert_array_equal(arr, golden[f"{key}__{name}__{field}"], err_msg=f"{key} {name} {field}")
