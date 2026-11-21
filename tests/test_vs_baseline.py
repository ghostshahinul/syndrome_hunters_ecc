"""Our decoders should be about as accurate as the PyMatching reference.

Union-find is known to be slightly less accurate than MWPM, so the bound is loose:
LER(ours) <= 1.5 * LER(baseline) + 3 standard errors.
Every LER measured here is plotted to results/tests/<run>/ler_vs_baseline.png.
"""

import math
from functools import lru_cache

import numpy as np
import pytest

from src.circuits import detector_error_model, surface_code_circuit
from src.data import sample
from src.decoders import DECODERS
from src.decoders.baseline import PyMatchingDecoder

from conftest import build_or_skip

CANDIDATES = sorted(n for n in DECODERS if n != PyMatchingDecoder.name)
CASES = [(3, 0.005), (5, 0.005)]
SHOTS = 20_000


@lru_cache(maxsize=None)
def setup_case(distance, p):
    circuit = surface_code_circuit(distance, p)
    return detector_error_model(circuit), sample(circuit, shots=SHOTS, seed=123)


def ler_and_stderr(decoder, samples):
    wrong = np.any(decoder.decode_batch(samples.detection_events) != samples.observable_flips, axis=1)
    ler = float(wrong.mean())
    return ler, math.sqrt(max(ler, 1 / samples.shots) * (1 - ler) / samples.shots)


@lru_cache(maxsize=None)
def baseline_ler(distance, p):
    dem, s = setup_case(distance, p)
    return ler_and_stderr(PyMatchingDecoder(dem), s)


def record(report, name, distance, p, ler, err):
    report.add("ler_vs_baseline", name, f"d={distance}, p={p:g}", ler, err,
               ylabel=f"logical error rate ({SHOTS} shots)", log=True)


@pytest.mark.parametrize("distance,p", CASES)
def test_baseline_ler_sane(report, distance, p):
    ler, err = baseline_ler(distance, p)
    record(report, PyMatchingDecoder.name, distance, p, ler, err)
    assert 0 < ler < 0.5


@pytest.mark.parametrize("name", CANDIDATES)
@pytest.mark.parametrize("distance,p", CASES)
def test_ler_close_to_baseline(report, name, distance, p):
    dem, s = setup_case(distance, p)
    candidate = build_or_skip(DECODERS[name], dem)
    ler_base, err_base = baseline_ler(distance, p)
    ler_cand, err_cand = ler_and_stderr(candidate, s)
    record(report, name, distance, p, ler_cand, err_cand)
    assert ler_cand <= 1.5 * ler_base + 3 * err_base, (
        f"{name}: LER {ler_cand:.4f} vs baseline {ler_base:.4f}"
    )
