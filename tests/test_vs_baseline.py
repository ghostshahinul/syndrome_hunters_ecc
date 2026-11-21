"""Our decoders should be about as accurate as the PyMatching reference.

Union-find is known to be slightly less accurate than MWPM, so the bound is loose:
LER(ours) <= 1.5 * LER(baseline) + 3 standard errors.
"""

import math

import numpy as np
import pytest

from src.circuits import detector_error_model, surface_code_circuit
from src.data import sample
from src.decoders import DECODERS
from src.decoders.baseline import PyMatchingDecoder

CANDIDATES = sorted(n for n in DECODERS if n != PyMatchingDecoder.name)


@pytest.mark.parametrize("name", CANDIDATES)
@pytest.mark.parametrize("distance,p", [(3, 0.005), (5, 0.005)])
def test_ler_close_to_baseline(name, distance, p):
    circuit = surface_code_circuit(distance, p)
    dem = detector_error_model(circuit)
    candidate = DECODERS[name](dem)
    baseline = PyMatchingDecoder(dem)

    s = sample(circuit, shots=20_000, seed=123)
    ler = lambda dec: np.mean(np.any(dec.decode_batch(s.detection_events) != s.observable_flips, axis=1))
    ler_base, ler_cand = ler(baseline), ler(candidate)
    stderr = math.sqrt(max(ler_base, 1 / s.shots) / s.shots)
    assert ler_cand <= 1.5 * ler_base + 3 * stderr, (
        f"{name}: LER {ler_cand:.4f} vs baseline {ler_base:.4f}"
    )
