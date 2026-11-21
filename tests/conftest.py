import numpy as np
import pytest

from src.circuits import detector_error_model, surface_code_circuit
from src.decoders import DECODERS


@pytest.fixture(scope="session")
def small_circuit():
    return surface_code_circuit(distance=3, p=0.001)


@pytest.fixture(scope="session")
def small_dem(small_circuit):
    return detector_error_model(small_circuit)


def build_or_skip(cls, dem):
    """Instantiate a decoder; skip the test if it isn't implemented yet."""
    decoder = cls(dem)
    try:
        decoder.decode_batch(np.zeros((1, dem.num_detectors), dtype=bool))
    except NotImplementedError:
        pytest.skip(f"{cls.name} not implemented yet")
    return decoder


@pytest.fixture(params=sorted(DECODERS))
def decoder(request, small_dem):
    return build_or_skip(DECODERS[request.param], small_dem)
