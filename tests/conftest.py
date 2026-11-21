import pytest

from src.circuits import detector_error_model, surface_code_circuit
from src.decoders import DECODERS


@pytest.fixture(scope="session")
def small_circuit():
    return surface_code_circuit(distance=3, p=0.001)


@pytest.fixture(scope="session")
def small_dem(small_circuit):
    return detector_error_model(small_circuit)


@pytest.fixture(params=sorted(DECODERS))
def decoder(request, small_dem):
    return DECODERS[request.param](small_dem)
