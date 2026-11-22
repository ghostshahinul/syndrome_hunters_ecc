import numpy as np
import pytest

from results_report import ResultsReport
from src.circuits import detector_error_model, surface_code_circuit
from src.decoders import DECODERS


def pytest_addoption(parser):
    parser.addoption("--no-plots", action="store_true",
                     help="don't write test plots to results/tests/")
    parser.addoption("--op-shots", type=int, default=50_000,
                     help="shots per batch for the accuracy/runtime operating-point test")
    parser.addoption("--op-batches", type=int, default=3,
                     help="independent batches per point for the operating-point test")


def pytest_configure(config):
    if not config.getoption("--no-plots"):
        config.pluginmanager.register(ResultsReport(config.rootpath), "results-report")


@pytest.fixture
def report(pytestconfig):
    """Record a metric for the post-run plots: report.add(plot, series, x, y, yerr=...).
    A no-op when plots are disabled."""
    plugin = pytestconfig.pluginmanager.get_plugin("results-report")
    if plugin is None:
        class _Null:
            def add(self, *_args, **_kwargs):
                pass

            def add_figure(self, *_args, **_kwargs):
                pass
        return _Null()
    return plugin


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
