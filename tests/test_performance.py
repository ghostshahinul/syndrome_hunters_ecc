"""Execution-speed measurements, plotted to results/tests/<run>/.

  stim_runtime.png      seconds per Stim stage (circuit, DEM, compile, sample) vs distance
  stim_sample_rate.png  Stim sampling throughput (µs per shot) vs distance
  decode_speed.png      µs per shot for every implemented decoder vs distance

Timings use the best of REPEATS runs to reduce noise. Assertions are only loose sanity
bounds; the point of these tests is the recorded data.
"""

import time

import numpy as np
import pytest

from src.circuits import detector_error_model, surface_code_circuit
from src.decoders import DECODERS

from conftest import build_or_skip

DISTANCES = [3, 5, 7]
P = 0.005
SHOTS = 10_000
REPEATS = 3


def best_time(fn, repeats=REPEATS):
    """(min seconds over `repeats` runs, result of the last run)."""
    best, result = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best, result


@pytest.mark.parametrize("distance", DISTANCES)
def test_stim_runtime(report, distance):
    t_circuit, circuit = best_time(lambda: surface_code_circuit(distance, P))
    t_dem, dem = best_time(lambda: detector_error_model(circuit))
    t_compile, sampler = best_time(lambda: circuit.compile_detector_sampler(seed=0))
    t_sample, (dets, obs) = best_time(lambda: sampler.sample(SHOTS, separate_observables=True))

    x = f"d={distance}"
    for stage, t in [("circuit", t_circuit), ("detector error model", t_dem),
                     ("compile sampler", t_compile), (f"sample {SHOTS} shots", t_sample)]:
        report.add("stim_runtime", stage, x, t, ylabel="seconds (best of 3)", log=True)
    report.add("stim_sample_rate", "stim", x, 1e6 * t_sample / SHOTS, ylabel="µs per shot")

    assert dets.shape == (SHOTS, dem.num_detectors)
    assert obs.shape == (SHOTS, dem.num_observables)


@pytest.mark.parametrize("name", sorted(DECODERS))
@pytest.mark.parametrize("distance", DISTANCES)
def test_decode_speed(report, name, distance):
    circuit = surface_code_circuit(distance, P)
    dem = detector_error_model(circuit)
    decoder = build_or_skip(DECODERS[name], dem)
    dets, _ = circuit.compile_detector_sampler(seed=0).sample(SHOTS, separate_observables=True)

    t_init, _ = best_time(lambda: DECODERS[name](dem))
    t_decode, pred = best_time(lambda: decoder.decode_batch(dets))

    x = f"d={distance}"
    report.add("decode_speed", name, x, 1e6 * t_decode / SHOTS,
               ylabel=f"µs per shot (p={P:g}, best of 3)", log=True)
    report.add("decoder_init_time", name, x, t_init, ylabel="seconds to build decoder", log=True)

    assert pred.shape == (SHOTS, dem.num_observables)
    assert np.isfinite(t_decode)
