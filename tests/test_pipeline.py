from pathlib import Path

import numpy as np
import pytest
import stim

from src.circuits import detector_error_model, surface_code_circuit
from src.data import default_path, load, sample, sample_surface_code, save
from src.decode import decode_file
from src.decoders.experimental import FaultJoinDecoder
from src.evaluate import benchmark, wilson


def test_blind_b8_entry_point_reads_stim_format(tmp_path: Path):
    circuit = surface_code_circuit(3, .007, rounds=2, basis="x")
    dem = detector_error_model(circuit)
    samples = sample(circuit, 129, seed=17)
    model, hits, predictions = [tmp_path / name for name in ("code.dem", "dets.b8", "obs.b8")]
    dem.to_file(model)
    stim.write_shot_data_file(data=samples.detection_events, path=hits, format="b8",
                              num_detectors=dem.num_detectors)
    assert decode_file(model, hits, predictions, batch_size=17) == samples.shots
    actual = stim.read_shot_data_file(path=predictions, format="b8", num_observables=dem.num_observables)
    np.testing.assert_array_equal(actual, FaultJoinDecoder(dem).decode_batch(samples.detection_events))
    with pytest.raises(FileExistsError):
        decode_file(model, hits, predictions)


def test_benchmark_scores_the_saved_shots_and_reports_paired_counts(tmp_path):
    shots = 1024
    s = sample_surface_code(3, .015, shots, seed=4)
    path = tmp_path / default_path(3, .015, 3, "z", 4, shots).name
    save(s, path)
    loaded = load(path)
    np.testing.assert_array_equal(loaded.observable_flips, s.observable_flips)
    np.testing.assert_array_equal(loaded.detection_events, s.detection_events)
    with pytest.raises(FileExistsError):
        save(s, path)
    candidate, ref = benchmark(["fault_join", "pymatching"], [3], [.015], shots,
                               seed=4, data_dir=tmp_path, repeats=1, batch_size=113, latency_shots=8)
    assert candidate["sample_sha256"] == ref["sample_sha256"]
    assert candidate["num_errors"] - ref["num_errors"] == candidate["decoder_only_errors"] - candidate["ref_only_errors"]
    assert ref["fault_join_only_errors"] == candidate["decoder_only_errors"]
    assert ref["alternative_only_errors"] == candidate["ref_only_errors"]
    assert ref["vs_fault_join_p"] == pytest.approx(candidate["paired_p"])
    assert candidate["ler_low95"] <= candidate["logical_error_rate"] <= candidate["ler_high95"]
    assert 0 < candidate["us_per_shot"] and 0 < candidate["single_p50_us"] <= candidate["single_p95_us"]
    lo, hi = wilson(0, shots)
    assert lo == 0 and 0 < hi < .01
