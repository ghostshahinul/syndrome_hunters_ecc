import json
from types import SimpleNamespace

import numpy as np
import pymatching
import pytest
import stim

from src.data import Samples
from src.circuits import detector_error_model, surface_code_circuit
from src.decoders import DECODERS
from src.decoders.ising import IsingHybrid, NvidiaPredecoder
from src.decoders import ising
from src.decoders.ising_table import CompiledIsingHybrid, active_detectors
from src.evaluate import _measure, _paired, benchmark
from src import ising_data as nv
from src.plots import plot_density


class KnownCorrection:
    """Test fixture removing D0 via the physical fault D0 L0."""
    info = {"device": "test"}

    def predecode(self, dets):
        residual = dets.copy()
        residual[:, 0] = False
        return residual, dets[:, :1].copy()

    def reset_peak(self):
        pass

    def peak_mib(self):
        return None


@pytest.mark.parametrize("name", ["ising_pymatching", "ising_pymatching_correlated"])
def test_logical_frame_residual_and_scoring(name):
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D1 L0")
    shots = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=bool)
    truth = np.array([[0], [1], [1], [1]], dtype=bool)
    original = shots.copy()
    make = lambda: IsingHybrid(name, dem, KnownCorrection())
    decoder = make()
    out = decoder.decode_batch(shots)
    assert out.dtype == bool and out.shape == (4, 1)
    np.testing.assert_array_equal(out[:, 0], [0, 1, 1, 0])
    np.testing.assert_array_equal(decoder.last_residual, [[0, 0], [0, 1], [0, 0], [0, 1]])
    np.testing.assert_array_equal(decoder.decode(shots[2]), [1])
    np.testing.assert_array_equal(shots, original)
    assert decoder.decode_batch(shots[:0]).shape == (0, 1)
    row, wrong = _measure(name, dem, Samples(shots, truth), 3, 2, 3, make)
    np.testing.assert_array_equal(wrong, [0, 0, 0, 1])
    assert row["num_errors"] == 1 and row["logical_error_rate"] == .25
    assert (row["events_before"], row["events_after"], row["density_factor"], row["events_removed_pct"]) == (4, 2, 2, 50)
    assert row["ai_s"] > 0 and row["global_s"] > 0 and row["shots_per_s"] > 0
    assert _paired(~wrong, wrong) == (3, 1, .625)


def test_reject_unsupported_source_and_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="source=nvidia"):
        benchmark(["ising_pymatching"], [3], [.005], 5)
    file = tmp_path / nv.MODEL_FILE
    with pytest.raises(FileNotFoundError):
        nv.verify_checkpoint(file)
    file.write_bytes(b"unrelated weights")
    with pytest.raises(ValueError, match="SHA-256"):
        nv.verify_checkpoint(file)


def test_bad_predecoder_dimensions_and_dtype():
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    backend = KnownCorrection()
    decoder = IsingHybrid("ising_pymatching", dem, backend)
    bits = np.zeros((3, 1), bool)
    backend.predecode = lambda d: (d[:, :0], d)
    with pytest.raises(ValueError, match="dimensions"):
        decoder.decode_batch(bits)
    backend.predecode = lambda d: (d, d.astype(np.uint8))
    with pytest.raises(ValueError, match="boolean"):
        decoder.decode_batch(bits)
    with pytest.raises(ValueError, match="shape"):
        decoder.decode_batch(np.zeros((3, 2), bool))


def test_exhaustive_table_preserves_logicals_density_and_support():
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D2 L0\ndetector D3")
    reference = IsingHybrid("ising_pymatching", dem, KnownCorrection())
    positions = active_detectors(dem)
    np.testing.assert_array_equal(positions, [0, 2])
    table = CompiledIsingHybrid(reference, positions)
    bits = np.zeros((4, 4), dtype=bool)
    bits[:, positions] = [[0, 0], [1, 1], [0, 1], [1, 0]]
    expected = reference.decode_batch(bits)
    np.testing.assert_array_equal(table.decode_batch(bits), expected)
    assert table.residual_events() == np.count_nonzero(reference.last_residual)
    np.testing.assert_array_equal(np.array([table.decode(row) for row in bits]), expected)
    np.testing.assert_array_equal(table.decode_batch(bits[::-1]), expected[::-1])
    assert table.decode_batch(bits[:0]).shape == (0, 1) and table.residual_events() == 0
    assert table.table_info["table_entries"] == 4
    bits[0, 1] = True
    with pytest.raises(ValueError, match="perfect matching"):
        table.decode_batch(bits)
    with pytest.raises(ValueError, match="independent"):
        active_detectors(stim.DetectorErrorModel("error(.1) D0 D1 L0"))
    with pytest.raises(ValueError, match="20 active"):
        active_detectors(stim.DetectorErrorModel("\n".join(f"error(.1) D{i} L0" for i in range(21))))


def test_format_split_preserves_label_boundary():
    exp = SimpleNamespace(dem=stim.DetectorErrorModel("error(0.1) D1 L0"))
    data = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    s = nv.split_samples(exp, data, 2, {})
    np.testing.assert_array_equal(s.observable_flips[:, 0], [1, 0])
    np.testing.assert_array_equal(s.detection_events, data[:, :2])
    for invalid in (data[:, :2], data[:1], data.astype(float), data + 1):
        with pytest.raises(ValueError):
            nv.split_samples(exp, invalid, 2, {})
    assert plot_density([{"p": .005, "density_after": None}]) is None


def test_six_decoder_benchmark_holds_out_observable_labels(monkeypatch):
    circuit = surface_code_circuit(3, .005, 3, "z")
    exp = SimpleNamespace(circuit=circuit, dem=detector_error_model(circuit), noise_sha256="fixture")
    dets, labels = circuit.compile_detector_sampler(seed=43).sample(101, separate_observables=True)
    shots = Samples(dets, labels.copy())
    monkeypatch.setattr(nv, "verify_checkpoint", lambda path: path)
    monkeypatch.setattr(nv, "experiment", lambda *args: exp)
    monkeypatch.setattr(nv, "dataset", lambda *args: shots)
    monkeypatch.setattr(ising, "NvidiaPredecoder", lambda *args: KnownCorrection())
    def run():
        return benchmark([*DECODERS, *ising.REFERENCE_HYBRIDS], [3], [.005], 101, source="nvidia",
                         batch_size=19, repeats=1, latency_shots=2)
    original = run()
    assert {r["sample_sha256"] for r in original} == {shots.checksum()}
    shots.observable_flips = ~labels
    changed = run()
    assert {r["sample_sha256"] for r in changed} == {shots.checksum()}
    for first, second in zip(original, changed):
        assert first["prediction_sha256"] == second["prediction_sha256"]
        assert first["num_errors"] + second["num_errors"] == 101


requires_repo = pytest.mark.skipif(not (nv.DEFAULT_REPO / ".git").exists(), reason="optional NVIDIA source checkout")


def optional_runtime():
    torch = pytest.importorskip("torch")
    for package in ("omegaconf", "hydra", "safetensors"):
        pytest.importorskip(package)
    return torch


@requires_repo
@pytest.mark.parametrize("basis", ["x", "z"])
def test_seeded_offline_format_and_three_way_pairing(tmp_path, basis):
    exp = nv.experiment(3, 3, .005, basis)
    original = nv.dataset(exp, 257, 109, tmp_path)
    directory, = tmp_path.iterdir()
    again = nv.read_artifacts(exp, directory, 257, 109)
    # Independent Stim measurement-to-detection conversion, with separate labels.
    raw = exp.circuit.compile_sampler(seed=109).sample(257)
    dets, obs = exp.circuit.compile_m2d_converter().convert(measurements=raw, separate_observables=True)
    np.testing.assert_array_equal(again.detection_events, dets)
    np.testing.assert_array_equal(again.observable_flips, obs)
    assert again.checksum() == original.checksum() == Samples(dets, obs).checksum()
    rows = benchmark(["pymatching", "pymatching_correlated", "fault_join"], [3], [.005], 257, basis,
                     seed=109, source="nvidia", data_dir=tmp_path, repeats=1, batch_size=64, latency_shots=2)
    assert all(r["sample_sha256"] == original.checksum() for r in rows)
    for row in rows:
        assert row["num_errors"] - rows[1]["num_errors"] == row["corr_decoder_only_errors"] - row["corr_ref_only_errors"]
    pairs = nv.audit(exp)
    assert pairs["nvidia"]["detectors"] == pairs["stim"]["detectors"] == 24
    for k in ("detector_order_sha256", "observable_support", "dem_sha256", "schedule_sha256"):
        assert pairs["nvidia"][k] != pairs["stim"][k]


@requires_repo
def test_offline_metadata_and_sample_corruption_fail_closed(tmp_path):
    exp = nv.experiment(3, 3, .003, "z")
    original = nv.dataset(exp, 17, 11, tmp_path)
    directory, = tmp_path.iterdir()
    meta_file = directory / "metadata_Z.json"
    for key in ("basis", "code_rotation", "source_revision", "seed", "n_rounds", "num_detectors",
                "num_observables", "num_shots", "p_error", "noise_model_sha256", "noise_model_params",
                "circuit_sha256", "dem_sha256", "append_observables", "sample_sha256"):
        broken = original.meta.copy()
        del broken[key]
        meta_file.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError):
            nv.read_artifacts(exp, directory, 17, 11)
    meta_file.write_text(json.dumps(original.meta), encoding="utf-8")
    with pytest.raises(ValueError, match="p_error"):
        nv.read_artifacts(nv.experiment(3, 3, .007, "z"), directory, 17, 11)
    bits = np.column_stack((original.detection_events, ~original.observable_flips))
    stim.write_shot_data_file(data=bits, path=directory / "samples_Z.dets", format="dets",
                              num_detectors=24, num_observables=1)
    with pytest.raises(ValueError, match="checksum"):
        nv.read_artifacts(exp, directory, 17, 11)


@requires_repo
def test_noise_sweep_reproduces_public_vector():
    exp = nv.experiment(3, 3, .003, "z")
    # Only the 25 named numeric entries are needed from the pinned public YAML.
    import re
    cfg = (nv.DEFAULT_REPO / "conf/config_public.yaml").read_text(encoding="utf-8")
    values = dict(re.findall(r"^\s+(p_(?:prep|meas|idle|cnot)_\w+):\s*([\d.eE+-]+)", cfg, re.M))
    assert len(values) == len(exp.noise_params) == 25
    assert exp.noise_params == pytest.approx({k: float(v) for k, v in values.items()})
    assert nv.experiment(3, 3, .005, "z").noise_sha256 != exp.noise_sha256


@requires_repo
def test_upstream_generator_builds_the_same_circuit_and_metadata(tmp_path, monkeypatch):
    exp = nv.experiment(3, 3, .005, "z")
    optional_runtime()
    pytest.importorskip("beliefmatching")
    from export import generate_test_data as official
    from qec.surface_code.stim_sample_io import read_stim_detector_samples
    constructed = []
    original = official.MemoryCircuit
    def capture(*args, **kwargs):
        memory = original(*args, **kwargs)
        constructed.append(memory)
        return memory
    monkeypatch.setattr(official, "MemoryCircuit", capture)
    official.generate_test_data(distance=3, n_rounds=3, p_error=.005, basis="Z", num_samples=19,
                                noise_model_params=exp.noise_params, output_dir=str(tmp_path),
                                write_stim_artifacts=True, write_cudaq_artifacts=False)
    assert constructed[0].stim_circuit == exp.circuit
    actual, metadata = read_stim_detector_samples(
        samples_path=tmp_path / "samples_Z.dets", metadata_path=tmp_path / "metadata_Z.json",
        distance=3, n_rounds=3, basis="Z", code_rotation="XV", num_detectors=24, num_observables=1,
        p_error=.005, noise_model_sha256=exp.noise_sha256, strict_noise=True)
    assert actual.shape == (19, 25)
    assert metadata["noise_model_params"] == exp.noise_params


@requires_repo
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_upstream_residual_and_frame_from_one_known_correction(basis, device, monkeypatch):
    nv.activate(nv.DEFAULT_REPO)
    torch = optional_runtime()
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from evaluation.logical_error_rate import _build_stab_maps
    from export import safetensors_utils
    # Model double for the correction convention only. Never used by the evaluator.
    class OneCorrection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(-10.0))

        def forward(self, grid):
            logits = torch.ones_like(grid) * self.bias
            logits[:, 0 if basis == "X" else 1, 0, 0, 0] = 10
            return logits

    exp = nv.experiment(3, 3, .005, basis)
    maps = _build_stab_maps(3, "XV")
    monkeypatch.setattr(ising, "verify_checkpoint", lambda path: path)
    monkeypatch.setattr(safetensors_utils, "load_safetensors",
                        lambda path, device: (OneCorrection().to(device), {"model_id": "1", "quant_format": "fp16"}))
    backend = NvidiaPredecoder(exp, device=device)
    # Probe every detector, including the terminal boundary, in its native index order.
    bits = np.vstack((np.zeros((1, 24), bool), np.eye(24, dtype=bool)))
    residual, frame = backend.predecode(bits)
    expected = bits.copy()
    # The published neural transform pads the last round of the complementary
    # check type with zeros. Its boundary convention is part of the reference.
    half = (exp.distance**2 - 1) // 2
    masked_block = 2 * exp.rounds - (2 if basis == "X" else 3)
    expected[:, masked_block * half:(masked_block + 1) * half] = False
    check = maps["Hx_i32" if basis == "X" else "Hz_i32"][:, 0].numpy().astype(bool)
    expected[:, :4] ^= check
    assert residual.shape == (25, exp.dem.num_detectors)
    np.testing.assert_array_equal(residual, expected)
    np.testing.assert_array_equal(residual[:, -half:], bits[:, -half:])
    assert frame.shape == (25, 1) and frame.all()
    # Arbitrary single-detector probes can violate a DEM component's even parity.
    # Use physically sampled syndromes for the global matching assertions.
    valid = exp.circuit.compile_detector_sampler(seed=51).sample(31)
    expected = valid.copy()
    expected[:, masked_block * half:(masked_block + 1) * half] = False
    expected[:, :half] ^= check
    prediction = IsingHybrid("ising_pymatching", exp.dem, backend).decode_batch(valid)
    reference = pymatching.Matching.from_detector_error_model(exp.dem).decode_batch(expected) ^ 1
    np.testing.assert_array_equal(prediction, reference)
    restored = IsingHybrid.from_experiment(ising.BOUNDARY_HYBRID, exp, device=device)
    before = valid.copy()
    restored.decode_batch(valid)
    expected = valid.copy()
    expected[:, :half] ^= check
    np.testing.assert_array_equal(restored.last_residual, expected)
    np.testing.assert_array_equal(before, valid)


@requires_repo
@pytest.mark.skipif(not nv.DEFAULT_CHECKPOINT.is_file(), reason="authorized Fast R9 checkpoint unavailable")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pretrained_fast_r9_against_official_frame_and_matcher(device):
    nv.activate(nv.DEFAULT_REPO)
    torch = optional_runtime()
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    exp = nv.experiment(9, 9, .005, "z")
    backend = NvidiaPredecoder(exp, device=device)
    dets = exp.circuit.compile_detector_sampler(seed=67).sample(8)
    with torch.inference_mode():
        upstream = backend.module(torch.tensor(dets.astype(np.uint8), device=device)).cpu().numpy()
    for name in ("ising_pymatching", "ising_pymatching_correlated"):
        correlated = name.endswith("_correlated")
        matcher = pymatching.Matching.from_detector_error_model(exp.dem, enable_correlations=correlated)
        reference = (upstream[:, :1] + matcher.decode_batch(upstream[:, 1:], enable_correlations=correlated)) % 2
        result = IsingHybrid(name, exp.dem, backend).decode_batch(dets)
        assert result.dtype == bool
        np.testing.assert_array_equal(result, reference)
    restored = IsingHybrid.from_experiment(ising.BOUNDARY_HYBRID, exp, device=device)
    residual = upstream[:, 1:].copy()
    # Z memory, d=T=9: the omitted final X-check block is detectors 600..639.
    residual[:, 600:640] ^= dets[:, 600:640]
    reference = (upstream[:, :1] + matcher.decode_batch(residual, enable_correlations=True)) % 2
    np.testing.assert_array_equal(restored.decode_batch(dets), reference)


@requires_repo
@pytest.mark.skipif(not nv.DEFAULT_CHECKPOINT.is_file(), reason="authorized Fast R9 checkpoint unavailable")
def test_full_nvidia_offline_runner_matches_exported_arrays(tmp_path, monkeypatch):
    nv.activate(nv.DEFAULT_REPO)
    torch = optional_runtime()
    from evaluation.logical_error_rate import run_inference_and_decode_pre_decoder_memory
    from omegaconf import OmegaConf
    exp = nv.experiment(9, 9, .005, "z")
    samples = nv.dataset(exp, 32, 867, tmp_path / "samples")
    artifact, = (tmp_path / "samples").iterdir()
    output = tmp_path / "official_output"
    for key, value in {"PREDECODER_STIM_SAMPLES_DIR": str(artifact), "PREDECODER_STIM_STRICT_NOISE": "1",
                       "PREDECODER_DECODE_MODE": "ising_decoding_pymatching", "PREDECODER_DECODE_OUTPUT_DIR": str(output),
                       "PREDECODER_INFERENCE_NUM_WORKERS": "0", "PREDECODER_TORCH_COMPILE": "0", "ONNX_WORKFLOW": "0"}.items():
        monkeypatch.setenv(key, value)
    cfg = OmegaConf.create(dict(code="surface", datapipe="memory", distance=9, n_rounds=9, enable_fp16=False,
                                data=dict(error_mode="circuit_level_surface_custom", code_rotation="XV",
                                          noise_model=exp.noise_params),
                                test=dict(num_samples=32, n_rounds=9, p_error=.005, meas_basis_test="Z",
                                          noise_model="train", latency_num_samples=0,
                                          dataloader=dict(batch_size=8, num_workers=0, persistent_workers=False))))
    backend = NvidiaPredecoder(exp, device="cpu")
    device = torch.device("cpu")
    dist = SimpleNamespace(rank=0, world_size=1, device=device)
    run_inference_and_decode_pre_decoder_memory(backend.module.model, device, dist, cfg)
    decoder = IsingHybrid("ising_pymatching", exp.dem, backend)
    predicted, residual = [], []
    for batch in np.split(samples.detection_events, 4):
        predicted.append(decoder.decode_batch(batch))
        residual.append(decoder.last_residual.copy())
    np.testing.assert_array_equal(np.load(output / "Z_observables.npy").reshape(-1, 1), samples.observable_flips)
    np.testing.assert_array_equal(np.load(output / "Z_predecoder_residual_detectors.npy"), np.vstack(residual))
    np.testing.assert_array_equal(np.load(output / "Z_ising_decoding_pymatching_predictions.npy").reshape(-1, 1),
                                  np.vstack(predicted))
