"""Seeded NVIDIA Stim memory experiments, strict offline artifacts and provenance."""

from __future__ import annotations

import argparse
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import stim

from src.circuits import detector_error_model, surface_code_circuit
from src.data import Samples

DEFAULT_REPO = Path("external/Ising-Decoding")
SOURCE_REV = "4e5ef40494d13842aeee6cdf735b4d3134286dfa"
MODEL_ID = "nvidia/Ising-Decoder-SurfaceCode-1-Fast"
MODEL_REV = "77b3a3f444612afe838fa0ece526b4c588541576"
MODEL_FILE = "ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors"
MODEL_SHA = "22db7ad25b584d7588493d6ef815381c8d1263cfb29f212391a591b285ad0a05"
DEFAULT_CHECKPOINT = Path("external/models") / MODEL_FILE


def fingerprint(obj) -> str:
    return hashlib.sha256(str(obj).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def activate(repo: Path) -> Path:
    """Import the audited upstream source only when the NVIDIA workflow is selected."""
    repo = repo.resolve()
    if not (repo / "code/qec/surface_code/stim_sample_io.py").is_file():
        raise FileNotFoundError(f"Missing NVIDIA Ising checkout at {repo}. See docs/ising.md.")
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--", "code", "conf"], text=True)
    if revision != SOURCE_REV or dirty.strip():
        raise ValueError(f"Expected clean NVIDIA inference source at {SOURCE_REV}, found {revision} at {repo}")
    code = repo / "code"
    for name in ("qec", "data", "evaluation", "export", "training", "model"):
        loaded = getattr(sys.modules.get(name), "__file__", None)
        if loaded and not Path(loaded).resolve().is_relative_to(code):
            raise RuntimeError(f"Conflicting upstream import {name!r} from {loaded}. Use the isolated interpreter.")
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    if os.name == "nt":
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        os.environ["PREDECODER_TORCH_COMPILE"] = "0"
    return repo


def verify_checkpoint(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fast R9 checkpoint missing at {path}. See docs/ising.md for authorized download.")
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != MODEL_SHA:
        raise ValueError(f"Fast R9 checkpoint SHA-256 mismatch at {path}")
    return path


@lru_cache(maxsize=2)
def _stim_module(name):
    """Load the upstream Stim-only files without importing its Torch mapping package."""
    import qec.noise_model
    path = Path(qec.noise_model.__file__).parent / "surface_code" / f"{name}.py"
    spec = spec_from_file_location(f"ising_stim_{name}", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class NvidiaExperiment:
    distance: int
    rounds: int
    p: float
    basis: str
    circuit: stim.Circuit
    dem: stim.DetectorErrorModel
    noise_params: dict
    noise_sha256: str

    def metadata(self, shots: int, seed: int) -> dict:
        return _stim_module("stim_sample_io").build_stim_sample_metadata(
            distance=self.distance, n_rounds=self.rounds, basis=self.basis, code_rotation="XV",
            num_detectors=self.circuit.num_detectors, num_observables=self.circuit.num_observables,
            num_shots=shots, p_error=self.p, noise_model_label="25-param",
            noise_model_params=self.noise_params, noise_model_sha256=self.noise_sha256,
            extra=dict(seed=seed, source_revision=SOURCE_REV, circuit_sha256=fingerprint(self.circuit),
                       dem_sha256=fingerprint(self.dem), stim_version=stim.__version__),
        )


def experiment(distance: int, rounds: int, p: float, basis: str, repo=DEFAULT_REPO) -> NvidiaExperiment:
    activate(Path(repo))
    from qec.noise_model import NoiseModel
    if distance < 3 or distance % 2 == 0 or rounds < 1 or basis.lower() not in ("x", "z"):
        raise ValueError("NVIDIA surface memory requires odd distance >=3, positive rounds and X/Z basis")
    noise = NoiseModel.from_single_p(p)
    q = noise.get_max_probability()
    memory = _stim_module("memory_circuit").MemoryCircuit(
        distance=distance, n_rounds=rounds, basis=basis.upper(), code_rotation="XV",
        idle_error=q, sqgate_error=q, tqgate_error=q, spam_error=(2.0 / 3.0) * q,
        noise_model=noise, add_boundary_detectors=True)
    memory.set_error_rates()
    circuit = memory.stim_circuit
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    if (dem.num_detectors, dem.num_observables) != (rounds * (distance**2 - 1), 1):
        raise ValueError("Unexpected detector dimensions for the pinned NVIDIA memory layout")
    return NvidiaExperiment(distance, rounds, p, basis.upper(), circuit, dem,
                           noise.canonical_parameters(), noise.sha256())


def read_artifacts(exp: NvidiaExperiment, directory: Path, shots: int, seed: int) -> Samples:
    io = _stim_module("stim_sample_io")
    sample_path, meta_path = io.resolve_stim_sample_paths(directory, exp.basis)
    expected = exp.metadata(shots, seed)
    actual = json.loads(meta_path.read_text(encoding="utf-8"))
    # The official loader allows legacy files without a noise fingerprint. Paired
    # benchmarks also require explicit generator, seed, circuit and DEM identities.
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Sample metadata mismatch for {key} in {meta_path}")
    arr, meta = io.read_stim_detector_samples(
        samples_path=sample_path, metadata_path=meta_path, distance=exp.distance, n_rounds=exp.rounds,
        basis=exp.basis, code_rotation="XV", num_detectors=exp.dem.num_detectors,
        num_observables=exp.dem.num_observables, p_error=exp.p, noise_model_sha256=exp.noise_sha256,
        noise_model_label="25-param", strict_noise=True)
    result = split_samples(exp, arr, shots, meta)
    if result.checksum() != meta.get("sample_sha256"):
        raise ValueError(f"Sample checksum mismatch in {sample_path}")
    return result


def split_samples(exp, bits, shots, metadata) -> Samples:
    nd, no = exp.dem.num_detectors, exp.dem.num_observables
    if bits.shape != (shots, nd + no) or bits.dtype not in (np.bool_, np.uint8):
        raise ValueError("Invalid detector/observable sample dimensions or dtype")
    if bits.dtype != np.bool_ and bits.max(initial=0) > 1:
        raise ValueError("Nonbinary detector/observable samples")
    return Samples(np.ascontiguousarray(bits[:, :nd], dtype=bool),
                   np.ascontiguousarray(bits[:, nd:], dtype=bool), metadata)


def dataset(exp: NvidiaExperiment, shots: int, seed: int, root: Path) -> Samples:
    """Generate via NVIDIA's measurement-to-detector recipe or verify the existing cache."""
    io = _stim_module("stim_sample_io")
    root = Path(root)
    directory = root / f"d{exp.distance}_r{exp.rounds}_p{exp.p}_{exp.basis}_s{seed}_n{shots}"
    if not directory.exists():
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root) as scratch:
            stage = Path(scratch) / "artifact"
            measurements = exp.circuit.compile_sampler(seed=seed).sample(shots=shots)
            bits = exp.circuit.compile_m2d_converter().convert(measurements=measurements, append_observables=True)
            result = split_samples(exp, bits, shots, exp.metadata(shots, seed))
            result.meta["sample_sha256"] = result.checksum()
            io.write_stim_detector_samples(path=stage / f"samples_{exp.basis}.dets", dets_and_obs=bits,
                                           num_detectors=exp.dem.num_detectors, num_observables=exp.dem.num_observables)
            io.write_metadata_json(stage / f"metadata_{exp.basis}.json", result.meta)
            try:
                stage.rename(directory)
            except OSError:
                if not directory.is_dir():
                    raise
    return read_artifacts(exp, directory, shots, seed)


def audit(exp: NvidiaExperiment) -> dict:
    legacy = surface_code_circuit(exp.distance, exp.p, exp.rounds, exp.basis.lower())
    legacy_dem = detector_error_model(legacy)
    def record(circuit, dem):
        detectors, observables, schedule = [], [], []
        for inst in circuit.without_noise().flattened():
            if inst.name == "DETECTOR":
                detectors.append([t.value for t in inst.targets_copy()])
            elif inst.name == "OBSERVABLE_INCLUDE":
                observables.append(str(inst))
            elif inst.name != "QUBIT_COORDS":
                schedule.append(str(inst))
        coords = circuit.get_detector_coordinates()
        return dict(detectors=circuit.num_detectors, observables=circuit.num_observables,
                    detector_coordinates_present=sum(bool(c) for c in coords.values()),
                    detector_coordinates_sha256=fingerprint(coords), detector_order_sha256=fingerprint(detectors),
                    first_detector_rec=detectors[0], observable_support=observables,
                    schedule_sha256=fingerprint(schedule), circuit_sha256=fingerprint(circuit),
                    dem_sha256=fingerprint(dem), dem_errors=dem.num_errors)
    return dict(distance=exp.distance, rounds=exp.rounds, p=exp.p, basis=exp.basis,
                stim=record(legacy, legacy_dem), nvidia=record(exp.circuit, exp.dem),
                noise_sha256=exp.noise_sha256, nvidia_noise=exp.noise_params)


def environment(repo, checkpoint):
    """Run small known-answer kernels to check CPU/CUDA support independently of model access."""
    activate(Path(repo))
    import torch
    from importlib.metadata import version
    devices = [("cpu", torch.float32)]
    if torch.cuda.is_available():
        devices.extend([("cuda", torch.float32), ("cuda", torch.float16)])
    probes = {}
    with torch.inference_mode():
        for device, dtype in devices:
            x = torch.ones((1, 4, 5, 5, 5), device=device, dtype=dtype)
            kernel = torch.ones((2, 4, 3, 3, 3), device=device, dtype=dtype)
            out = torch.nn.functional.conv3d(x, kernel)
            probes[f"{device}_{dtype}"] = bool(torch.all(out == 108).item())
    gpu = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu = dict(name=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                   total_mib=total/2**20, free_mib=free/2**20, compiled_arches=torch.cuda.get_arch_list())
    smi = None
    try:
        smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
                                       "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return dict(platform=platform.platform(), cpu=platform.processor(), python=platform.python_version(),
                packages={n: version(n) for n in ("torch", "stim", "pymatching", "numpy", "scipy", "numba",
                                                  "matplotlib", "hydra-core", "safetensors", "huggingface-hub")},
                torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(), gpu=gpu, nvidia_smi=smi,
                conv3d_known_answer=probes, source_revision=SOURCE_REV, model=MODEL_ID, model_revision=MODEL_REV,
                checkpoint_verified=bool(verify_checkpoint(checkpoint)) if checkpoint.is_file() else False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("audit", "download", "doctor"))
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--p", type=float, default=.005)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    a = ap.parse_args()
    if a.action == "download":
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=MODEL_ID, revision=MODEL_REV, filename=MODEL_FILE,
                               local_dir=DEFAULT_CHECKPOINT.parent)
        print(verify_checkpoint(Path(path)))
        return
    if a.action == "doctor":
        report = environment(a.repo, a.checkpoint)
    else:
        report = dict(source_revision=SOURCE_REV, comparisons=[
            audit(experiment(d, d, a.p, b, a.repo)) for d in a.distances for b in ("x", "z")])
    rendered = json.dumps(report, indent=2) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("x", encoding="utf-8") as f:
            f.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
