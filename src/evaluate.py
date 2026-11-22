"""Benchmark each decoder on the identical Stim detector and observable arrays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
from pathlib import Path
from statistics import median

import numpy as np
from scipy.stats import binomtest

from src.circuits import detector_error_model, surface_code_circuit
from src.data import Samples, default_path, load, sample
from src.decoders import DECODERS, get_decoder
from src.decoders.ising import MATCHERS, IsingHybrid
from src.decoders.ising_table import TABLE_NAME, CompiledIsingHybrid
from src import ising_data

RESULTS_DIR = Path("results")
HYBRID_NAMES = (*MATCHERS, TABLE_NAME)
FIELDS = [
    "decoder", "distance", "rounds", "p", "basis", "shots", "seed", "sample_sha256",
    "num_errors", "logical_error_rate", "ler_stderr", "ler_low95", "ler_high95",
    "ref_only_errors", "decoder_only_errors", "paired_p",
    "init_s", "warmup_s", "decode_s", "us_per_shot", "single_p50_us", "single_p95_us",
    "py_peak_mem_mb", "batch_size", "repeats", "latency_shots",
    "source", "source_revision", "circuit_sha256", "dem_sha256", "noise_sha256",
    "corr_ref_only_errors", "corr_decoder_only_errors", "corr_paired_p",
    "model_load_s", "ai_s", "global_s", "shots_per_s", "events_before", "events_after",
    "density_before", "density_after", "density_factor", "events_removed_pct",
    "cpu", "device", "gpu", "gpu_peak_mib", "model", "model_revision", "checkpoint_sha256",
    "precision", "torch_version", "cuda_runtime", "gpu_total_mib", "gpu_free_before_model_mib", "prediction_sha256",
    "compile_device", "table_build_s", "table_entries", "table_bytes", "table_build_gpu_peak_mib", "table_lookup_engine",
]


def wilson(errors: int, shots: int) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval, including zero observed errors."""
    z = 1.959963984540054
    p = errors / shots
    denom = 1 + z * z / shots
    center = (p + z * z / (2 * shots)) / denom
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * shots)) / shots) / denom
    low = max(0.0, center - radius) if errors else 0.0
    high = min(1.0, center + radius) if errors < shots else 1.0
    return low, high


def _measure(name, dem, samples, batch_size, repeats, latency_shots, factory=None):
    shots, dets = samples.shots, samples.detection_events
    if min(shots, batch_size, repeats, latency_shots) < 1:
        raise ValueError("shots, batch_size, repeats and latency_shots must be positive")
    started = time.perf_counter()
    decoder = factory() if factory is not None else get_decoder(name)(dem)
    hybrid = isinstance(decoder, IsingHybrid)
    init_s = time.perf_counter() - started
    started = time.perf_counter()
    warm_batch = dets[:min(shots, batch_size)]
    decoder.decode_batch(warm_batch)
    decoder.decode(warm_batch[np.argmax(np.count_nonzero(warm_batch, axis=1))])
    warmup_s = time.perf_counter() - started

    predictions = np.empty_like(samples.observable_flips)
    trials = []
    residual_events = 0
    if hybrid:
        decoder.predecoder.reset_peak()
    for trial in range(repeats):
        duration = 0
        ai_s = global_s = 0.0
        for pos in range(0, shots, batch_size):
            batch = dets[pos:pos + batch_size]
            start = time.perf_counter_ns()
            out = decoder.decode_batch(batch)
            duration += time.perf_counter_ns() - start
            if hybrid:
                ai_s += decoder.last_ai_s
                global_s += decoder.last_global_s
                if trial == 0:
                    residual_events += decoder.residual_events()
            if trial == 0:
                if out.shape != (len(batch), dem.num_observables) or out.dtype != bool:
                    raise ValueError(f"{name} violated the prediction shape or dtype contract")
                predictions[pos:pos + len(batch)] = out
        trials.append((duration * 1e-9, ai_s, global_s))
    # Use the same middle timed pass(es) for all stage and total measurements.
    trials.sort()
    middle = trials[(repeats - 1) // 2:repeats // 2 + 1]
    decode_s, ai_s, global_s = (median(values) for values in zip(*middle))
    extra = dict(device="cpu", global_s=decode_s)
    before = int(np.count_nonzero(dets))
    if hybrid:
        extra.update(decoder.predecoder.info)
        if isinstance(decoder, CompiledIsingHybrid):
            extra.update(decoder.table_info)
        extra.update(ai_s=ai_s, global_s=global_s, gpu_peak_mib=decoder.predecoder.peak_mib(),
                     events_after=int(residual_events), density_after=residual_events / dets.size,
                     density_factor=before / residual_events if residual_events else None,
                     events_removed_pct=100 * (before - residual_events) / before if before else None)

    # Include Python dispatch in single-shot latency. Use the same evenly
    # spaced input shots for every decoder, and time memory in a separate pass.
    latencies = []
    for pos in np.linspace(0, shots - 1, min(shots, latency_shots), dtype=int):
        row = dets[pos]
        start = time.perf_counter_ns()
        decoder.decode(row)
        latencies.append((time.perf_counter_ns() - start) / 1000)
    tracemalloc.start()
    try:
        decoder.decode_batch(dets[:batch_size])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    wrong = (predictions != samples.observable_flips).any(axis=1)
    errors = int(wrong.sum())
    rate = errors / shots
    low, high = wilson(errors, shots)
    return {
        "decoder": decoder.name, "shots": shots, "num_errors": errors, "logical_error_rate": rate,
        "ler_stderr": math.sqrt(rate * (1 - rate) / shots), "ler_low95": low, "ler_high95": high,
        "init_s": init_s, "warmup_s": warmup_s, "decode_s": decode_s,
        "us_per_shot": 1e6 * decode_s / shots, "single_p50_us": float(np.median(latencies)),
        "single_p95_us": float(np.percentile(latencies, 95)), "py_peak_mem_mb": peak / 2**20,
        "batch_size": min(shots, batch_size), "repeats": repeats, "latency_shots": len(latencies),
        "shots_per_s": shots / decode_s, "events_before": before, "density_before": before / dets.size,
        "cpu": platform.processor() or platform.machine(),
        "prediction_sha256": hashlib.sha256(np.packbits(predictions, axis=1).tobytes()).hexdigest(), **extra,
    }, wrong


def evaluate_samples(decoder_name, dem, samples: Samples, batch_size=4096, repeats=3, latency_shots=256):
    return _measure(decoder_name, dem, samples, batch_size, repeats, latency_shots)[0]


def _paired(reference, candidate):
    ref_only = int((reference & ~candidate).sum())
    own_only = int((candidate & ~reference).sum())
    pairs = ref_only + own_only
    return ref_only, own_only, float(binomtest(own_only, pairs).pvalue) if pairs else 1.0


def benchmark(decoder_names, distances, ps, shots, basis="z", seed=0, *, rounds=None,
              batch_size=4096, repeats=3, latency_shots=256, data_dir=None, source="stim",
              ising_repo=ising_data.DEFAULT_REPO, checkpoint=ising_data.DEFAULT_CHECKPOINT,
              device="auto", precision="auto") -> list[dict]:
    rows = []
    names = [n if n in HYBRID_NAMES else get_decoder(n).name for n in decoder_names]
    if not names or source not in ("stim", "nvidia"):
        raise ValueError("Select at least one decoder and a supported circuit source")
    if any(n in HYBRID_NAMES for n in names):
        if source != "nvidia":
            raise ValueError("Ising requires source=nvidia with its original detector ordering and noise model")
        ising_data.verify_checkpoint(checkpoint)
    if len(names) != len(set(names)):
        raise ValueError("duplicate decoder in benchmark")
    point = 0
    for d in distances:
        r = d if rounds is None else rounds
        for p in ps:
            if source == "nvidia":
                exp = ising_data.experiment(d, r, p, basis, ising_repo)
                circuit, dem = exp.circuit, exp.dem
                samples = ising_data.dataset(exp, shots, seed, Path(data_dir or "data/ising"))
            else:
                circuit = surface_code_circuit(d, p, r, basis)
                dem = detector_error_model(circuit)
            if source == "stim" and data_dir is None:
                samples = sample(circuit, shots, seed)
            elif source == "stim":
                path = Path(data_dir) / default_path(d, p, r, basis, seed, shots).name
                if not path.exists():
                    path = Path(data_dir) / default_path(d, p, r, basis, seed).name
                samples = load(path)
                expected = dict(distance=d, p=p, rounds=r, basis=basis, seed=seed, shots=shots)
                if any(samples.meta.get(k) != v for k, v in expected.items()) or (
                    samples.detection_events.shape != (shots, dem.num_detectors)
                    or samples.observable_flips.shape != (shots, dem.num_observables)
                ):
                    raise ValueError(f"sample archive does not match this experiment: {path}")
            identity = dict(source=source, sample_sha256=samples.checksum(),
                            circuit_sha256=ising_data.fingerprint(circuit), dem_sha256=ising_data.fingerprint(dem))
            if source == "nvidia":
                identity.update(source_revision=ising_data.SOURCE_REV, noise_sha256=exp.noise_sha256)
            scores = {}
            order = names[point % len(names):] + names[:point % len(names)]
            point += 1
            for name in order:
                factory = None
                if name in HYBRID_NAMES:
                    cls = CompiledIsingHybrid if name == TABLE_NAME else IsingHybrid
                    factory = partial(cls.from_experiment, name, exp, ising_repo, checkpoint, device, precision)
                row, wrong = _measure(name, dem, samples, batch_size, repeats, latency_shots, factory)
                row.update(distance=d, rounds=r, p=p, basis=basis, seed=seed, **identity)
                scores[name] = row, wrong
                print(f"{name:>22}  d={d} p={p:g}  {row['num_errors']}/{shots} failures"
                      f"  {row['us_per_shot']:.3f} us/shot", flush=True)
            for name in names:
                row, wrong = scores[name]
                if "pymatching" in scores:
                    row["ref_only_errors"], row["decoder_only_errors"], row["paired_p"] = _paired(
                        scores["pymatching"][1], wrong)
                if "pymatching_correlated" in scores:
                    row["corr_ref_only_errors"], row["corr_decoder_only_errors"], row["corr_paired_p"] = _paired(
                        scores["pymatching_correlated"][1], wrong)
                rows.append(row)
    return rows


def write_results(rows: list[dict], directory: Path = RESULTS_DIR, config=None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    path = directory / f"bench_{now:%Y%m%dT%H%M%S%fZ}.csv"
    with path.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    metadata = {
        "created_utc": now.isoformat(), "config": config, "python": sys.version.split()[0],
        "platform": platform.platform(), "cpu": platform.processor() or platform.machine(),
        "packages": {n: version(n) for n in ("stim", "pymatching", "numpy", "scipy", "numba", "matplotlib")},
        "timing": "Median of repeated warmed serial batch passes. Sampling, scoring, setup and memory profiling excluded.",
        "single_call": "Dedicated decode(syndrome) API for FaultJoin and both PyMatching modes. Median and p95.",
        "memory": "Python-traced peak MiB on one warm batch. Native allocations may be invisible.",
        "hybrid_timing": "ai_s includes detector packing, both transfers, synchronized NVIDIA preprocessing, CNN and residual construction. global_s includes original-DEM matching and frame XOR. decode_s covers the complete warmed decoder call. model_load_s is part of init_s.",
        "density": "Event counts divided by shots times detectors. Factor is before/after, undefined if after is zero. Percent removed is undefined if before is zero.",
        "gpu_memory": "PyTorch peak allocated MiB on the selected CUDA device, reset after warm-up; includes resident model.",
        "rows": rows,
    }
    with path.with_suffix(".json").open("x", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
        f.write("\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decoders", nargs="+", default=list(DECODERS), choices=[*DECODERS, "experimental", *HYBRID_NAMES])
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.001, 0.003, 0.005, 0.007, 0.01])
    ap.add_argument("--shots", type=int, default=50_000)
    ap.add_argument("--basis", choices=("x", "z"), default="z")
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--batch-size", type=int, help="shots per call (default 8 with Ising, 4096 for classical)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--latency-shots", type=int, default=256)
    ap.add_argument("--data-dir", type=Path, help="Stim NPZ input directory or NVIDIA .dets cache")
    ap.add_argument("--source", choices=("stim", "nvidia"), default="stim")
    ap.add_argument("--ising-repo", type=Path, default=ising_data.DEFAULT_REPO)
    ap.add_argument("--checkpoint", type=Path, default=ising_data.DEFAULT_CHECKPOINT)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--precision", choices=("auto", "fp16", "fp32"), default="auto")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    a = ap.parse_args()
    if a.batch_size is None:
        a.batch_size = 8 if set(a.decoders).intersection(HYBRID_NAMES) else 4096
    if min(a.shots, a.batch_size, a.repeats, a.latency_shots) < 1:
        ap.error("shots, batch size, repeats and latency shots must be positive")
    rows = benchmark(a.decoders, a.distances, a.ps, a.shots, a.basis, a.seed, rounds=a.rounds,
                     batch_size=a.batch_size, repeats=a.repeats, latency_shots=a.latency_shots, data_dir=a.data_dir,
                     source=a.source, ising_repo=a.ising_repo, checkpoint=a.checkpoint, device=a.device, precision=a.precision)
    print(f"wrote {write_results(rows, a.out_dir, vars(a))}")


if __name__ == "__main__":
    main()
