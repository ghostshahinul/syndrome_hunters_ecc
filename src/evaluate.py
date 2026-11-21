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
from importlib.metadata import version
from pathlib import Path
from statistics import median

import numpy as np
from scipy.stats import binomtest

from src.circuits import detector_error_model, surface_code_circuit
from src.data import Samples, default_path, load, sample
from src.decoders import DECODERS, get_decoder

RESULTS_DIR = Path("results")
FIELDS = [
    "decoder", "distance", "rounds", "p", "basis", "shots", "seed", "sample_sha256",
    "num_errors", "logical_error_rate", "ler_stderr", "ler_low95", "ler_high95",
    "ref_only_errors", "decoder_only_errors", "paired_p",
    "init_s", "warmup_s", "decode_s", "us_per_shot", "single_p50_us", "single_p95_us",
    "py_peak_mem_mb", "batch_size", "repeats", "latency_shots",
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


def _measure(name, dem, samples, batch_size, repeats, latency_shots):
    shots, dets = samples.shots, samples.detection_events
    if min(shots, batch_size, repeats, latency_shots) < 1:
        raise ValueError("shots, batch_size, repeats and latency_shots must be positive")
    started = time.perf_counter()
    decoder = get_decoder(name)(dem)
    init_s = time.perf_counter() - started
    started = time.perf_counter()
    warm_batch = dets[:min(shots, batch_size)]
    decoder.decode_batch(warm_batch)
    decoder.decode(warm_batch[np.argmax(np.count_nonzero(warm_batch, axis=1))])
    warmup_s = time.perf_counter() - started

    predictions = np.empty_like(samples.observable_flips)
    trials = []
    for trial in range(repeats):
        duration = 0
        for pos in range(0, shots, batch_size):
            batch = dets[pos:pos + batch_size]
            start = time.perf_counter_ns()
            out = decoder.decode_batch(batch)
            duration += time.perf_counter_ns() - start
            if trial == 0:
                if out.shape != (len(batch), dem.num_observables) or out.dtype != bool:
                    raise ValueError(f"{name} violated the prediction shape or dtype contract")
                predictions[pos:pos + len(batch)] = out
        trials.append(duration * 1e-9)
    decode_s = median(trials)

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
    }, wrong


def evaluate_samples(decoder_name, dem, samples: Samples, batch_size=4096, repeats=3, latency_shots=256):
    return _measure(decoder_name, dem, samples, batch_size, repeats, latency_shots)[0]


def benchmark(decoder_names, distances, ps, shots, basis="z", seed=0, *, rounds=None,
              batch_size=4096, repeats=3, latency_shots=256, data_dir=None) -> list[dict]:
    rows = []
    names = [get_decoder(n).name for n in decoder_names]
    if len(names) != len(set(names)):
        raise ValueError("duplicate decoder in benchmark")
    point = 0
    for d in distances:
        r = d if rounds is None else rounds
        for p in ps:
            circuit = surface_code_circuit(d, p, r, basis)
            dem = detector_error_model(circuit)
            if data_dir is None:
                samples = sample(circuit, shots, seed)
            else:
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
            digest = hashlib.sha256()
            for array in (samples.detection_events, samples.observable_flips):
                digest.update(np.packbits(array, axis=1).tobytes())
            scores = {}
            order = names[point % len(names):] + names[:point % len(names)]
            point += 1
            for name in order:
                row, wrong = _measure(name, dem, samples, batch_size, repeats, latency_shots)
                row.update(distance=d, rounds=r, p=p, basis=basis, seed=seed, sample_sha256=digest.hexdigest())
                scores[name] = row, wrong
                print(f"{name:>22}  d={d} p={p:g}  {row['num_errors']}/{shots} failures"
                      f"  {row['us_per_shot']:.3f} us/shot", flush=True)
            for name in names:
                row, wrong = scores[name]
                if "pymatching" in scores:
                    ref = scores["pymatching"][1]
                    ref_only = int((ref & ~wrong).sum())
                    own_only = int((wrong & ~ref).sum())
                    discordant = ref_only + own_only
                    row.update(ref_only_errors=ref_only, decoder_only_errors=own_only,
                               paired_p=binomtest(own_only, discordant).pvalue if discordant else 1.0)
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
    }
    with path.with_suffix(".json").open("x", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
        f.write("\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decoders", nargs="+", default=list(DECODERS), choices=[*DECODERS, "experimental"])
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.001, 0.003, 0.005, 0.007, 0.01])
    ap.add_argument("--shots", type=int, default=50_000)
    ap.add_argument("--basis", choices=("x", "z"), default="z")
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--latency-shots", type=int, default=256)
    ap.add_argument("--data-dir", type=Path, help="read previously sampled NPZ archives from this directory")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    a = ap.parse_args()
    if min(a.shots, a.batch_size, a.repeats, a.latency_shots) < 1:
        ap.error("shots, batch size, repeats and latency shots must be positive")
    rows = benchmark(a.decoders, a.distances, a.ps, a.shots, a.basis, a.seed, rounds=a.rounds,
                     batch_size=a.batch_size, repeats=a.repeats, latency_shots=a.latency_shots, data_dir=a.data_dir)
    print(f"wrote {write_results(rows, a.out_dir, vars(a))}")


if __name__ == "__main__":
    main()
