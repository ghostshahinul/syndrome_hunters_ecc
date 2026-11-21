"""Benchmark decoders: logical error rate, latency and memory.

CLI:  python -m src.evaluate --decoders pymatching --distances 3 5 7 --ps 0.001 0.005 --shots 100000
Appends nothing and overwrites nothing: every run writes a new results/bench_<timestamp>.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np

from src.circuits import detector_error_model, surface_code_circuit
from src.data import Samples, sample
from src.decoders import DECODERS, get_decoder

RESULTS_DIR = Path("results")
FIELDS = [
    "decoder", "distance", "rounds", "p", "basis", "shots", "seed",
    "num_errors", "logical_error_rate", "ler_stderr",
    "init_s", "decode_s", "us_per_shot", "py_peak_mem_mb",
]


def evaluate_samples(decoder_name: str, dem, samples: Samples) -> dict:
    """Decode pre-sampled data and score it. Memory is Python-heap peak (tracemalloc),
    so allocations inside native extensions such as PyMatching are not counted."""
    cls = get_decoder(decoder_name)

    t0 = time.perf_counter()
    decoder = cls(dem)
    init_s = time.perf_counter() - t0

    tracemalloc.start()
    t0 = time.perf_counter()
    predictions = decoder.decode_batch(samples.detection_events)
    decode_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    wrong = np.any(predictions != samples.observable_flips, axis=1)
    n_err, shots = int(wrong.sum()), samples.shots
    ler = n_err / shots
    return {
        "decoder": decoder_name,
        "shots": shots,
        "num_errors": n_err,
        "logical_error_rate": ler,
        "ler_stderr": math.sqrt(ler * (1 - ler) / shots),
        "init_s": init_s,
        "decode_s": decode_s,
        "us_per_shot": 1e6 * decode_s / shots,
        "py_peak_mem_mb": peak / 2**20,
    }


def benchmark(decoder_names, distances, ps, shots, basis="z", seed=0) -> list[dict]:
    rows = []
    for d in distances:
        for p in ps:
            circuit = surface_code_circuit(d, p, basis=basis)
            dem = detector_error_model(circuit)
            samples = sample(circuit, shots, seed)
            for name in decoder_names:
                try:
                    row = evaluate_samples(name, dem, samples)
                except NotImplementedError as e:
                    print(f"skip  {name}: {e}")
                    continue
                row.update(distance=d, rounds=d, p=p, basis=basis, seed=seed)
                rows.append(row)
                print(
                    f"{name:>12}  d={d:<2} p={p:<7g} LER={row['logical_error_rate']:.3e}"
                    f" ± {row['ler_stderr']:.1e}  {row['us_per_shot']:.2f} µs/shot"
                )
    return rows


def write_results(rows: list[dict], directory: Path = RESULTS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"bench_{datetime.now():%Y%m%d-%H%M%S}.csv"
    with open(path, "x", newline="") as f:  # "x": fail rather than overwrite
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoders", nargs="+", default=list(DECODERS))
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.001, 0.003, 0.005, 0.007, 0.01])
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--basis", default="z")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = benchmark(args.decoders, args.distances, args.ps, args.shots, args.basis, args.seed)
    if rows:
        print(f"wrote {write_results(rows)}")


if __name__ == "__main__":
    main()
