"""Instrument the original wrapper by bracketing its actual operations."""

import argparse
import csv
import inspect
from pathlib import Path
from time import perf_counter

import numpy as np
import pymatching

from src.circuits import detector_error_model, surface_code_circuit
from src.decoders.baseline import CorrelatedMatchingDecoder


def profile(distance, shots, seed, repeat):
    circuit = surface_code_circuit(distance, 0.005)
    start = perf_counter()
    dem = detector_error_model(circuit)
    after_dem = perf_counter()
    decoder = CorrelatedMatchingDecoder(dem)
    after_graph = perf_counter()
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, truth = sampler.sample(shots, separate_observables=True)
    after_sampling = perf_counter()
    decoder.decode_batch(dets[:256])
    warmup = perf_counter() - after_sampling
    pre = native = post = 0.0
    failures = 0
    for i in range(0, shots, 4096):
        t0 = perf_counter()
        batch = decoder._check_input(dets[i : i + 4096])
        t1 = perf_counter()
        raw = decoder.matching.decode_batch(batch, enable_correlations=True)
        t2 = perf_counter()
        predicted = raw.astype(bool)
        failures += np.count_nonzero((predicted != truth[i : i + 4096]).any(axis=1))
        t3 = perf_counter()
        pre += t1 - t0
        native += t2 - t1
        post += t3 - t2
    return {
        "distance": distance,
        "p": 0.005,
        "shots": shots,
        "seed": seed,
        "repeat": repeat,
        "dem_s": after_dem - start,
        "graph_s": after_graph - after_dem,
        "sampling_s": after_sampling - after_graph,
        "warmup_s": warmup,
        "preprocessing_s": pre,
        "decoding_s": native,
        "prediction_and_score_s": post,
        "failures": failures,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(
        pymatching.__version__,
        inspect.signature(pymatching.Matching.decode_batch),
        flush=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", newline="") as f:
        writer = None
        for d in (3, 7, 11):
            for rep in range(3):
                row = profile(d, 4096, 20261121, rep)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)
                print(d, rep, row["decoding_s"], flush=True)


if __name__ == "__main__":
    main()
