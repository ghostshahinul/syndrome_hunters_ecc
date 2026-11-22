"""Run one decoder in a fresh process for isolated timing and native memory accounting."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from statistics import median

import numpy as np
import psutil
import stim

from src.decoders.baseline import CorrelatedMatchingDecoder, PyMatchingDecoder
from src.decoders.fast_correlated import NativeCorrelated, PackedCorrelated
from src.evaluate import wilson

NAMES = (
    "mwpm",
    "correlated",
    "packed",
    "native_control",
    "fast_serial",
    "fast_parallel",
)


def peak_mib(proc):
    info = proc.memory_info()
    if hasattr(info, "peak_wset"):
        return info.peak_wset / 2**20
    import resource
    import sys

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        2**20 if sys.platform == "darwin" else 1024
    )


def make_decoder(name, dem, workers):
    if name == "mwpm":
        return PyMatchingDecoder(dem)
    if name == "correlated":
        return CorrelatedMatchingDecoder(dem)
    if name == "packed":
        return PackedCorrelated(dem)
    return NativeCorrelated(
        dem,
        fast=name not in ("native_control", "control_parallel"),
        workers=workers if name in ("fast_parallel", "control_parallel") else 1,
    )


def run(folder, name, config):
    proc = psutil.Process()
    before = proc.memory_info().rss / 2**20
    before_peak = peak_mib(proc)
    dem = stim.DetectorErrorModel.from_file(folder / "circuit.dem")
    dense = name in ("mwpm", "correlated")
    t = time.perf_counter()
    packed = np.load(folder / "dets.npy")
    if dense:
        dets = np.unpackbits(
            packed, axis=1, count=dem.num_detectors, bitorder="little"
        ).view(bool)
        del packed
    else:
        dets = packed
    input_s = time.perf_counter() - t
    t = time.perf_counter()
    decoder = make_decoder(name, dem, config["workers"])
    setup_s = time.perf_counter() - t
    batch = config["batch_size"]
    decode = decoder.decode_batch if dense else decoder.decode_packed
    t = time.perf_counter()
    decode(dets[:batch])
    warmup_s = time.perf_counter() - t
    warm_rss = proc.memory_info().rss / 2**20
    shots = len(dets)
    output = np.empty(
        (shots, dem.num_observables if dense else (dem.num_observables + 7) // 8),
        dtype=bool if dense else np.uint8,
    )
    timings = []
    copy_ns = 0
    for rep in range(config["repeats"]):
        elapsed = 0
        for pos in range(0, shots, batch):
            view = dets[pos : pos + batch]
            start = time.perf_counter_ns()
            predictions = decode(view)
            end = time.perf_counter_ns()
            elapsed += end - start
            if rep == 0:
                output[pos : pos + len(view)] = predictions
                copy_ns += time.perf_counter_ns() - end
        timings.append(elapsed * 1e-9)
    high = peak_mib(proc)
    latencies = []
    for index in np.linspace(
        0, shots - 1, min(config["latency_shots"], shots), dtype=int
    ):
        view = dets[index] if dense else dets[index : index + 1]
        start = time.perf_counter_ns()
        if dense:
            decoder.decode(view)
        else:
            decode(view)
        latencies.append((time.perf_counter_ns() - start) / 1000)
    t = time.perf_counter()
    packed_out = np.packbits(output, axis=1, bitorder="little") if dense else output
    truth = np.load(folder / "obs.npy")
    failures = int(np.count_nonzero(np.any(packed_out != truth, axis=1)))
    post_s = time.perf_counter() - t
    np.save(folder / (name + ".npy"), packed_out)
    if isinstance(decoder, NativeCorrelated):
        decoder.close()
    seconds = median(timings)
    low, high_ci = wilson(failures, shots)
    return {
        "decoder": name,
        "workers": config["workers"]
        if name in ("fast_parallel", "control_parallel")
        else 1,
        "shots": shots,
        "num_errors": failures,
        "logical_error_rate": failures / shots,
        "ler_low95": low,
        "ler_high95": high_ci,
        "decode_s": seconds,
        "us_per_shot": seconds * 1e6 / shots,
        "shots_per_s": shots / seconds,
        "single_p50_us": float(np.median(latencies)),
        "single_p95_us": float(np.percentile(latencies, 95)),
        "graph_s": setup_s,
        "warmup_s": warmup_s,
        "load_and_prepare_s": input_s,
        "collect_prediction_s": copy_ns / 1e9,
        "load_truth_pack_score_s": post_s,
        "input_mib": dets.nbytes / 2**20,
        "warm_rss_mib": warm_rss,
        "process_peak_mib": high,
        "warm_rss_delta_mib": warm_rss - before,
        "peak_growth_mib": high - before_peak,
        "prediction_sha256": hashlib.sha256(packed_out).hexdigest(),
        "trials_s": timings,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    ap.add_argument("name", choices=(*NAMES, "control_parallel"))
    a = ap.parse_args()
    config = json.loads((a.folder / "config.json").read_text())
    row = run(a.folder, a.name, config)
    (a.folder / (a.name + ".json")).write_text(json.dumps(row), encoding="utf-8")


if __name__ == "__main__":
    main()
