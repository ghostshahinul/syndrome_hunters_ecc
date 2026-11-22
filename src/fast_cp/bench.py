"""Benchmark the published Stim grid with common packed shots and isolated decoder processes."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import psutil

from src.circuits import detector_error_model, surface_code_circuit
from src.data import Samples
from src.fast_cp.worker import NAMES


def append_csv(path, rows):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        if new:
            writer.writeheader()
        writer.writerows(rows)


def prepare(work, d, p, cfg):
    circuit = surface_code_circuit(d, p, rounds=d, basis=cfg["basis"])
    t = time.perf_counter()
    dem = detector_error_model(circuit)
    dem_s = time.perf_counter() - t
    dem.to_file(work / "circuit.dem")
    kwargs = {"shots": cfg["shots"], "separate_observables": True}
    t = time.perf_counter()
    dets, obs = circuit.compile_detector_sampler(seed=cfg["seed"]).sample(
        **kwargs, bit_packed=True
    )
    packed_s = time.perf_counter() - t
    t = time.perf_counter()
    dense, labels = circuit.compile_detector_sampler(seed=cfg["seed"]).sample(**kwargs)
    dense_s = time.perf_counter() - t
    for bits, packed in ((dense, dets), (labels, obs)):
        np.testing.assert_array_equal(
            np.packbits(bits, axis=1, bitorder="little"), packed
        )
    checksum = Samples(dense, labels).checksum()
    np.save(work / "dets.npy", dets)
    np.save(work / "obs.npy", obs)
    return {
        "distance": d,
        "rounds": d,
        "p": p,
        "basis": cfg["basis"],
        "seed": cfg["seed"],
        "sample_sha256": checksum,
        "dem_sha256": hashlib.sha256(str(dem).encode()).hexdigest(),
        "circuit_sha256": hashlib.sha256(str(circuit).encode()).hexdigest(),
        "num_detectors": dem.num_detectors,
        "dem_s": dem_s,
        "sampling_packed_s": packed_s,
        "sampling_dense_s": dense_s,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7, 9, 11])
    ap.add_argument(
        "--ps", nargs="+", type=float, default=[0.001, 0.003, 0.005, 0.007, 0.01]
    )
    ap.add_argument("--shots", type=int, default=50000)
    ap.add_argument("--basis", choices=["x", "z"], default="z")
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--latency-shots", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--out", type=Path, default=Path("results/fast_correlated_pymatching")
    )
    a = ap.parse_args()
    if min(a.shots, a.batch_size, a.workers, a.repeats, a.latency_shots) < 1:
        ap.error("sample and worker counts must be positive")
    cfg = {k: v for k, v in vars(a).items() if k != "out"}
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    dest = a.out / stamp
    dest.mkdir(parents=True, exist_ok=False)
    work = Path("data/fast_cp") / stamp
    work.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        **cfg,
        variants=NAMES,
        python=sys.version,
        platform=platform.platform(),
        cpu=platform.processor(),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(),
        versions={
            n: version(n)
            for n in ("stim", "pymatching", "numpy", "psutil", "scipy", "matplotlib")
        },
        native_build=json.loads(Path("external/fast_cp_build/build.json").read_text()),
    )
    (dest / "config.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8"
    )
    (work / "config.json").write_text(json.dumps(cfg), encoding="utf8")
    print(dest, flush=True)
    point_index = 0
    for d in a.distances:
        for p in a.ps:
            point = prepare(work, d, p, cfg)
            append_csv(dest / "sampling.csv", [point])
            order = (
                NAMES[point_index % len(NAMES) :] + NAMES[: point_index % len(NAMES)]
            )
            point_index += 1
            rows = {}
            for name in order:
                subprocess.run(
                    [sys.executable, "-m", "src.fast_cp.worker", str(work), name],
                    check=True,
                )
                rows[name] = json.loads((work / (name + ".json")).read_text())
            reference = np.load(work / "correlated.npy")
            for name, row in rows.items():
                mismatch = int(
                    (np.load(work / (name + ".npy")) != reference).any(axis=1).sum()
                )
                row["different_from_correlated"] = mismatch
                row["speedup"] = rows["correlated"]["decode_s"] / row["decode_s"]
                trials = row.pop("trials_s")
                append_csv(
                    dest / "raw.csv",
                    [
                        {
                            "distance": d,
                            "rounds": d,
                            "p": p,
                            "basis": a.basis,
                            "seed": a.seed,
                            "sample_sha256": point["sample_sha256"],
                            "decoder": name,
                            "shots": a.shots,
                            "repeat": k + 1,
                            "decode_s": seconds,
                        }
                        for k, seconds in enumerate(trials)
                    ],
                )
                append_csv(dest / "summary.csv", [{**point, **row}])
                if name != "mwpm" and mismatch:
                    raise ValueError(
                        f"{name} disagrees with correlated on {mismatch} shots at d={d}, p={p}"
                    )
            print(
                f"d={d} p={p:g} | "
                + "  ".join(f"{n} {rows[n]['us_per_shot']:.2f}" for n in NAMES),
                flush=True,
            )


if __name__ == "__main__":
    main()
