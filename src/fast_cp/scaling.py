"""Extend a finished benchmark with additional worker counts on its identical grid."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pymatching
import stim

from src.fast_cp.bench import append_csv, prepare


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--workers", type=int, nargs="+", default=[6, 12])
    a = ap.parse_args()
    cfg = json.loads((a.run / "config.json").read_text())
    with (a.run / "summary.csv").open(newline="") as stream:
        base = {
            (int(r["distance"]), float(r["p"])): r
            for r in csv.DictReader(stream)
            if r["decoder"] == "correlated"
        }
    if len(base) != len(cfg["ps"]) * len(cfg["distances"]):
        raise ValueError("finish the parent grid before measuring scaling")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    dest = a.run / ("scaling_" + stamp)
    dest.mkdir()
    work = Path("data/fast_cp") / stamp
    work.mkdir(parents=True)
    (dest / "config.json").write_text(
        json.dumps(
            {
                "parent": a.run.name,
                "workers": a.workers,
                "matched_control_workers": cfg["workers"],
                "seed": cfg["seed"],
            },
            indent=2,
        )
        + "\n"
    )
    for k, ((d, p), original) in enumerate(base.items()):
        point = prepare(work, d, p, cfg)
        assert point["sample_sha256"] == original["sample_sha256"]
        dem = stim.DetectorErrorModel.from_file(work / "circuit.dem")
        ref = pymatching.Matching.from_detector_error_model(
            dem, enable_correlations=True
        ).decode_batch(
            np.load(work / "dets.npy"),
            enable_correlations=True,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )
        assert hashlib.sha256(ref).hexdigest() == original["prediction_sha256"]
        counts = a.workers[k % len(a.workers) :] + a.workers[: k % len(a.workers)]
        strategies = [("fast_parallel", n, f"fast_{n}w") for n in counts]
        strategies.insert(
            k % (len(counts) + 1),
            ("control_parallel", cfg["workers"], "native_control_parallel"),
        )
        for backend, n, label in strategies:
            (work / "config.json").write_text(json.dumps({**cfg, "workers": n}))
            subprocess.run(
                [sys.executable, "-m", "src.fast_cp.worker", str(work), backend],
                check=True,
            )
            row = json.loads((work / (backend + ".json")).read_text())
            np.testing.assert_array_equal(ref, np.load(work / (backend + ".npy")))
            row.update(
                decoder=label,
                different_from_correlated=0,
                speedup=float(original["decode_s"]) / row["decode_s"],
            )
            trials = row.pop("trials_s")
            append_csv(
                dest / "raw.csv",
                [
                    {
                        "distance": d,
                        "rounds": d,
                        "p": p,
                        "basis": cfg["basis"],
                        "seed": cfg["seed"],
                        "sample_sha256": point["sample_sha256"],
                        "decoder": row["decoder"],
                        "shots": cfg["shots"],
                        "repeat": j + 1,
                        "decode_s": s,
                    }
                    for j, s in enumerate(trials)
                ],
            )
            append_csv(dest / "summary.csv", [{**point, **row}])
        print(f"Verified additional worker counts at d={d}, p={p:g}", flush=True)


if __name__ == "__main__":
    main()
