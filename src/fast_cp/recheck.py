"""Repeat a complete point when a timing exceeds a larger circuit/noise by over 5x."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    from src.fast_cp.report import load

    rows = load(args.run)
    points = set()
    for r in rows:
        larger = [
            v
            for v in rows
            if v["decoder"] == r["decoder"]
            and (
                (v["distance"] == r["distance"] and v["p"] > r["p"])
                or (v["distance"] > r["distance"] and v["p"] == r["p"])
            )
        ]
        if larger and r["us_per_shot"] > 5 * min(v["us_per_shot"] for v in larger):
            points.add((r["distance"], r["p"]))
    if not points:
        print("No points trigger the timing replication rule")
        return
    import json

    cfg = json.loads((args.run / "config.json").read_text())
    extra = sorted(
        {
            r["workers"]
            for r in rows
            if r["decoder"].startswith("fast_")
            and r["workers"] not in (1, cfg["workers"])
        }
    )
    for d, p in sorted(points):
        folder = args.run / "replication" / f"d{d}_p{p:g}"
        if folder.exists():
            raise FileExistsError(folder)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.fast_cp.bench",
                "--out",
                str(folder),
                "--distances",
                str(d),
                "--ps",
                str(p),
                "--shots",
                str(cfg["shots"]),
                "--repeats",
                str(cfg["repeats"]),
                "--seed",
                str(cfg["seed"]),
                "--basis",
                cfg["basis"],
                "--workers",
                str(cfg["workers"]),
                "--latency-shots",
                str(cfg["latency_shots"]),
                "--batch-size",
                str(cfg["batch_size"]),
            ],
            check=True,
        )
        (child,) = folder.glob("20*")
        if extra:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.fast_cp.scaling",
                    str(child),
                    "--workers",
                    *map(str, extra),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
