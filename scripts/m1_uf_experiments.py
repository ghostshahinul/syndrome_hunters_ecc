"""Milestone 1: UF accuracy experiments (weight resolution S x growth order).

Circuit-level p = 0.5%, rounds = d, d = 3/5/7/9, 50k shots, seed 0 (same shots for every
config). Writes results/uf_experiments/uf_experiments_<timestamp>.csv.

    python scripts/m1_uf_experiments.py [--shots N]
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numba
import numpy as np

from src.circuits import surface_code_circuit
from src.decoders.cube3d.graph import detector_graph
from src.decoders.cube3d.union_find import UnionFind3D, UnionFind3DConfig, set_threads
from src.machine import fingerprint
from src.operating_point import wilson_interval

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "uf_experiments"
P = 0.005
CONFIGS = [(S, gr) for gr in ("all", "smallest") for S in (2, 4, 8)]


def timed(uf, dets, threads, repeats=2):
    set_threads(threads)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = uf.run(dets)
        best = min(best, time.perf_counter() - t0)
    return res, 1e6 * best / len(dets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=50_000)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
    args = ap.parse_args()
    max_threads = numba.config.NUMBA_NUM_THREADS
    fp = fingerprint()
    print(fp)
    rows = []
    for d in args.distances:
        c = surface_code_circuit(d, P)
        dets, obs = c.compile_detector_sampler(seed=0).sample(args.shots, separate_observables=True)
        g = detector_graph("circuit", d, P)
        for S, growth in CONFIGS:
            uf = UnionFind3D(g, UnionFind3DConfig(scale=S, growth=growth))
            uf.run(dets[:16])  # JIT warm-up
            res, t1 = timed(uf, dets, 1)
            _, tn = timed(uf, dets, max_threads)
            k = int((res.predictions != obs).any(axis=1).sum())
            lo, hi = wilson_interval(k, args.shots)
            row = {"d": d, "p": P, "rounds": d, "shots": args.shots, "scale": S, "growth": growth,
                   "failures": k, "ler": k / args.shots, "ci_low": lo, "ci_high": hi,
                   "ok_fraction": float(res.ok.mean()), "max_length": int(uf.tables.length.max()),
                   "us_per_shot_1thr": t1, f"us_per_shot_{max_threads}thr": tn,
                   "threads_max": max_threads, "backend": "numba", **fp}
            rows.append(row)
            print(f"d={d} S={S} growth={growth:8s} LER {100 * k / args.shots:.3f}% "
                  f"[{100 * lo:.3f},{100 * hi:.3f}] ok={res.ok.mean():.4f} "
                  f"1thr {t1:.2f} us  {max_threads}thr {tn:.2f} us", flush=True)
    set_threads(max_threads)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"uf_experiments_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
