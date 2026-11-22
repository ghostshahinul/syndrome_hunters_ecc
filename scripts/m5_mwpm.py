"""Milestone 5 (B): three tiers (Cube3D-Local -> UnionFind3D -> MWPM3D) vs UnionFind3D alone.

Circuit-level noise, rounds = d, p in {0.1, 0.2, 0.5}%, d = 3/5/7/9, seed 0. For each point:
  - Hierarchical (fast path + UF, 1 numba thread) on every shot: predictions, time, and
    UF's largest cluster per shot;
  - MWPM3D on the shots with the largest clusters, one shot at a time, largest first, with
    each shot's time recorded (stops once `--budget` seconds of MWPM time are used);
  - for each threshold T in {d, 2d, 3d, 4d} whose routed set (largest cluster > T) was fully
    decoded: LER of the three-tier decoder and its time per shot
      = Hierarchical batch time / shots + (sum of MWPM times of the routed shots) / shots.
PyMatching runs on the same shots as the reference (LER, single-thread time, and agreement
with MWPM3D on the routed shots).

    python -m scripts.m5_mwpm [--budget 90]
Writes results/mwpm/mwpm_tier_<timestamp>.csv.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pymatching

from src.circuits import surface_code_circuit
from src.decoders.cube3d.graph import detector_graph
from src.decoders.cube3d.hierarchical import Hierarchical
from src.decoders.cube3d.mwpm import MWPM3D
from src.decoders.cube3d.union_find import UnionFind3D, bits_to_bool, set_threads
from src.figures import timestamp, write_rows
from src.machine import fingerprint
from src.operating_point import wilson_interval

OUT = Path("results") / "mwpm"
SHOTS = {0.001: 20_000, 0.002: 20_000, 0.005: 10_000}


def best(fn, n, repeats=2):
    t, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        t = min(t, time.perf_counter() - t0)
    return out, 1e6 * t / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=90.0, help="MWPM seconds per (p, d)")
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.001, 0.002, 0.005])
    args = ap.parse_args()
    fp = fingerprint()
    print(fp, flush=True)
    set_threads(1)
    rows = []
    for p in args.ps:
        for d in args.distances:
            N = SHOTS.get(p, 10_000)
            c = surface_code_circuit(d, p)
            dets, obs = c.compile_detector_sampler(seed=0).sample(N, separate_observables=True)
            dets = np.ascontiguousarray(dets, dtype=np.uint8)
            g = detector_graph("circuit", d, p)
            uf, h, mw = UnionFind3D(g), Hierarchical(g), MWPM3D(g)
            uf.run(dets[:64]); h.run(dets[:64])
            ufr, t_uf = best(lambda: uf.run(dets), N)
            hr, t_h = best(lambda: h.run(dets), N)
            m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
            pm, t_pm = best(lambda: m.decode_batch(dets).astype(bool), N)
            base = dict(d=d, p=p, rounds=d, shots=N, **fp, threads=1)

            def add(decoder, pred, us, **extra):
                k = int((pred != obs).any(axis=1).sum())
                lo, hi = wilson_interval(k, N)
                rows.append({"decoder": decoder, **base, "failures": k, "ler": k / N, "ci_low": lo,
                             "ci_high": hi, "us_per_shot": us, **extra})
                return rows[-1]

            add("PyMatching", pm, t_pm, backend="pymatching")
            add("UnionFind3D", ufr.predictions, t_uf, backend="numba")
            add("Hierarchical", hr.predictions, t_h, backend="numba")

            size = hr.max_cluster
            thresholds = [d, 2 * d, 3 * d, 4 * d]
            cand = np.flatnonzero(size > thresholds[0])
            cand = cand[np.argsort(-size[cand], kind="stable")]  # largest clusters first
            mw_bits = np.zeros(N, np.uint64)
            mw_time = np.zeros(N)
            done = np.zeros(N, bool)
            spent = 0.0
            for s in cand:
                t0 = time.perf_counter()
                mw_bits[s] = mw.decode_shot(dets[s])
                mw_time[s] = time.perf_counter() - t0
                done[s] = True
                spent += mw_time[s]
                if spent > args.budget:
                    break
            mw_pred = bits_to_bool(mw_bits, g.n_obs)
            agree = int((mw_pred[done] == pm[done]).all(axis=1).sum())
            for T in thresholds:
                routed = size > T
                if not done[routed].all():
                    print(f"p={p} d={d} T={T}: skipped ({routed.sum()} shots exceed the {args.budget:.0f}s MWPM budget)")
                    continue
                pred = hr.predictions.copy()
                pred[routed] = mw_pred[routed]
                us = t_h + 1e6 * mw_time[routed].sum() / N
                hw = (hr.predictions != obs).any(axis=1)
                tw = (pred != obs).any(axis=1)
                r = add("Hierarchical + MWPM", pred, us, backend="numba+python", threshold=T,
                        routed_fraction=float(routed.mean()), fixed=int((hw & ~tw).sum()),
                        broken=int((~hw & tw).sum()),
                        mwpm_us_per_routed_shot=1e6 * mw_time[routed].mean() if routed.any() else 0.0)
                print(f"p={p} d={d} T={T:>2}: routed {100 * r['routed_fraction']:.2f}% LER {100 * r['ler']:.4f}% "
                      f"[{100 * r['ci_low']:.4f},{100 * r['ci_high']:.4f}] fixed {r['fixed']} broken {r['broken']} "
                      f"| {us:.2f} us/shot (MWPM {r['mwpm_us_per_routed_shot']:.0f} us per routed shot)", flush=True)
            print(f"p={p} d={d} N={N}: PM {int((pm != obs).any(1).sum())} fails {t_pm:.2f} us | UF "
                  f"{int((ufr.predictions != obs).any(1).sum())} {t_uf:.2f} us | Hier {int((hr.predictions != obs).any(1).sum())} "
                  f"{t_h:.2f} us | MWPM3D==PyMatching on {agree}/{int(done.sum())} decoded shots", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = write_rows(rows, OUT / f"mwpm_tier_{timestamp()}.csv")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
