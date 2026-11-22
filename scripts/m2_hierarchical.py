"""Milestone 2: Hierarchical (Cube3D-Local + UnionFind3D) vs UnionFind3D alone.

Circuit-level noise, rounds = d, p in {0.1, 0.2, 0.5}%, d = 3/5/7/9, the same sampled
shots (seed 0) for every decoder at a (d, p). For each gate and K it records the LER with
Wilson CIs, the fast-path fraction, the fast path's failures on the shots it accepted vs
UF's failures on those same shots, and 1-thread / max-thread µs/shot. PyMatching
(single-thread) is timed in the same run as the reference.

    python -m scripts.m2_hierarchical [--shots N]
Writes results/hierarchical/hierarchical_<timestamp>.csv.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numba
import numpy as np
import pymatching

from src.circuits import surface_code_circuit
from src.decoders.cube3d.graph import detector_graph
from src.decoders.cube3d.hierarchical import Hierarchical, HierarchicalConfig
from src.decoders.cube3d.union_find import UnionFind3D, set_threads
from src.machine import fingerprint
from src.operating_point import wilson_interval

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "hierarchical"
PS = (0.001, 0.002, 0.005)
DS = (3, 5, 7, 9)
K_SWEEP = tuple(range(2, 11))


def best_time(fn, shots, repeats=2):
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, 1e6 * best / shots


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--distances", type=int, nargs="+", default=list(DS))
    ap.add_argument("--ps", type=float, nargs="+", default=list(PS))
    args = ap.parse_args()
    N = args.shots
    tmax = numba.config.NUMBA_NUM_THREADS
    fp = fingerprint()
    print(fp, flush=True)
    rows = []

    def add(**kw):
        k = kw["failures"]
        lo, hi = wilson_interval(k, N)
        rows.append({"d": kw.pop("d"), "p": kw.pop("p"), "rounds": kw["rounds"], "shots": N,
                     **kw, "ler": k / N, "ci_low": lo, "ci_high": hi, "threads_max": tmax, **fp})

    for p in args.ps:
        for d in args.distances:
            c = surface_code_circuit(d, p)
            dets, obs = c.compile_detector_sampler(seed=0).sample(N, separate_observables=True)
            dets_u8 = np.ascontiguousarray(dets, dtype=np.uint8)
            g = detector_graph("circuit", d, p)
            uf = UnionFind3D(g)
            h = Hierarchical(g)
            uf.run(dets_u8[:32])
            h.run(dets_u8[:32], gate="whole_shot")
            h.run(dets_u8[:32], gate="partial")

            # UF alone
            set_threads(1)
            ufr, uf_t1 = best_time(lambda: uf.run(dets_u8), N)
            set_threads(tmax)
            _, uf_tn = best_time(lambda: uf.run(dets_u8), N)
            uf_fail = (ufr.predictions != obs).any(axis=1)
            add(d=d, p=p, rounds=d, decoder="UnionFind3D", gate="", K="", failures=int(uf_fail.sum()),
                us_1thr=uf_t1, us_maxthr=uf_tn, fast_fraction="", fast_fraction_nontrivial="",
                cleared_fraction="", accepted_nontrivial="", fast_fail_on_accepted="",
                uf_fail_on_accepted="", backend="numba")

            # PyMatching (reference baseline, single thread)
            m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
            pm, pm_t = best_time(lambda: m.decode_batch(dets_u8), N)
            add(d=d, p=p, rounds=d, decoder="PyMatching", gate="", K="",
                failures=int((pm.astype(bool) != obs).any(axis=1).sum()), us_1thr=pm_t, us_maxthr="",
                fast_fraction="", fast_fraction_nontrivial="", cleared_fraction="",
                accepted_nontrivial="", fast_fail_on_accepted="", uf_fail_on_accepted="",
                backend="pymatching")

            configs = [("whole_shot", 6, True), ("partial", None, True)]
            configs += [(gate, K, False) for gate in ("whole_shot", "partial") for K in K_SWEEP]
            for gate, K, main_cfg in configs:
                run = lambda: h.run(dets_u8, K=K, gate=gate)
                set_threads(1)
                res, t1 = best_time(run, N, repeats=2 if main_cfg else 1)
                set_threads(tmax)
                _, tn = best_time(run, N, repeats=2 if main_cfg else 1)
                fail = (res.predictions != obs).any(axis=1)
                acc = res.fast & (res.lit_count > 0)
                add(d=d, p=p, rounds=d, decoder="Hierarchical", gate=gate,
                    K="inf" if K is None else K, failures=int(fail.sum()), us_1thr=t1, us_maxthr=tn,
                    fast_fraction=res.fast_fraction, fast_fraction_nontrivial=res.fast_fraction_nontrivial,
                    cleared_fraction=res.cleared_fraction, accepted_nontrivial=int(acc.sum()),
                    fast_fail_on_accepted=int(fail[acc].sum()), uf_fail_on_accepted=int(uf_fail[acc].sum()),
                    backend="numba", main=main_cfg)
                if main_cfg:
                    r = rows[-1]
                    print(f"p={p} d={d} {gate:10s} K={r['K']:>3}: LER {100 * r['ler']:.4f}% "
                          f"[{100 * r['ci_low']:.4f},{100 * r['ci_high']:.4f}] vs UF {100 * uf_fail.mean():.4f}% "
                          f"| fast {100 * res.fast_fraction:.1f}% (nontriv {100 * res.fast_fraction_nontrivial:.1f}%) "
                          f"cleared {100 * res.cleared_fraction:.1f}% | on accepted: fast {r['fast_fail_on_accepted']} "
                          f"vs UF {r['uf_fail_on_accepted']} of {r['accepted_nontrivial']} | "
                          f"1thr {t1:.2f} vs UF {uf_t1:.2f} us | {tmax}thr {tn:.2f} vs UF {uf_tn:.2f} us "
                          f"| PM {pm_t:.2f} us", flush=True)
    set_threads(tmax)
    for r in rows:
        r.setdefault("main", "")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"hierarchical_{datetime.now():%Y%m%d_%H%M%S}.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
