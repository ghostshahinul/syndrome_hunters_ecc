"""Item 2: ensemble UF, LER vs M and µs/shot vs threads.

Circuit-level p = 0.5%, rounds = d, d = 3/5/7/9, 50,000 shots, seed 0 (same shots for all).
  accuracy: UF3D correlated ensemble for M in {1, 4, 8, 12}, plain-UF ensemble for M in {1, 8},
            PyMatching and Correlated PyMatching. All single-thread (fair speed comparison).
  threads:  d = 7/9, correlated ensemble M = 8, 10,000 shots, threads 1/2/4/8/12, in both
            layouts: parallel over variants (cores used per shot = min(M, threads)) and
            parallel over shots (1 core per shot).
Writes results/ensemble/ensemble_<ts>.csv and ensemble_threads_<ts>.csv, then the figure
results/plots/ensemble_vs_M_<ts>.png (+ _latest).

    python -m scripts.item2_ensemble
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pymatching

from src.circuits import surface_code_circuit
from src.decoders.cube3d.ensemble import EnsembleConfig, UFEnsemble
from src.decoders.cube3d.graph import detector_graph
from src.decoders.cube3d.union_find import set_threads
from src.figures import plot_ensemble_vs_m, save_figure, timestamp, write_rows
from src.machine import fingerprint
from src.operating_point import wilson_interval

OUT = Path("results") / "ensemble"
P = 0.005
N = 50_000
N_THREADS = 10_000


def timed(fn, n):
    t0 = time.perf_counter()
    out = fn()
    return out, 1e6 * (time.perf_counter() - t0) / n


def main() -> None:
    fp = fingerprint()
    stamp = timestamp()
    print(fp, flush=True)
    rows, trows = [], []
    for d in (3, 5, 7, 9):
        c = surface_code_circuit(d, P)
        dets, obs = c.compile_detector_sampler(seed=0).sample(N, separate_observables=True)
        dets = np.ascontiguousarray(dets, np.uint8)
        g = detector_graph("circuit", d, P)

        def add(name, pred, us, M="", cores=1, threads=1):
            k = int((pred != obs).any(axis=1).sum())
            lo, hi = wilson_interval(k, N)
            rows.append({"decoder": name, "d": d, "p": P, "rounds": d, "shots": N, "M": M, "failures": k,
                         "ler": k / N, "ci_low": lo, "ci_high": hi, "us_per_shot": us, "threads": threads,
                         "cores_per_shot": cores, **fp})
            print(f"d={d} {name:<28} M={str(M):>2} LER {100 * k / N:.3f}% [{100 * lo:.3f},{100 * hi:.3f}] "
                  f"{us:8.2f} us/shot", flush=True)

        dem = c.detector_error_model(decompose_errors=True)
        m = pymatching.Matching.from_detector_error_model(dem)
        pm, t = timed(lambda: m.decode_batch(dets).astype(bool), N)
        add("PyMatching", pm, t)
        mc = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        pmc, t = timed(lambda: mc.decode_batch(dets, enable_correlations=True).astype(bool), N)
        add("Correlated PyMatching", pmc, t)

        set_threads(1)
        for correlated, Ms in ((False, (1, 8)), (True, (1, 4, 8, 12))):
            for M in Ms:
                ens = UFEnsemble(g, EnsembleConfig(M=M, correlated=correlated, parallel="shots"))
                ens.run(dets[:32])
                r, t = timed(lambda: ens.run(dets), N)
                add("UF3D correlated ensemble" if correlated else "UF3D ensemble", r.predictions, t, M=M)

        if d in (7, 9):
            sub = dets[:N_THREADS]
            for mode in ("variants", "shots"):
                ens = UFEnsemble(g, EnsembleConfig(M=8, correlated=True, parallel=mode))
                for th in (1, 2, 4, 8, 12):
                    set_threads(th)
                    ens.run(sub[:32])
                    _, t = timed(lambda: ens.run(sub), N_THREADS)
                    cores = min(8, th) if mode == "variants" else 1
                    trows.append({"decoder": f"UF3D correlated ensemble M=8 ({mode})", "d": d, "p": P,
                                  "shots": N_THREADS, "threads": th, "parallel": mode,
                                  "cores_per_shot": cores, "us_per_shot": t, **fp})
                    print(f"d={d} threads={th:>2} parallel over {mode:<8}: {t:8.2f} us/shot "
                          f"(cores per shot {cores})", flush=True)
            set_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    print("wrote", write_rows(rows, OUT / f"ensemble_{stamp}.csv"))
    print("wrote", write_rows(trows, OUT / f"ensemble_threads_{stamp}.csv"))
    meta = {**fp, "backend": "numba", "threads_note": "1 thread (all decoders)"}
    print("wrote", save_figure(plot_ensemble_vs_m(rows, meta), "ensemble_vs_M", stamp=stamp)[0])


if __name__ == "__main__":
    main()
