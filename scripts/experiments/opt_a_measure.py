"""Phase A: measure before optimizing.

p = 0.5% circuit-level, rounds = d, d = 5/7/9, 50,000 shots, seed 0.
1. Time split (numba, 1 thread, best of 3 after warm-up, same shots): PyMatching, plain
   UnionFind3D, UF3D correlated, Hierarchical (correlated deep tier), and the local fast
   path alone (Cube3D-Local at the Hierarchical gate K=6).
2. Per-shot counters from the instrumented reference (it runs the same algorithm as the
   numba kernel, bit-identically): growth rounds (pass 1 / pass 2), rounds in which no edge
   completed, lit count, clusters, whether pass 2 ran / was kept, clusters touched by a
   reweighted edge. Reported over all shots and over the shots Hierarchical sends to UF.

    python -m scripts.experiments.opt_a_measure [--shots N]
Writes results/optimization/phaseA_<ts>.csv (time split) and phaseA_counters_<ts>.csv.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.machine import clear_numba_cache

N_CLEARED = clear_numba_cache()  # before any kernel is imported / compiled

import numpy as np  # noqa: E402
import pymatching  # noqa: E402

from src.circuits import surface_code_circuit  # noqa: E402
from src.decoders.cube3d.graph import detector_graph  # noqa: E402
from src.decoders.cube3d.hierarchical import Hierarchical  # noqa: E402
from src.decoders.cube3d.local import Cube3DLocal  # noqa: E402
from src.decoders.cube3d.union_find import (UnionFind3D, UnionFind3DConfig, set_threads,  # noqa: E402
                                            uf_correlated_shot_reference)
from src.figures import timestamp, write_rows  # noqa: E402
from src.machine import fingerprint  # noqa: E402

OUT = Path("results") / "optimization"
P = 0.005
COUNTERS = ("lit", "rounds_pass1", "idle_rounds_pass1", "rounds_pass2", "clusters_pass1",
            "touched_clusters", "reweighted_edges", "max_cluster")


def best(fn, n, repeats=3):
    t = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - t0)
    return 1e6 * t / n


def summarize(name, vals):
    v = np.asarray(vals, dtype=float)
    return {f"{name}_mean": v.mean(), f"{name}_p50": np.percentile(v, 50), f"{name}_p90": np.percentile(v, 90),
            f"{name}_p99": np.percentile(v, 99), f"{name}_max": v.max()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=50_000)
    ap.add_argument("--distances", type=int, nargs="+", default=[5, 7, 9])
    args = ap.parse_args()
    N = args.shots
    fp = fingerprint()
    print(f"cleared {N_CLEARED} numba cache files; {fp}", flush=True)
    set_threads(1)
    trows, crows = [], []
    for d in args.distances:
        c = surface_code_circuit(d, P)
        dets, obs = c.compile_detector_sampler(seed=0).sample(N, separate_observables=True)
        dets = np.ascontiguousarray(dets, np.uint8)
        g = detector_graph("circuit", d, P)
        uf, ufc, h, loc = UnionFind3D(g), UnionFind3D(g, UnionFind3DConfig(correlated=True)), Hierarchical(g), Cube3DLocal(g)
        m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
        fns = {"PyMatching": lambda: m.decode_batch(dets), "UnionFind3D": lambda: uf.run(dets),
               "UF3D correlated": lambda: ufc.run(dets), "Hierarchical": lambda: h.run(dets),
               "local fast path only (K=6)": lambda: loc.run(dets, K=6, backend="numba")}
        for f in fns.values():  # warm-up (JIT compile after the cache was cleared)
            f()
        times = {k: best(f, N) for k, f in fns.items()}
        hres = h.run(dets)
        for k, t in times.items():
            trows.append({"d": d, "p": P, "shots": N, "decoder": k, "us_per_shot": t, "threads": 1, **fp})
        print(f"d={d} time split (us/shot, 1 thread): " + ", ".join(f"{k} {t:.2f}" for k, t in times.items()),
              f"| Hierarchical sends {100 * hres.uf_called.mean():.1f}% of shots to UF", flush=True)

        t0 = time.perf_counter()
        stats = []
        for s in range(N):
            st: dict = {}
            uf_correlated_shot_reference(g, ufc.tables, dets[s], st)
            st.pop("labels", None)
            stats.append(st)
        print(f"d={d} reference counters on {N} shots in {time.perf_counter() - t0:.0f} s", flush=True)
        for subset, mask in (("all shots", np.ones(N, bool)), ("shots Hierarchical sends to UF", hres.uf_called)):
            sel = [stats[i] for i in np.flatnonzero(mask)]
            row = {"d": d, "p": P, "shots": N, "subset": subset, "n": len(sel)}
            for k in COUNTERS:
                row.update(summarize(k, [x.get(k, 0) for x in sel]))
            row["pass2_ran_frac"] = float(np.mean([x.get("pass2_ran", False) for x in sel]))
            row["pass2_kept_frac"] = float(np.mean([x.get("pass2_kept", False) for x in sel]))
            row["idle_round_frac"] = float(np.sum([x.get("idle_rounds_pass1", 0) for x in sel])
                                           / max(1, np.sum([x.get("rounds_pass1", 0) for x in sel])))
            crows.append({**row, **fp})
            print(f"d={d} [{subset}, n={len(sel)}] " + " ".join(
                f"{k}: mean {row[k + '_mean']:.2f} p50 {row[k + '_p50']:.0f} p90 {row[k + '_p90']:.0f} "
                f"p99 {row[k + '_p99']:.0f};" for k in COUNTERS)
                + f" pass2 ran {100 * row['pass2_ran_frac']:.1f}% kept {100 * row['pass2_kept_frac']:.1f}%"
                f" idle-round share {100 * row['idle_round_frac']:.1f}%", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = timestamp()
    print("wrote", write_rows(trows, OUT / f"phaseA_{stamp}.csv"))
    print("wrote", write_rows(crows, OUT / f"phaseA_counters_{stamp}.csv"))


if __name__ == "__main__":
    main()
