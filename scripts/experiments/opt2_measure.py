"""Final round, Step 1: how often (and how expensively) does correlated UF's pass 2 matter?

p = 0.5% circuit-level, d = 5/7/9, 50,000 shots (seed 0). From the instrumented reference
(bit-identical to numba): the fraction of shots where pass 2 runs, where its correction is
kept, where it CHANGES the predicted observable, and where it runs although no shortened edge
has an endpoint among pass 1's visited nodes (then pass 2 must reproduce pass 1 exactly).
Time split (numba, 1 thread, best of 3, same shots): plain UF (= pass 1), UF3D correlated
(pass 1 + pass 2), the local tier alone and Hierarchical.

    python -m scripts.experiments.opt2_measure
Writes results/optimization/final_step1_<ts>.csv.
"""

from __future__ import annotations

import time
from pathlib import Path

from src.machine import clear_numba_cache

N_CLEARED = clear_numba_cache()

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

N = 50_000
P = 0.005


def best(fn, n):
    t = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - t0)
    return 1e6 * t / n


def main() -> None:
    fp = fingerprint()
    print(f"cleared {N_CLEARED} cache files; {fp}", flush=True)
    set_threads(1)
    rows = []
    for d in (5, 7, 9):
        c = surface_code_circuit(d, P)
        dets = np.ascontiguousarray(c.compile_detector_sampler(seed=0).sample(N, separate_observables=True)[0],
                                    np.uint8)
        g = detector_graph("circuit", d, P)
        uf, ufc = UnionFind3D(g), UnionFind3D(g, UnionFind3DConfig(correlated=True))
        h, loc = Hierarchical(g), Cube3DLocal(g)
        m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
        fns = {"PyMatching": lambda: m.decode_batch(dets), "plain UF (= pass 1)": lambda: uf.run(dets),
               "UF3D correlated (pass 1 + 2)": lambda: ufc.run(dets), "local tier only (K=6)":
               lambda: loc.run(dets, K=6, backend="numba"), "Hierarchical": lambda: h.run(dets)}
        for f in fns.values():
            f()
        t = {k: best(f, N) for k, f in fns.items()}
        ran = kept = changed = needed = 0
        t0 = time.perf_counter()
        for s in range(N):
            st: dict = {}
            uf_correlated_shot_reference(g, ufc.tables, dets[s], st)
            ran += bool(st.get("pass2_ran"))
            kept += bool(st.get("pass2_kept"))
            changed += bool(st.get("pass2_changed_obs"))
            needed += bool(st.get("pass2_ran")) and bool(st.get("pass2_needed"))
        row = {"d": d, "p": P, "shots": N, "pass2_ran": ran / N, "pass2_kept": kept / N,
               "pass2_changed_obs": changed / N, "pass2_ran_but_not_needed": (ran - needed) / N,
               **{f"us_{k}": v for k, v in t.items()},
               "us_pass2_estimate": t["UF3D correlated (pass 1 + 2)"] - t["plain UF (= pass 1)"], **fp}
        rows.append(row)
        print(f"d={d}: pass 2 runs {100 * ran / N:.2f}% | kept {100 * kept / N:.2f}% | changes the prediction "
              f"{100 * changed / N:.3f}% | runs but no shortened edge touches pass-1 nodes {100 * (ran - needed) / N:.2f}% "
              f"(reference {time.perf_counter() - t0:.0f} s)", flush=True)
        print("   time split (us/shot): " + ", ".join(f"{k} {v:.2f}" for k, v in t.items())
              + f"; pass 2 ~ {row['us_pass2_estimate']:.2f}", flush=True)
    out = Path("results") / "optimization"
    out.mkdir(parents=True, exist_ok=True)
    print("wrote", write_rows(rows, out / f"final_step1_{timestamp()}.csv"))


if __name__ == "__main__":
    main()
