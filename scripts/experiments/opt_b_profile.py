"""Phase B3: time the Phase-B kernels against the pre-Phase-B kernels in ONE run.

The old union_find / local / hierarchical modules are loaded from git (`--old-rev`, default
the Phase A commit) into a temporary directory, with their imports rewritten to point at each
other. Both versions decode the same shots: same predictions are asserted, then each engine
is timed single-thread, best of 3, interleaved. Growth rounds per shot (round-by-round vs
executed with skip-ahead) come from the instrumented reference on a subsample.

    python -m scripts.experiments.opt_b_profile [--shots N] [--old-rev REV]
Writes results/optimization/phaseB_<ts>.csv.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from src.machine import clear_numba_cache

N_CLEARED = clear_numba_cache()

import numpy as np  # noqa: E402
import pymatching  # noqa: E402

from src.circuits import surface_code_circuit  # noqa: E402
from src.decoders.cube3d.graph import detector_graph  # noqa: E402
from src.decoders.cube3d import hierarchical as hier_new  # noqa: E402
from src.decoders.cube3d import union_find as uf_new  # noqa: E402
from src.figures import timestamp, write_rows  # noqa: E402
from src.machine import fingerprint  # noqa: E402

OUT = Path("results") / "optimization"
P = 0.005
MODULES = {"union_find": "uf_old", "local": "local_old", "hierarchical": "hier_old"}


def load_old(rev: str):
    tmp = Path(tempfile.mkdtemp(prefix="opt_old_"))
    for name, alias in MODULES.items():
        src = subprocess.run(["git", "show", f"{rev}:src/decoders/cube3d/{name}.py"], capture_output=True,
                             text=True, check=True).stdout
        for n2, a2 in MODULES.items():
            src = src.replace(f"from src.decoders.cube3d.{n2} import", f"from {a2} import")
        (tmp / f"{alias}.py").write_text(src, encoding="utf-8")
    sys.path.insert(0, str(tmp))
    return importlib.import_module("uf_old"), importlib.import_module("hier_old")


def best(fn, n, repeats=3):
    t = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - t0)
    return 1e6 * t / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=50_000)
    ap.add_argument("--old-rev", default="7a62911")
    ap.add_argument("--rounds-shots", type=int, default=5_000)
    ap.add_argument("--distances", type=int, nargs="+", default=[5, 7, 9])
    args = ap.parse_args()
    uf_old, hier_old = load_old(args.old_rev)
    fp = fingerprint()
    print(f"cleared {N_CLEARED} cache files; old = {args.old_rev}; {fp}", flush=True)
    uf_new.set_threads(1)
    rows = []
    N = args.shots
    for d in args.distances:
        c = surface_code_circuit(d, P)
        dets = np.ascontiguousarray(c.compile_detector_sampler(seed=0).sample(N, separate_observables=True)[0],
                                    np.uint8)
        g = detector_graph("circuit", d, P)
        eng = {
            ("UnionFind3D", "old"): uf_old.UnionFind3D(g),
            ("UnionFind3D", "new"): uf_new.UnionFind3D(g),
            ("UF3D correlated", "old"): uf_old.UnionFind3D(g, uf_old.UnionFind3DConfig(correlated=True)),
            ("UF3D correlated", "new"): uf_new.UnionFind3D(g, uf_new.UnionFind3DConfig(correlated=True)),
            ("Hierarchical", "old"): hier_old.Hierarchical(g),
            ("Hierarchical", "new"): hier_new.Hierarchical(g),
        }
        m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
        for name in ("UnionFind3D", "UF3D correlated", "Hierarchical"):
            po = eng[(name, "old")].run(dets).predictions
            pn = eng[(name, "new")].run(dets).predictions
            assert np.array_equal(po, pn), f"{name} d={d}: old and new predictions differ"
        m.decode_batch(dets[:64])
        t = {"PyMatching": best(lambda: m.decode_batch(dets), N)}
        for (name, ver), e in eng.items():
            t[f"{name} ({ver})"] = best(lambda: e.run(dets), N)
        # growth rounds per shot from the instrumented reference (identical algorithm)
        ufc = eng[("UF3D correlated", "new")]
        st_all = []
        for s in range(min(args.rounds_shots, N)):
            st: dict = {}
            uf_new.uf_correlated_shot_reference(g, ufc.tables, dets[s], st)
            st_all.append(st)
        r1 = np.mean([x["rounds_pass1"] for x in st_all])
        i1 = np.mean([x["idle_rounds_pass1"] for x in st_all])
        r2 = np.mean([x["rounds_pass2"] for x in st_all])
        line = {"d": d, "p": P, "shots": N, **{f"us_{k}": v for k, v in t.items()},
                "rounds_pass1_round_by_round": r1 + i1, "rounds_pass1_executed": r1,
                "rounds_pass2_executed": r2, "threads": 1, **fp}
        rows.append(line)
        print(f"d={d}: " + ", ".join(f"{k} {v:.2f}" for k, v in t.items()), flush=True)
        for name in ("UnionFind3D", "UF3D correlated", "Hierarchical"):
            o, nw = t[f"{name} (old)"], t[f"{name} (new)"]
            print(f"   {name}: {o:.2f} -> {nw:.2f} us ({o / nw:.2f}x); vs PyMatching {nw / t['PyMatching']:.2f}x",
                  flush=True)
        print(f"   pass-1 growth rounds per shot: {r1 + i1:.2f} round-by-round -> {r1:.2f} executed "
              f"({i1:.2f} skipped); pass 2 executed {r2:.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    print("wrote", write_rows(rows, OUT / f"phaseB_{timestamp()}.csv"))


if __name__ == "__main__":
    main()
