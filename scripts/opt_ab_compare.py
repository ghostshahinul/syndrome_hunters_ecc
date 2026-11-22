"""Optimization phases C-E: compare decoder configurations on the same shots, in one run.

p = 0.5% circuit-level, rounds = d, B batches x N shots (seeds 0..B-1, the same for every
config), single thread, warm-up first, numba cache cleared at start. Per config: failures
pooled over all batches with a 95% Wilson CI, mean batch µs/shot, and optional extra counts
(e.g. the share of shots answered by the single-fault table). Plain PyMatching runs in the
same loop as the reference.

Configs are named on the command line from CONFIGS below, e.g.
    python -m scripts.opt_ab_compare --configs hier hier_notable ufc ufc_notable --tag phaseC
Writes results/optimization/<tag>_<ts>.csv.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.machine import clear_numba_cache

N_CLEARED = clear_numba_cache()

import numpy as np  # noqa: E402
import pymatching  # noqa: E402

from src.circuits import surface_code_circuit  # noqa: E402
from src.decoders.cube3d.graph import detector_graph  # noqa: E402
from src.decoders.cube3d.hierarchical import Hierarchical, HierarchicalConfig  # noqa: E402
from src.decoders.cube3d.union_find import UnionFind3D, UnionFind3DConfig, set_threads  # noqa: E402
from src.figures import timestamp, write_rows  # noqa: E402
from src.machine import fingerprint  # noqa: E402
from src.operating_point import wilson_interval  # noqa: E402

OUT = Path("results") / "optimization"
P = 0.005


def _uf(**kw):
    return lambda g: UnionFind3D(g, UnionFind3DConfig(**kw))


def _h(uf=None, **kw):
    return lambda g: Hierarchical(g, HierarchicalConfig(uf=uf or UnionFind3DConfig(correlated=True), **kw))


# name -> factory(graph) -> engine with .run(dets) returning .predictions (+ optional .table)
CONFIGS = {
    "uf": _uf(),
    "ufc": _uf(correlated=True, table=True),
    "ufc_notable": _uf(correlated=True),
    "hier": _h(),
    "hier_notable": _h(table=False),
    # Phase D (cluster-local pass 2) configs were removed with the reverted code; see PROGRESS.md
}


def register(name, factory):
    CONFIGS[name] = factory


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--shots", type=int, default=50_000)
    ap.add_argument("--batches", type=int, default=5)
    args = ap.parse_args(argv)
    fp = fingerprint()
    print(f"cleared {N_CLEARED} numba cache files; {fp}", flush=True)
    set_threads(1)
    rows = []
    for d in args.distances:
        c = surface_code_circuit(d, P)
        g = detector_graph("circuit", d, P)
        m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
        engines = {"PyMatching": None, **{k: CONFIGS[k](g) for k in args.configs}}
        fails = {k: 0 for k in engines}
        times = {k: [] for k in engines}
        extra = {k: {} for k in engines}
        for b in range(args.batches):
            dets, obs = c.compile_detector_sampler(seed=b).sample(args.shots, separate_observables=True)
            dets = np.ascontiguousarray(dets, np.uint8)
            for k, e in engines.items():
                fn = (lambda: m.decode_batch(dets).astype(bool)) if e is None else (lambda: e.run(dets))
                if b == 0:
                    fn()  # warm-up
                t0 = time.perf_counter()
                out = fn()
                times[k].append(1e6 * (time.perf_counter() - t0) / args.shots)
                pred = out if e is None else out.predictions
                fails[k] += int(np.any(pred != obs, axis=1).sum())
                for attr in ("table", "fast"):
                    v = getattr(out, attr, None)
                    if v is not None:
                        extra[k][attr] = extra[k].get(attr, 0) + int(np.sum(v))
        total = args.shots * args.batches
        pm_t = float(np.mean(times["PyMatching"]))
        for k in engines:
            lo, hi = wilson_interval(fails[k], total)
            t = float(np.mean(times[k]))
            row = {"tag": args.tag, "config": k, "d": d, "p": P, "shots": total, "failures": fails[k],
                   "ler": fails[k] / total, "ci_low": lo, "ci_high": hi, "us_per_shot": t,
                   "us_min": min(times[k]), "us_max": max(times[k]), "vs_pymatching": t / pm_t,
                   **{f"{a}_fraction": v / total for a, v in extra[k].items()}, "threads": 1, **fp}
            rows.append(row)
            ex = " ".join(f"{a} {100 * v / total:.2f}%" for a, v in extra[k].items())
            print(f"d={d} {k:<22} LER {100 * fails[k] / total:.4f}% [{100 * lo:.4f},{100 * hi:.4f}] "
                  f"{t:8.2f} us ({t / pm_t:.2f}x PM) {ex}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    print("wrote", write_rows(rows, OUT / f"{args.tag}_{timestamp()}.csv"))


if __name__ == "__main__":
    main()
