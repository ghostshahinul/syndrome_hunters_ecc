"""One-minute demo: PyMatching vs our plain UnionFind3D vs our Hierarchical decoder.

Circuit-level rotated Z memory, p = 0.5%, d = 7, rounds = 7, 50,000 shots (seed 0), every
decoder single-threaded (the fair comparison). Prints a table and saves
results/plots/demo_latest.png (plus a timestamped copy).

    python -m scripts.demo                 # about 30-60 s, including numba compilation
    python -m scripts.demo --d 9 --shots 20000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pymatching

from src.circuits import surface_code_circuit
from src.decoders.cube3d.graph import detector_graph
from src.decoders.cube3d.hierarchical import Hierarchical
from src.decoders.cube3d.union_find import UnionFind3D, set_threads
from src.figures import plot_accuracy_runtime, save_figure
from src.machine import fingerprint
from src.operating_point import wilson_interval


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, default=7)
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--shots", type=int, default=50_000)
    args = ap.parse_args()
    t_start = time.perf_counter()
    set_threads(1)

    circuit = surface_code_circuit(args.d, args.p)
    dets, obs = circuit.compile_detector_sampler(seed=0).sample(args.shots, separate_observables=True)
    dets = np.ascontiguousarray(dets, np.uint8)
    g = detector_graph("circuit", args.d, args.p)
    m = pymatching.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))
    uf, hier = UnionFind3D(g), Hierarchical(g)
    decoders = {
        "PyMatching": lambda x: m.decode_batch(x).astype(bool),
        "UnionFind3D": lambda x: uf.run(x).predictions,
        "Hierarchical": lambda x: hier.run(x).predictions,
    }
    for decode in decoders.values():  # warm-up (numba compiles on first use)
        decode(dets[:64])

    rows = []
    for name, decode in decoders.items():
        best = float("inf")
        for _ in range(2):
            t0 = time.perf_counter()
            pred = decode(dets)
            best = min(best, time.perf_counter() - t0)
        k = int(np.any(pred != obs, axis=1).sum())
        lo, hi = wilson_interval(k, args.shots)
        rows.append({"decoder": name, "d": args.d, "p": args.p, "shots": args.shots, "failures": k,
                     "ler": k / args.shots, "ci_low": lo, "ci_high": hi,
                     "us_per_shot": 1e6 * best / args.shots, "threads": 1})

    pm = rows[0]
    print(f"\nRotated Z memory, circuit-level p = {100 * args.p:.2f}%, d = {args.d}, {args.shots:,} shots, "
          f"single thread\n")
    print(f"{'decoder':<14} {'failures':>8}  {'logical failure rate [95% CI]':<30} {'us/shot':>8}  vs PyMatching")
    for r in rows:
        ci = f"{100 * r['ler']:.3f}% [{100 * r['ci_low']:.3f}, {100 * r['ci_high']:.3f}]"
        print(f"{r['decoder']:<14} {r['failures']:>8}  {ci:<30} {r['us_per_shot']:>8.2f}  "
              f"LER x{r['ler'] / pm['ler']:.2f}, time x{r['us_per_shot'] / pm['us_per_shot']:.2f}")

    meta = {**fingerprint(), "backend": "numba", "threads_note": "every decoder single-thread"}
    stamped, latest = save_figure(plot_accuracy_runtime(rows, meta), "demo")
    print(f"\nsaved {latest} (and {stamped.name}); total {time.perf_counter() - t_start:.0f} s")


if __name__ == "__main__":
    main()
