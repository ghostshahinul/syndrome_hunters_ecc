"""Standard benchmark: measure, write CSVs, then draw every standard figure.

    python -m src.benchmark                     # everything (about 10 min on a laptop)
    python -m src.benchmark --quick             # small shot counts, for a smoke run
    python -m src.benchmark --only accuracy     # one part: accuracy | sweep | threads

Parts
  accuracy  circuit-level p = 0.5%, d = 3/5/7/9, every decoder on the same shots (seed 0),
            every decoder single-thread (the fair comparison with PyMatching).
            -> results/accuracy_runtime_<ts>.csv, accuracy_runtime + cube_collapse_story PNGs
  sweep     p in 0.1-0.8%, d = 3/5/7/9, single-thread.
            -> results/sweep_<ts>.csv, accuracy_runtime_sweep + fast_path_fraction PNGs
  threads   d = 7/9, p = 0.5%, UnionFind3D and Hierarchical at 1/2/4/8/12 threads, plus
            single-thread PyMatching. -> results/thread_scaling_<ts>.csv, thread_scaling PNG

Timing hygiene: every decoder is warmed up (numba JIT, caches) before timing; times are the
best of `repeats` full-batch decode calls; the machine fingerprint (CPU, MHz, power state,
git commit) is written into every row. Compare times only within one run.
PyMatching appears only as a reference baseline.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numba
import numpy as np
import stim

from src.circuits import detector_error_model, surface_code_circuit
from src.decoders.cube3d.graph import build_detector_graph
from src.decoders.cube3d.hierarchical import Hierarchical, HierarchicalConfig
from src.decoders.cube3d.union_find import UnionFind3D, UnionFind3DConfig, set_threads
from src.figures import (plot_accuracy_runtime, plot_cube_collapse_story, plot_fast_path_fraction,
                         plot_sweep, plot_thread_scaling, save_figure, timestamp, write_rows)
from src.machine import fingerprint
from src.operating_point import DECODER_FACTORIES, wilson_interval

RESULTS_DIR = Path("results")
BASE_META = {"basis": "z", "rounds_rule": "rounds = distance", "noise": "circuit-level noise"}


@dataclass
class Prepared:
    decode: Callable[[np.ndarray], np.ndarray]
    numba: bool = False
    extra: Callable[[np.ndarray], dict] | None = None   # extra CSV fields (e.g. fast_fraction)


def _baseline(name: str) -> Callable[[stim.DetectorErrorModel], Prepared]:
    return lambda dem: Prepared(DECODER_FACTORIES[name](dem))


def _uf(dem: stim.DetectorErrorModel, correlated: bool = False) -> Prepared:
    eng = UnionFind3D(build_detector_graph(dem), UnionFind3DConfig(correlated=correlated))
    return Prepared(lambda d: eng.run(d).predictions, numba=True)


def _hier(gate: str) -> Callable[[stim.DetectorErrorModel], Prepared]:
    def build(dem: stim.DetectorErrorModel) -> Prepared:
        eng = Hierarchical(build_detector_graph(dem), HierarchicalConfig(gate=gate))
        return Prepared(lambda d: eng.run(d).predictions, numba=True,
                        extra=lambda d: {"fast_fraction": eng.run(d).fast_fraction})
    return build


ENTRIES: dict[str, Callable[[stim.DetectorErrorModel], Prepared]] = {
    "Cube-Collapse": _baseline("Cube-Collapse"),
    "PyMatching": _baseline("PyMatching"),
    "Correlated PyMatching": _baseline("Correlated PyMatching"),
    "PyMatching 2D on flat": _baseline("PyMatching 2D on flat"),
    "UnionFind3D": _uf,
    "UF3D correlated": lambda dem: _uf(dem, correlated=True),
    "Hierarchical": _hier("whole_shot"),
    "Hierarchical (partial)": _hier("partial"),
}


def best_time(fn: Callable[[], np.ndarray], shots: int, repeats: int) -> tuple[np.ndarray, float]:
    best, out = float("inf"), None
    for i in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        if dt > 5.0:  # slow decoders: one timing is enough
            break
    return out, 1e6 * best / shots


def _row(name, d, p, shots, k, us, threads, backend, fp, run, **extra) -> dict:
    lo, hi = wilson_interval(k, shots)
    return {"decoder": name, "d": d, "p": p, "rounds": d, "shots": shots, "failures": k,
            "ler": k / shots, "ci_low": lo, "ci_high": hi, "us_per_shot": us, "threads": threads,
            "backend": backend, "fast_fraction": extra.pop("fast_fraction", ""), **extra,
            **BASE_META, "run": run, **fp}


def measure_points(decoders, distances, ps, shots, seed, thread_counts, repeats, fp, run, log=print):
    rows = []
    for p in ps:
        for d in distances:
            circuit = surface_code_circuit(d, p)
            dem = detector_error_model(circuit)
            dets, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
            dets = np.ascontiguousarray(dets, dtype=np.uint8)
            for name in decoders:
                prep = ENTRIES[name](dem)
                prep.decode(dets[:64])  # warm-up
                for th in (thread_counts if prep.numba else (1,)):
                    if prep.numba:
                        set_threads(th)
                    pred, us = best_time(lambda: prep.decode(dets), shots, repeats)
                    k = int(np.any(pred != obs, axis=1).sum())
                    extra = prep.extra(dets) if prep.extra else {}
                    rows.append(_row(name, d, p, shots, k, us, th, "numba" if prep.numba else "python/C++",
                                     fp, run, **extra))
                    if log:
                        r = rows[-1]
                        log(f"p={p:<6g} d={d} {name:<23} thr={th:<2} LER {100 * r['ler']:.4f}% "
                            f"[{100 * r['ci_low']:.4f},{100 * r['ci_high']:.4f}] {us:8.2f} us/shot"
                            + (f"  fast {100 * extra['fast_fraction']:.1f}%" if extra else ""))
    set_threads(numba.config.NUMBA_NUM_THREADS)
    return rows


def next_run_number(directory: Path = RESULTS_DIR) -> int:
    nums = []
    for f in directory.glob("accuracy_runtime_*.csv"):
        try:
            with open(f, newline="", encoding="utf-8") as fh:
                r = next(csv.DictReader(fh), None)
            if r and str(r.get("run", "")).isdigit():
                nums.append(int(r["run"]))
        except OSError:
            pass
    return max(nums, default=0) + 1


def meta_for(fp: dict, run: int, backend: str = "numba") -> dict:
    return {**BASE_META, **fp, "run": run, "backend": backend}


def run_accuracy(args, fp, run, stamp, max_threads):
    # Fair speed reporting: single thread only here; multi-thread scaling is its own figure.
    rows = measure_points(args.decoders, args.distances, [args.p], args.shots, args.seed,
                          (1,), args.repeats, fp, run)
    csv_path = write_rows(rows, args.out / f"accuracy_runtime_{stamp}.csv")
    meta = meta_for(fp, run)
    figs = [save_figure(plot_accuracy_runtime(rows, meta), "accuracy_runtime", args.out / "plots", stamp)]
    if any(r["decoder"] == "Cube-Collapse" for r in rows):
        figs.append(save_figure(plot_cube_collapse_story(rows, meta), "cube_collapse_story",
                                args.out / "plots", stamp))
    return csv_path, figs


def run_sweep(args, fp, run, stamp):
    rows = measure_points(args.sweep_decoders, args.distances, args.sweep_ps, args.sweep_shots, args.seed,
                          (1,), 1, fp, run)
    csv_path = write_rows(rows, args.out / f"sweep_{stamp}.csv")
    meta = meta_for(fp, run)
    figs = [save_figure(plot_sweep(rows, meta), "accuracy_runtime_sweep", args.out / "plots", stamp)]
    fp_rows = [r for r in rows if r["decoder"] == "Hierarchical"]
    if fp_rows:
        figs.append(save_figure(plot_fast_path_fraction(fp_rows, meta), "fast_path_fraction",
                                args.out / "plots", stamp))
    return csv_path, figs


def run_threads(args, fp, run, stamp, max_threads):
    counts = [t for t in (1, 2, 4, 8, 12) if t <= max_threads]
    if max_threads not in counts:
        counts.append(max_threads)
    rows = []
    for d in args.thread_distances:
        rows += measure_points(["UnionFind3D", "Hierarchical"], [d], [args.p], args.thread_shots, args.seed,
                               counts, 3, fp, run)
        rows += measure_points(["PyMatching"], [d], [args.p], args.thread_shots, args.seed, (1,), 3, fp, run)
    csv_path = write_rows(rows, args.out / f"thread_scaling_{stamp}.csv")
    figs = [save_figure(plot_thread_scaling(rows, meta_for(fp, run)), "thread_scaling", args.out / "plots", stamp)]
    return csv_path, figs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=("accuracy", "sweep", "threads"), action="append")
    ap.add_argument("--quick", action="store_true", help="tiny shot counts (smoke run)")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR, help="results directory (plots go to <out>/plots)")
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--shots", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--decoders", nargs="+", default=list(ENTRIES))
    ap.add_argument("--sweep-ps", type=float, nargs="+", default=[0.001, 0.002, 0.003, 0.005, 0.008])
    ap.add_argument("--sweep-shots", type=int, default=100_000)
    ap.add_argument("--sweep-decoders", nargs="+",
                    default=["PyMatching", "Correlated PyMatching", "UnionFind3D", "UF3D correlated",
                             "Hierarchical", "Cube-Collapse"])
    ap.add_argument("--thread-distances", type=int, nargs="+", default=[7, 9])
    ap.add_argument("--thread-shots", type=int, default=100_000)
    args = ap.parse_args(argv)
    if args.quick:
        args.shots, args.sweep_shots, args.thread_shots, args.repeats = 2000, 2000, 4000, 1
    for n in args.decoders + args.sweep_decoders:
        if n not in ENTRIES:
            raise SystemExit(f"unknown decoder {n!r}; choose from {list(ENTRIES)}")
    parts = args.only or ["accuracy", "sweep", "threads"]
    max_threads = numba.config.NUMBA_NUM_THREADS
    fp = fingerprint()
    run = next_run_number(args.out)
    stamp = timestamp()
    print(f"benchmark run {run} ({stamp}): {fp}")
    outputs = []
    if "accuracy" in parts:
        outputs.append(run_accuracy(args, fp, run, stamp, max_threads))
    if "sweep" in parts:
        outputs.append(run_sweep(args, fp, run, stamp))
    if "threads" in parts:
        outputs.append(run_threads(args, fp, run, stamp, max_threads))
    for csv_path, figs in outputs:
        print(f"wrote {csv_path}")
        for stamped, latest in figs:
            print(f"wrote {stamped} (and {latest.name})")


if __name__ == "__main__":
    main()
