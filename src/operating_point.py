"""Accuracy vs batch time at one operating point (circuit-level rotated memory).

For each distance: sample `shots` once (fixed seed), decode the same samples with every
decoder, and record logical failure rate (95% Wilson interval) and batch time per shot
(wall time of one decode call over all shots / shots, decoder construction excluded).

CLI:  python -m src.operating_point --distances 3 5 7 --p 0.005 --shots 50000
Each run writes results/operating_point/operating_point_run<N>.csv and .png, with N
one higher than the last run (never overwrites). pytest uses the same numbering.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pymatching
import stim

from src.circuits import detector_error_model, surface_code_circuit
from src.decoders import DECODERS
from src.decoders.cube_collapse import CubeCollapseDecoder
from src.decoders.cube_collapse.baselines import decode_2d, pymatching_2d
from src.decoders.cube_collapse.filter import temporal_filter
from src.decoders.cube_collapse.layout import build_cube

# (label, factory(dem) -> decode(dets) -> predictions). Order = legend + colour order.
DecodeFn = Callable[[np.ndarray], np.ndarray]


def _cube_collapse(dem: stim.DetectorErrorModel) -> DecodeFn:
    return CubeCollapseDecoder(dem).decode_batch


def _pymatching(dem: stim.DetectorErrorModel) -> DecodeFn:
    m = pymatching.Matching.from_detector_error_model(dem)
    return lambda dets: m.decode_batch(dets).astype(bool)


def _pymatching_correlated(dem: stim.DetectorErrorModel) -> DecodeFn:
    m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    return lambda dets: m.decode_batch(dets, enable_correlations=True).astype(bool)


def _pymatching_2d_on_flat(dem: stim.DetectorErrorModel) -> DecodeFn:
    """Same cube + filter as Cube-Collapse, then MWPM on its 2D graph (timed end to end)."""
    cc = CubeCollapseDecoder(dem)
    m = pymatching_2d(cc.graph)

    def decode(dets: np.ndarray) -> np.ndarray:
        raw = build_cube(dets, cc.layout, cc.config.include_final_layer)
        return decode_2d(m, temporal_filter(raw, cc.config.late_k).flat, cc.num_observables)
    return decode


DECODER_FACTORIES: dict[str, Callable[[stim.DetectorErrorModel], DecodeFn]] = {
    "Cube-Collapse": _cube_collapse,
    "PyMatching": _pymatching,
    "Correlated PyMatching": _pymatching_correlated,
    "PyMatching 2D on flat": _pymatching_2d_on_flat,
}


def implemented_registry_decoders() -> dict[str, Callable[[stim.DetectorErrorModel], DecodeFn]]:
    """Other registered decoders (e.g. union-find) once they stop raising NotImplementedError."""
    out = {}
    skip = {"pymatching", "cube_collapse"}
    for name, cls in DECODERS.items():
        if name in skip:
            continue
        probe_dem = detector_error_model(surface_code_circuit(3, 0.001))
        try:
            cls(probe_dem).decode_batch(np.zeros((1, probe_dem.num_detectors), dtype=bool))
        except NotImplementedError:
            continue
        out[name] = lambda dem, cls=cls: cls(dem).decode_batch
    return out


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


@dataclass
class OperatingPoint:
    decoder: str
    distance: int
    rounds: int
    p: float
    shots: int
    failures: int
    ler: float
    ler_low: float
    ler_high: float
    us_per_shot: float


def measure(distances=(3, 5, 7), p: float = 0.005, shots: int = 50_000, seed: int = 0,
            factories: dict | None = None, log: Callable[[str], None] | None = print) -> list[OperatingPoint]:
    factories = factories or {**DECODER_FACTORIES, **implemented_registry_decoders()}
    rows = []
    for d in distances:
        circuit = surface_code_circuit(d, p)
        dem = detector_error_model(circuit)
        dets, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
        for label, factory in factories.items():
            decode = factory(dem)
            decode(dets[:16])  # warm-up: first-call allocation / caches
            t0 = time.perf_counter()
            pred = decode(dets)
            elapsed = time.perf_counter() - t0
            k = int(np.any(pred != obs, axis=1).sum())
            lo, hi = wilson_interval(k, shots)
            row = OperatingPoint(label, d, d, p, shots, k, k / shots, lo, hi, 1e6 * elapsed / shots)
            rows.append(row)
            if log:
                log(f"d={d} {label:<24} failures={k:>5}/{shots}  LER={100 * row.ler:.3f}% "
                    f"[{100 * lo:.3f}, {100 * hi:.3f}]  {row.us_per_shot:.2f} us/shot")
    return rows


# ---- plot ------------------------------------------------------------------------------
# Reference categorical palette, fixed order (dataviz skill, references/palette.md),
# plus a distinct marker per decoder so identity never relies on colour alone.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def plot_operating_point(rows: list[OperatingPoint], path: Path, run_number: int | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    decoders = list(dict.fromkeys(r.decoder for r in rows))
    distances = sorted({r.distance for r in rows})
    style = {name: (SERIES_COLORS[i % 8], SERIES_MARKERS[i % 8]) for i, name in enumerate(decoders)}
    first = rows[0]

    fig, axes = plt.subplots(1, len(distances), figsize=(5.4 * len(distances), 5.2),
                             sharey=True, facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    for ax, d in zip(axes, distances):
        ax.set_facecolor(SURFACE)
        pts = [r for r in rows if r.distance == d]
        for r in pts:
            color, marker = style[r.decoder]
            y = 100 * r.ler
            ax.errorbar(r.us_per_shot, y, yerr=[[y - 100 * r.ler_low], [100 * r.ler_high - y]],
                        fmt=marker, color=color, ms=9, mec=SURFACE, mew=1.5,
                        elinewidth=2, capsize=4, capthick=2, zorder=3)
        ax.set_title(f"Distance {d}", fontsize=14, color=INK, pad=10)
        ax.set_xlim(0, max(r.us_per_shot for r in pts) * 1.12)
        ax.set_xlabel("Batch time per shot (µs)", fontsize=12, color=INK_2)
        ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK_2, labelsize=11, length=4, color=AXIS)
    axes[0].set_ylabel("Logical failures (%)", fontsize=12, color=INK_2)

    fig.text(0.06, 0.985, "Accuracy and batch time at one operating point",
             fontsize=22, fontweight="bold", color=INK, va="top")
    run_label = f"  ·  run {run_number}" if run_number is not None else ""
    fig.text(0.06, 0.915, f"Rotated Z memory  ·  rounds = distance  ·  {first.shots:,} shots per point"
             f"  ·  physical error {100 * first.p:.2f}%  ·  circuit-level noise{run_label}",
             fontsize=12, color=INK_2, va="top")
    handles = [Line2D([], [], color=style[n][0], marker=style[n][1], ms=9, lw=2.5, label=n)
               for n in decoders]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, 0.875), ncol=len(decoders),
               frameon=False, fontsize=12, handlelength=2.2, columnspacing=2.2)
    fig.text(0.06, 0.015, "Lower and further left is better. Whiskers show 95% Wilson intervals "
             "for the failure rate. Batch time = one decode call over all shots / shots.",
             fontsize=10.5, color=INK_2)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.71, bottom=0.17, wspace=0.18)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def write_csv(rows: list[OperatingPoint], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0])))
        w.writeheader()
        w.writerows(asdict(r) for r in rows)
    return path


RUNS_DIR = Path("results") / "operating_point"
RUN_STEM = "operating_point_run"


def next_run_number(directory: Path = RUNS_DIR) -> int:
    """1 + the highest N among existing operating_point_run<N>.* files (1 if none)."""
    numbers = [int(m.group(1)) for f in directory.glob(f"{RUN_STEM}*")
               if (m := re.fullmatch(rf"{RUN_STEM}(\d+)\.\w+", f.name))]
    return max(numbers, default=0) + 1


def save_run(rows: list[OperatingPoint], directory: Path = RUNS_DIR) -> tuple[int, Path, Path]:
    """Write operating_point_run<N>.csv/.png with the next free N. Never overwrites."""
    directory.mkdir(parents=True, exist_ok=True)
    while True:
        n = next_run_number(directory)
        csv_path = directory / f"{RUN_STEM}{n}.csv"
        try:
            write_csv(rows, csv_path)  # opened with "x": claims N atomically
            break
        except FileExistsError:
            continue
    png_path = plot_operating_point(rows, directory / f"{RUN_STEM}{n}.png", run_number=n)
    return n, csv_path, png_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--shots", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = measure(args.distances, args.p, args.shots, args.seed)
    n, csv_path, png_path = save_run(rows)
    print(f"run {n}: wrote {csv_path} and {png_path}")


if __name__ == "__main__":
    main()
