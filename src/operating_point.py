"""Accuracy vs batch time at one operating point (circuit-level rotated memory).

For each distance, `batches` independent batches of `shots` shots are sampled (seeds
seed, seed+1, ...). Every decoder decodes the same batches. Per decoder and distance:
  - logical failure rate pooled over all batches (95% Wilson interval);
  - batch time per shot = one decode call over a batch / shots, averaged over the batches,
    with the fastest and slowest batch kept as a spread (decoder construction and warm-up
    excluded).  
numba decoders (UnionFind3D, Hierarchical) run on `threads` threads (default 1, the fair
comparison with single-threaded PyMatching). The machine fingerprint goes into every row.

Default decoders: PyMatching, Correlated PyMatching, UnionFind3D, UF3D correlated,
Hierarchical (whose deep tier is UF3D correlated). Cube-Collapse, PyMatching 2D on flat and
Hierarchical (partial) can be added with --decoders.

CLI:  python -m src.operating_point --distances 3 5 7 9 --p 0.005 --shots 50000 --batches 5
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
from src.figures import display_name, machine_line, style_for
from src.machine import fingerprint

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
        out[display_name(name)] = lambda dem, cls=cls: cls(dem).decode_batch
    return out


OPERATING_POINT_DECODERS = ("PyMatching", "Correlated PyMatching", "UnionFind3D", "UF3D correlated",
                            "Hierarchical")


def available_factories() -> dict[str, Callable[[stim.DetectorErrorModel], DecodeFn]]:
    """Every decoder the operating point can run, by display name."""
    return {**DECODER_FACTORIES, **implemented_registry_decoders()}


def default_factories(names=OPERATING_POINT_DECODERS) -> dict[str, Callable[[stim.DetectorErrorModel], DecodeFn]]:
    avail = available_factories()
    missing = [n for n in names if n not in avail]
    if missing:
        raise KeyError(f"unknown decoder(s) {missing}; choose from {list(avail)}")
    return {n: avail[n] for n in names}


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
    us_per_shot: float          # mean over batches of (batch decode time / shots per batch)
    batches: int = 1
    shots_per_batch: int = 0
    us_min: float = 0.0         # fastest batch
    us_max: float = 0.0         # slowest batch
    threads: int = 1            # numba threads (PyMatching is always single-threaded)
    backend: str = ""
    cpu: str = ""
    cpu_mhz: str = ""
    power: str = ""
    git_commit: str = ""


NUMBA_DECODERS = {"UnionFind3D", "UF3D correlated", "UF3D ensemble", "Hierarchical", "Hierarchical (partial)",
                  "FaultJoin"}


def measure(distances=(3, 5, 7, 9), p: float = 0.005, shots: int = 50_000, seed: int = 0,
            factories: dict | None = None, log: Callable[[str], None] | None = print,
            batches: int = 1, threads: int = 1) -> list[OperatingPoint]:
    """`shots` per batch; `batches` independent batches per distance (seeds seed + b)."""
    import numba

    from src.decoders.cube3d.union_find import set_threads

    factories = factories or default_factories()
    fp = fingerprint()
    old_threads = numba.get_num_threads()
    set_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    rows = []
    try:
        for d in distances:
            circuit = surface_code_circuit(d, p)
            dem = detector_error_model(circuit)
            sampler = circuit.compile_detector_sampler(seed=seed)
            decoders = {label: factory(dem) for label, factory in factories.items()}
            fails = {label: 0 for label in decoders}
            times = {label: [] for label in decoders}
            for b in range(batches):
                if b:
                    sampler = circuit.compile_detector_sampler(seed=seed + b)
                dets, obs = sampler.sample(shots, separate_observables=True)
                for label, decode in decoders.items():
                    if b == 0:
                        # warm-up (untimed): numba JIT, first-call allocation, caches. 2,000 shots so
                        # lazily compiled paths (e.g. FaultJoin's UF stage, rare at d=3) compile here
                        decode(dets[:2000])
                    t0 = time.perf_counter()
                    pred = decode(dets)
                    times[label].append(1e6 * (time.perf_counter() - t0) / shots)
                    fails[label] += int(np.any(pred != obs, axis=1).sum())
            total = shots * batches
            for label in decoders:
                k = fails[label]
                lo, hi = wilson_interval(k, total)
                ts = times[label]
                numba_dec = label in NUMBA_DECODERS
                row = OperatingPoint(label, d, d, p, total, k, k / total, lo, hi, float(np.mean(ts)),
                                     batches, shots, float(min(ts)), float(max(ts)),
                                     threads if numba_dec else 1,
                                     "numba" if numba_dec else "python/C++",
                                     fp["cpu"], fp["cpu_mhz"], fp["power"], fp["git_commit"])
                rows.append(row)
                if log:
                    log(f"d={d} {label:<24} failures={k:>6}/{total}  LER={100 * row.ler:.3f}% "
                        f"[{100 * lo:.3f}, {100 * hi:.3f}]  {row.us_per_shot:.2f} us/shot "
                        f"(batches {row.us_min:.2f}-{row.us_max:.2f})")
    finally:
        set_threads(old_threads)
    return rows


# ---- plot ------------------------------------------------------------------------------
# Reference categorical palette, fixed order (dataviz skill, references/palette.md),
# plus a distinct marker per decoder so identity never relies on colour alone.
# Per-decoder styles now come from the shared registry in src/figures.py (STYLES).
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
    style = {name: (style_for(name).color, style_for(name).marker) for name in decoders}  # shared registry
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
        ax.set_xlim(0, max(r.us_per_shot for r in pts) * 1.15)
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
    shots_label = (f"{first.batches} batches × {first.shots_per_batch:,} shots per point"
                   if first.batches > 1 else f"{first.shots:,} shots per point")
    fig.text(0.06, 0.915, f"Rotated Z memory  ·  rounds = distance  ·  {shots_label}"
             f"  ·  physical error {100 * first.p:.2f}%  ·  circuit-level noise{run_label}",
             fontsize=12, color=INK_2, va="top")
    handles = [Line2D([], [], color=style[n][0], marker=style[n][1], ms=9, lw=2.5, label=n)
               for n in decoders]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, 0.875), ncol=len(decoders),
               frameon=False, fontsize=12, handlelength=2.2, columnspacing=2.2)
    numba_threads = sorted({r.threads for r in rows if r.backend == "numba"})
    batch_note = f" Time = mean over {first.batches} batches." if first.batches > 1 else ""
    meta = {"cpu": first.cpu or None, "power": first.power, "git_commit": first.git_commit,
            "threads_note": (f"numba decoders on {numba_threads[-1]} thread(s), PyMatching 1 thread"
                             if numba_threads else "")}
    fig.text(0.06, 0.015, "Lower and further left is better. Whiskers show 95% Wilson intervals "
             "for the failure rate. Batch time = one decode call over all shots / shots." + batch_note
             + "\n" + machine_line(meta), fontsize=10.5, color=INK_2, linespacing=1.5)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.71, bottom=0.2, wspace=0.18)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_operating_point_uncertainty(rows: list[OperatingPoint], path: Path, run_number: int | None = None,
                                     decoders=("PyMatching", "Correlated PyMatching", "UnionFind3D",
                                               "Hierarchical")) -> Path:
    """The operating point with its uncertainty drawn as boxes: height = 95% Wilson interval of
    the failure rate, width = fastest to slowest batch; marker at (mean time, LER). Each panel
    has its own y range so the intervals are visible, and each box is labelled with its interval."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    from src.figures import SLIDE_COLORS

    rows = [r for r in rows if r.decoder in decoders]
    distances = sorted({r.distance for r in rows})
    first = rows[0]
    fig, axes = plt.subplots(1, len(distances), figsize=(5.6 * len(distances), 6.4), facecolor=SURFACE,
                             squeeze=False)
    axes = axes[0]
    for ax, d in zip(axes, distances):
        ax.set_facecolor(SURFACE)
        pts = [r for r in rows if r.distance == d]
        lo, hi = 100 * min(r.ler_low for r in pts), 100 * max(r.ler_high for r in pts)
        span = hi - lo
        xmax = max(r.us_max for r in pts) * 1.15
        for r in pts:
            color = SLIDE_COLORS.get(r.decoder, MUTED)
            y0, y1 = 100 * r.ler_low, 100 * r.ler_high
            ax.add_patch(Rectangle((r.us_min, y0), r.us_max - r.us_min, y1 - y0, facecolor=color, alpha=0.22,
                                   edgecolor=color, lw=1.6, zorder=2))
            ax.plot(r.us_per_shot, 100 * r.ler, marker=style_for(r.decoder).marker, color=color, ms=9,
                    mec=SURFACE, mew=1.2, zorder=4)
            ax.annotate(f"{100 * r.ler:.3f}%\n[{y0:.3f}, {y1:.3f}]", (r.us_per_shot, y1),
                        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9.5, color=color, fontweight="bold", linespacing=1.15, zorder=5)
        ax.set_xlim(0, xmax)
        ax.set_ylim(lo - 0.12 * span, hi + 0.32 * span)
        ax.set_title(f"Distance {d}", fontsize=14, color=INK, pad=10)
        ax.set_xlabel("Batch time per shot (µs)", fontsize=12, color=INK_2)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK_2, labelsize=11, length=4, color=AXIS)
    axes[0].set_ylabel("Logical failures per shot (%)", fontsize=12, color=INK_2)

    fig.text(0.05, 0.985, "Operating point with its range of uncertainty", fontsize=22, fontweight="bold",
             color=INK, va="top")
    run_label = f"  ·  run {run_number}" if run_number is not None else ""
    fig.text(0.05, 0.915, f"Rotated Z memory  ·  rounds = distance  ·  {first.batches} batches × "
             f"{first.shots_per_batch:,} shots per point  ·  physical error {100 * first.p:.2f}%  ·  "
             f"circuit-level noise{run_label}", fontsize=12, color=INK_2, va="top")
    handles = [Line2D([], [], color=SLIDE_COLORS.get(n, MUTED), marker=style_for(n).marker, ms=9, lw=0,
                      mec=SURFACE, label=n) for n in decoders]
    handles.append(Patch(facecolor=INK_2, alpha=0.22, edgecolor=INK_2,
                         label="box: height = 95% CI, width = fastest–slowest batch"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 0.875), ncol=len(handles),
               frameon=False, fontsize=12, handlelength=1.6, columnspacing=2.0)
    meta = {"cpu": first.cpu or None, "power": first.power, "git_commit": first.git_commit,
            "threads_note": "every decoder single-thread (PyMatching too)"}
    fig.text(0.05, 0.015, "Lower and further left is better. Each panel has its own vertical scale. Label = "
             "failure rate and its 95% Wilson interval. Hierarchical's deep tier is UF3D correlated (not drawn; "
             "at d = 5–9 its failure rate is within 0.002 points of Hierarchical's).\n" + machine_line(meta),
             fontsize=10.5, color=INK_2, linespacing=1.5)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.75, bottom=0.17, wspace=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


DARK = {"bg": "#0f1115", "panel": "#161a21", "ink": "#f2f2f0", "ink2": "#b9bcc4", "grid": "#2a303b",
        "axis": "#4a5160"}
DARK_COLORS = {"PyMatching": "#ff8a4c", "Correlated PyMatching": "#3ddc97", "UnionFind3D": "#f49ac1",
               "UF3D correlated": "#c77dff", "Hierarchical": "#4ea8ff", "FaultJoin": "#ffd166"}


def plot_operating_point_dark(rows: list[OperatingPoint], path: Path, run_number: int | None = None,
                              title: str = "Every team decoder on the same shots") -> Path:
    """Dark-theme slide version of the operating-point figure: x = batch time per shot (mean over
    batches), y = logical failure %, with 95% Wilson whiskers; larger type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    decoders = list(dict.fromkeys(r.decoder for r in rows))
    distances = sorted({r.distance for r in rows})
    first = rows[0]
    fig, axes = plt.subplots(1, len(distances), figsize=(5.8 * len(distances), 7.4), sharey=True,
                             facecolor=DARK["bg"], squeeze=False)
    axes = axes[0]
    for ax, d in zip(axes, distances):
        ax.set_facecolor(DARK["panel"])
        pts = [r for r in rows if r.distance == d]
        for r in pts:
            color = DARK_COLORS.get(r.decoder, DARK["ink2"])
            y = 100 * r.ler
            big = r.decoder == "Hierarchical"
            ax.errorbar(r.us_per_shot, y, yerr=[[y - 100 * r.ler_low], [100 * r.ler_high - y]],
                        fmt=style_for(r.decoder).marker, color=color, ms=14 if big else 11, mec=DARK["panel"],
                        mew=1.5, elinewidth=2.2, capsize=5, capthick=2.2, zorder=4 if big else 3)
        ax.set_title(f"Distance {d}", fontsize=20, color=DARK["ink"], pad=12)
        ax.set_xlim(0, max(r.us_per_shot for r in pts) * 1.15)
        ax.set_xlabel("Batch time per shot (µs)", fontsize=17, color=DARK["ink2"])
        ax.grid(True, color=DARK["grid"], lw=0.9, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(DARK["axis"])
        ax.tick_params(colors=DARK["ink2"], labelsize=14, length=5, color=DARK["axis"])
    axes[0].set_ylabel("Logical failures per shot (%)", fontsize=17, color=DARK["ink2"])
    fig.text(0.045, 0.975, title, fontsize=28, fontweight="bold", color=DARK["ink"], va="top")
    run_label = f"  ·  run {run_number}" if run_number is not None else ""
    fig.text(0.045, 0.905, f"Rotated Z memory  ·  circuit-level p = {100 * first.p:.1f}%  ·  rounds = d  ·  "
             f"{first.batches} × {first.shots_per_batch:,} shots per point, the same shots for every decoder  ·  "
             f"single thread{run_label}", fontsize=15, color=DARK["ink2"], va="top")
    handles = [Line2D([], [], color=DARK_COLORS.get(n, DARK["ink2"]), marker=style_for(n).marker, ms=12, lw=0,
                      mec=DARK["panel"], label=n) for n in decoders]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.04, 0.855), ncol=len(handles), frameon=False,
               fontsize=16, handletextpad=0.4, columnspacing=1.8, labelcolor=DARK["ink"])
    meta = {"cpu": first.cpu or None, "power": first.power, "git_commit": first.git_commit,
            "threads_note": "every decoder single-thread (PyMatching too)"}
    fig.text(0.045, 0.02, "Lower and further left is better. x = batch time per shot (mean over "
             f"{first.batches} batches); y = logical failures, whiskers = 95% Wilson interval. "
             "FaultJoin is a teammate decoder (branch shahi_2).\n" + machine_line(meta),
             fontsize=12.5, color=DARK["ink2"], linespacing=1.5)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.71, bottom=0.2, wspace=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=DARK["bg"])
    plt.close(fig)
    return path


def markdown_table(rows: list[OperatingPoint]) -> str:
    """LER [95% CI] and batch µs/shot per decoder (rows) and distance (columns)."""
    decoders = list(dict.fromkeys(r.decoder for r in rows))
    distances = sorted({r.distance for r in rows})
    at = {(r.decoder, r.distance): r for r in rows}
    out = ["| Decoder | " + " | ".join(f"d = {d}" for d in distances) + " |", "|---|" + "---|" * len(distances)]
    for n in decoders:
        cells = []
        for d in distances:
            r = at.get((n, d))
            cells.append("–" if r is None else f"{100 * r.ler:.3f}% [{100 * r.ler_low:.3f}, {100 * r.ler_high:.3f}]; "
                         f"{r.us_per_shot:.2f} µs")
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(out)


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
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--shots", type=int, default=50_000, help="shots per batch")
    ap.add_argument("--batches", type=int, default=5, help="independent batches per point")
    ap.add_argument("--threads", type=int, default=1, help="numba threads for our decoders")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decoders", nargs="+", default=list(OPERATING_POINT_DECODERS),
                    help=f"display names; available: {list(DECODER_FACTORIES)} + registry decoders")
    args = ap.parse_args()
    rows = measure(args.distances, args.p, args.shots, args.seed, default_factories(args.decoders),
                   batches=args.batches, threads=args.threads)
    n, csv_path, png_path = save_run(rows)
    print(f"run {n}: wrote {csv_path} and {png_path}")


if __name__ == "__main__":
    main()
