"""Standard figures for every benchmark run, with one style registry for all of them.

Figures (all read the CSV rows written by src/benchmark.py, so they can be regenerated
from any old results file):
  accuracy_runtime     one panel per distance; x = batch µs/shot, y = logical failure %
  accuracy_runtime_sweep  LER vs p (log-log), one line per decoder, one panel per distance
  thread_scaling       µs/shot vs numba threads for UnionFind3D and Hierarchical, with
                       single-thread PyMatching as a horizontal reference line
  fast_path_fraction   share of shots the Cube3D-Local fast path decodes alone vs p
  cube_collapse_story  LER vs d: Cube-Collapse, PyMatching 2D on flat, PyMatching 3D

save_figure() writes results/plots/<name>_<timestamp>.png and <name>_latest.png.
The timestamped file is created exclusively and never overwritten.

CLI (regenerate from saved CSVs):
  python -m src.figures --accuracy results/accuracy_runtime_<ts>.csv --sweep ... --threads ...
"""

from __future__ import annotations

import argparse
import csv
import shutil
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

PLOTS_DIR = Path("results") / "plots"

# ---- style registry ---------------------------------------------------------------------
# Reference categorical palette in its fixed order (dataviz skill, references/palette.md).
# The first four slots are the styles the operating-point figure has always used; new
# decoders take the next free slots. A decoder keeps its color + marker in every figure.
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


@dataclass(frozen=True)
class Style:
    color: str
    marker: str


STYLES: dict[str, Style] = {
    "Cube-Collapse":          Style("#2a78d6", "o"),
    "PyMatching":             Style("#eb6834", "s"),
    "Correlated PyMatching":  Style("#1baf7a", "^"),
    "PyMatching 2D on flat":  Style("#eda100", "D"),
    "UnionFind3D":            Style("#e87ba4", "v"),
    "Hierarchical":           Style("#008300", "P"),
    "Hierarchical (partial)": Style("#6250d6", "X"),
    "Cube3D-Local":           Style("#e34948", "*"),
    # The 8 palette slots are used up; the two MWPM series take neutral inks + unique markers.
    "MWPM3D":                 Style("#52514e", "h"),
    "Hierarchical + MWPM":    Style("#0b0b0b", "p"),
    # UF variants: same hue as UnionFind3D (family), distinct marker (variant).
    "UF3D correlated":        Style("#e87ba4", "d"),
    "UF3D ensemble":          Style("#e87ba4", ">"),
    "UF3D correlated ensemble": Style("#e87ba4", "<"),
}
FALLBACK = Style(MUTED, "o")

# registry / internal names -> display names
DISPLAY_NAMES = {
    "pymatching": "PyMatching",
    "cube_collapse": "Cube-Collapse",
    "union_find_3d": "UnionFind3D",
    "hierarchical": "Hierarchical",
    "hierarchical_partial": "Hierarchical (partial)",
    "cube3d_local": "Cube3D-Local",
    "mwpm_3d": "MWPM3D",
    "hierarchical_mwpm": "Hierarchical + MWPM",
    "uf3d_correlated": "UF3D correlated",
    "uf3d_ensemble": "UF3D ensemble",
}

# ordinal ramp for distance (sequential blue, steps 250 -> 700), plus a marker per d
DISTANCE_COLORS = {3: "#86b6ef", 5: "#3987e5", 7: "#1c5cab", 9: "#0d366b", 11: "#0d366b"}
DISTANCE_MARKERS = {3: "o", 5: "s", 7: "^", 9: "D", 11: "v"}


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def style_for(name: str) -> Style:
    return STYLES.get(display_name(name), FALLBACK)


def order_decoders(names: Iterable[str]) -> list[str]:
    """Registry order first (stable legend order across figures), unknown names after."""
    names = list(dict.fromkeys(names))
    known = [n for n in STYLES if n in names]
    return known + [n for n in names if n not in STYLES]


# ---- CSV helpers --------------------------------------------------------------------------
NUMERIC = {"d", "rounds", "shots", "failures", "threads"}
FLOATS = {"p", "ler", "ci_low", "ci_high", "us_per_shot", "fast_fraction"}


def read_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in NUMERIC & r.keys():
                r[k] = int(r[k]) if r[k] not in ("", None) else None
            for k in FLOATS & r.keys():
                r[k] = float(r[k]) if r[k] not in ("", None) else None
            rows.append(r)
    return rows


def write_rows(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_figure(fig, name: str, directory: Path = PLOTS_DIR, stamp: str | None = None) -> tuple[Path, Path]:
    """Write <name>_<stamp>.png (exclusive; a suffix is added on collision) and copy it to
    <name>_latest.png. Returns (timestamped path, latest path)."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = stamp or timestamp()
    path, k = directory / f"{name}_{stamp}.png", 1
    while path.exists():
        k += 1
        path = directory / f"{name}_{stamp}_{k}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    latest = directory / f"{name}_latest.png"
    shutil.copyfile(path, latest)
    return path, latest


# ---- shared anatomy ---------------------------------------------------------------------
def _axes_style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", which="major", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=11, length=4, color=AXIS, which="both")


def _header(fig, title: str, subtitle: str) -> None:
    fig.text(0.06, 0.985, title, fontsize=22, fontweight="bold", color=INK, va="top")
    fig.text(0.06, 0.915, subtitle, fontsize=12, color=INK_2, va="top")


def _plain(v: float, _pos=None) -> str:
    return f"{v:g}"


def _log_axis_plain(axis, ticks: list[float] | None = None) -> None:
    """Log axis with plain decimal labels (0.1, 0.2, 0.5) instead of 2x10^-1."""
    if ticks:
        axis.set_ticks(ticks)
    axis.set_major_formatter(FuncFormatter(_plain))
    axis.set_minor_formatter(NullFormatter())


def _width(n_panels: int, per_panel: float) -> float:
    return max(11.0, per_panel * n_panels)


def _legend(fig, handles, y: float = 0.875) -> float:
    """Legend row(s) under the subtitle. Returns the `top` to pass to subplots_adjust."""
    ncol = min(len(handles), max(2, int(fig.get_figwidth() // 4.2)))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, y),
               ncol=ncol, frameon=False, fontsize=12, handlelength=2.2, columnspacing=2.0)
    rows = -(-len(handles) // ncol)
    row_h = 0.27 / fig.get_figheight()  # ~0.27 inch per legend row at fontsize 12
    return y - rows * row_h - 0.09


def _footer(fig, lines: list[str]) -> int:
    """Wrapped footer at the bottom of the figure; returns the number of printed lines."""
    width = int(fig.get_figwidth() * 11.0)
    wrapped = [w for line in lines for w in textwrap.wrap(line, width)]
    fig.text(0.06, 0.015, "\n".join(wrapped), fontsize=10.5, color=INK_2, va="bottom", linespacing=1.5)
    return len(wrapped)


def machine_line(meta: dict) -> str:
    from src.machine import short_cpu
    parts = [f"CPU {short_cpu(meta.get('cpu') or None)}"]
    if meta.get("logical_cpus"):
        parts[-1] += f" ({meta['logical_cpus']} logical)"
    if meta.get("power"):
        parts.append(str(meta["power"]))
    if meta.get("threads_note"):
        parts.append(str(meta["threads_note"]))
    if meta.get("backend"):
        parts.append(f"backend {meta['backend']}")
    if meta.get("git_commit"):
        parts.append(f"commit {meta['git_commit']}")
    return "  ·  ".join(parts)


def _subtitle(meta: dict, shots: int | None = None, p: float | None = None) -> str:
    parts = [f"Rotated {str(meta.get('basis', 'z')).upper()} memory",
             str(meta.get("rounds_rule", "rounds = distance"))]
    if shots:
        parts.append(f"{shots:,} shots per point")
    if p is not None:
        parts.append(f"physical error {100 * p:.2f}%")
    parts.append(str(meta.get("noise", "circuit-level noise")))
    if meta.get("run") is not None:
        parts.append(f"run {meta['run']}")
    return "  ·  ".join(parts)


# ---- accuracy vs runtime (the standard figure) -------------------------------------------
def plot_accuracy_runtime(rows: list[dict], meta: dict | None = None, show_multithread: bool = False):
    """rows: decoder, d, p, shots, ler, ci_low, ci_high, us_per_shot, threads.
    Fair speed reporting: by default only single-thread rows are drawn, so every decoder is
    compared with single-thread PyMatching. show_multithread=True adds hollow markers for the
    multi-thread rows (our scaling; see thread_scaling.png)."""
    meta = meta or {}
    if not show_multithread:
        rows = [r for r in rows if (r.get("threads") or 1) == 1]
    decoders = order_decoders(r["decoder"] for r in rows)
    distances = sorted({r["d"] for r in rows})
    fig, axes = plt.subplots(1, len(distances), figsize=(_width(len(distances), 5.4), 6.0),
                             sharey=True, facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    multi = sorted({r["threads"] for r in rows if (r.get("threads") or 1) > 1})
    for ax, d in zip(axes, distances):
        _axes_style(ax)
        pts = [r for r in rows if r["d"] == d]
        for r in pts:
            st = style_for(r["decoder"])
            y = 100 * r["ler"]
            hollow = (r.get("threads") or 1) > 1
            ax.errorbar(r["us_per_shot"], y, yerr=[[y - 100 * r["ci_low"]], [100 * r["ci_high"] - y]],
                        fmt=st.marker, color=st.color, ms=9, mec=st.color if hollow else SURFACE,
                        mfc=SURFACE if hollow else st.color, mew=2 if hollow else 1.5,
                        elinewidth=2, capsize=4, capthick=2, zorder=3)
        ax.set_title(f"Distance {d}", fontsize=14, color=INK, pad=10)
        ax.set_xlim(0, max(r["us_per_shot"] for r in pts) * 1.12)
        ax.set_xlabel("Batch time per shot (µs)", fontsize=12, color=INK_2)
    axes[0].set_ylabel("Logical failures (%)", fontsize=12, color=INK_2)

    first = rows[0]
    _header(fig, "Accuracy and batch time at one operating point",
            _subtitle(meta, first.get("shots"), first.get("p")))
    handles = [Line2D([], [], color=style_for(n).color, marker=style_for(n).marker, ms=9, lw=2.5,
                      label=n, mec=SURFACE) for n in decoders]
    if multi:
        handles.append(Line2D([], [], color=INK_2, marker="o", ms=9, lw=0, mfc=SURFACE, mew=2,
                              label=f"hollow marker = {multi[-1]} threads"))
    top = _legend(fig, handles)
    meta = {**meta, "threads_note": meta.get("threads_note") or
            (f"filled = 1 thread, hollow = {multi[-1]} numba threads" if multi
             else "every decoder single-thread (PyMatching too)")}
    nlines = _footer(fig, ["Lower and further left is better. Whiskers show 95% Wilson intervals for the "
                           "failure rate. Batch time = one decode call over all shots / shots.", machine_line(meta)])
    fig.subplots_adjust(left=0.07, right=0.98, top=top, bottom=0.14 + 0.04 * nlines, wspace=0.18)
    return fig


# ---- LER vs p sweep ---------------------------------------------------------------------
def plot_sweep(rows: list[dict], meta: dict | None = None):
    """rows: decoder, d, p, failures, ler, ci_low, ci_high. Points with 0 failures are omitted."""
    meta = meta or {}
    decoders = order_decoders(r["decoder"] for r in rows)
    distances = sorted({r["d"] for r in rows})
    fig, axes = plt.subplots(1, len(distances), figsize=(_width(len(distances), 5.0), 6.0),
                             sharey=True, facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    omitted = 0
    for ax, d in zip(axes, distances):
        _axes_style(ax)
        ax.grid(True, axis="x", which="major", color=GRID, lw=0.8, zorder=0)
        for n in decoders:
            pts = sorted((r for r in rows if r["d"] == d and r["decoder"] == n and (r.get("threads") or 1) == 1),
                         key=lambda r: r["p"])
            omitted += sum(1 for r in pts if r["failures"] == 0)
            pts = [r for r in pts if r["failures"] > 0]
            if not pts:
                continue
            st = style_for(n)
            x = [100 * r["p"] for r in pts]
            y = [100 * r["ler"] for r in pts]
            lo = [100 * (r["ler"] - r["ci_low"]) for r in pts]
            hi = [100 * (r["ci_high"] - r["ler"]) for r in pts]
            ax.errorbar(x, y, yerr=[lo, hi], fmt="-" + st.marker, color=st.color, lw=2, ms=8,
                        mec=SURFACE, mew=1.2, elinewidth=1.5, capsize=3, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _log_axis_plain(ax.xaxis, sorted({100 * r["p"] for r in rows}))
        _log_axis_plain(ax.yaxis)
        ax.set_title(f"Distance {d}", fontsize=14, color=INK, pad=10)
        ax.set_xlabel("Physical error rate p (%)", fontsize=12, color=INK_2)
    axes[0].set_ylabel("Logical failures (%)", fontsize=12, color=INK_2)
    first = rows[0]
    _header(fig, "Logical failure rate vs physical error rate", _subtitle(meta, first.get("shots")))
    top = _legend(fig, [Line2D([], [], color=style_for(n).color, marker=style_for(n).marker, ms=8, lw=2,
                               label=n, mec=SURFACE) for n in decoders])
    note = f" {omitted} points with zero failures are not drawn." if omitted else ""
    if "Hierarchical" in decoders:
        note += (" Hierarchical's deep tier is UF3D correlated; where the two fail on exactly the same"
                 " shots, the Hierarchical line hides it.")
    _footer(fig, ["Lower is better; a steeper fall with d means more protection from a larger code. "
                  "Whiskers show 95% Wilson intervals." + note, machine_line(meta)])
    fig.subplots_adjust(left=0.07, right=0.98, top=top, bottom=0.22, wspace=0.12)
    return fig


# ---- thread scaling -----------------------------------------------------------------------
def plot_thread_scaling(rows: list[dict], meta: dict | None = None):
    """rows: decoder, d, threads, us_per_shot. PyMatching rows (threads=1) become a
    horizontal reference line in each panel."""
    meta = meta or {}
    distances = sorted({r["d"] for r in rows})
    decoders = [n for n in order_decoders(r["decoder"] for r in rows) if n != "PyMatching"]
    fig, axes = plt.subplots(1, len(distances), figsize=(_width(len(distances), 6.2), 6.0),
                             facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    for ax, d in zip(axes, distances):
        _axes_style(ax)
        ymax = 0.0
        notes = []
        for n in decoders:
            pts = sorted((r for r in rows if r["d"] == d and r["decoder"] == n), key=lambda r: r["threads"])
            if not pts:
                continue
            st = style_for(n)
            xs = [r["threads"] for r in pts]
            ys = [r["us_per_shot"] for r in pts]
            ymax = max(ymax, *ys)
            ax.plot(xs, ys, "-" + st.marker, color=st.color, lw=2, ms=8, mec=SURFACE, mew=1.2, zorder=3)
            notes.append(f"{n}: {ys[0] / ys[-1]:.1f}× faster at {xs[-1]} threads")
        pm = [r for r in rows if r["d"] == d and r["decoder"] == "PyMatching"]
        if pm:
            y = pm[0]["us_per_shot"]
            ymax = max(ymax, y)
            ax.axhline(y, color=style_for("PyMatching").color, lw=1.5, ls="--", zorder=2)
            ax.annotate("PyMatching, 1 thread", (1, y), textcoords="offset points", xytext=(4, 4),
                        color=INK_2, fontsize=10.5)
        if notes:
            ax.text(0.97, 0.72, "\n".join(notes), transform=ax.transAxes, ha="right", va="top",
                    fontsize=10.5, color=INK_2, linespacing=1.6)
        threads = sorted({r["threads"] for r in rows if r["decoder"] != "PyMatching"})
        ax.set_xscale("log", base=2)
        ax.set_xticks(threads)
        ax.set_xticklabels([str(t) for t in threads])
        ax.set_ylim(0, ymax * 1.15)
        ax.set_title(f"Distance {d}", fontsize=14, color=INK, pad=10)
        ax.set_xlabel("numba threads", fontsize=12, color=INK_2)
    axes[0].set_ylabel("Batch time per shot (µs)", fontsize=12, color=INK_2)
    first = rows[0]
    _header(fig, "Parallel speed-up over shots", _subtitle(meta, first.get("shots"), first.get("p")))
    handles = [Line2D([], [], color=style_for(n).color, marker=style_for(n).marker, ms=8, lw=2,
                      label=n, mec=SURFACE) for n in decoders]
    handles.append(Line2D([], [], color=style_for("PyMatching").color, lw=1.5, ls="--",
                          label="PyMatching (single-thread reference)"))
    top = _legend(fig, handles)
    _footer(fig, ["Lower is better. Shots are independent, so the batch is split across threads "
                  "(numba prange); labels give the speed-up over 1 thread. This shows how our decoders "
                  "scale, not a like-for-like race: PyMatching is drawn single-threaded, but it can also "
                  "be run on separate shot batches in parallel processes.", machine_line(meta)])
    fig.subplots_adjust(left=0.08, right=0.97, top=top, bottom=0.2, wspace=0.18)
    return fig


# ---- fast-path fraction -------------------------------------------------------------------
def plot_fast_path_fraction(rows: list[dict], meta: dict | None = None):
    """rows: d, p, fast_fraction (Hierarchical whole-shot gate)."""
    meta = meta or {}
    distances = sorted({r["d"] for r in rows})
    fig, ax = plt.subplots(figsize=(11, 6.4), facecolor=SURFACE)
    _axes_style(ax)
    for d in distances:
        pts = sorted((r for r in rows if r["d"] == d), key=lambda r: r["p"])
        xs = [100 * r["p"] for r in pts]
        ys = [100 * r["fast_fraction"] for r in pts]
        c = DISTANCE_COLORS.get(d, INK_2)
        ax.plot(xs, ys, "-" + DISTANCE_MARKERS.get(d, "o"), color=c, lw=2, ms=8, mec=SURFACE, mew=1.2,
                label=f"d = {d}", zorder=3)
    ax.set_xscale("log")
    _log_axis_plain(ax.xaxis, sorted({100 * r["p"] for r in rows}))
    ax.set_ylim(0, 105)
    ax.set_xlabel("Physical error rate p (%)", fontsize=12, color=INK_2)
    ax.set_ylabel("Shots decoded by the fast path alone (%)", fontsize=12, color=INK_2)
    first = rows[0]
    K = first.get("K") or 6
    fig.text(0.08, 0.975, "How often the fast path finishes the job", fontsize=20, fontweight="bold",
             color=INK, va="top")
    fig.text(0.08, 0.915, "\n".join(textwrap.wrap(
             _subtitle(meta, first.get("shots")) + f"  ·  whole-shot gate, K = {K}", 140)),
             fontsize=11, color=INK_2, va="top")
    ax.legend(frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _footer(fig, ["Includes shots with no detection events. The rest go to UnionFind3D.",
                  machine_line(meta)])
    fig.subplots_adjust(left=0.1, right=0.87, top=0.84, bottom=0.2)
    return fig


# ---- Cube-Collapse story ------------------------------------------------------------------
STORY_DECODERS = ("Cube-Collapse", "PyMatching 2D on flat", "PyMatching")


def plot_cube_collapse_story(rows: list[dict], meta: dict | None = None):
    """rows: decoder, d, ler, ci_low, ci_high (one p). PyMatching = full 3D matching."""
    meta = meta or {}
    fig, ax = plt.subplots(figsize=(11, 6.4), facecolor=SURFACE)
    _axes_style(ax)
    labels = {"PyMatching": "PyMatching 3D (full space-time graph)",
              "PyMatching 2D on flat": "PyMatching 2D on the flattened syndrome",
              "Cube-Collapse": "Cube-Collapse (flatten, then local rules)"}
    distances = sorted({r["d"] for r in rows})
    for n in STORY_DECODERS:
        pts = sorted((r for r in rows if r["decoder"] == n and (r.get("threads") or 1) == 1), key=lambda r: r["d"])
        if not pts:
            continue
        st = style_for(n)
        xs = [r["d"] for r in pts]
        ys = [100 * r["ler"] for r in pts]
        ax.errorbar(xs, ys, yerr=[[100 * (r["ler"] - r["ci_low"]) for r in pts],
                                  [100 * (r["ci_high"] - r["ler"]) for r in pts]],
                    fmt="-" + st.marker, color=st.color, lw=2, ms=9, mec=SURFACE, mew=1.2,
                    elinewidth=1.5, capsize=3, label=labels[n], zorder=3)
        ax.annotate(f"{ys[-1]:.2f}%", (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, -4),
                    fontsize=11, color=INK_2)
    ax.set_xticks(distances)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Code distance d", fontsize=12, color=INK_2)
    ax.set_ylabel("Logical failures (%)", fontsize=12, color=INK_2)
    first = rows[0]
    fig.text(0.08, 0.975, "Flattening time throws away the information", fontsize=20,
             fontweight="bold", color=INK, va="top")
    fig.text(0.08, 0.915, "\n".join(textwrap.wrap(_subtitle(meta, first.get("shots"), first.get("p")), 140)),
             fontsize=11, color=INK_2, va="top")
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    _footer(fig, ["A working decoder's failure rate falls as d grows. After flattening, even exact "
                  "matching gets worse with d.", machine_line(meta)])
    fig.subplots_adjust(left=0.1, right=0.9, top=0.84, bottom=0.2)
    return fig


# ---- MWPM third tier: accuracy vs cost --------------------------------------------------
def plot_mwpm_tier(rows: list[dict], meta: dict | None = None):
    """rows from scripts/m5_mwpm.py at one p: decoder, d, ler, ci_low, ci_high, us_per_shot,
    and for "Hierarchical + MWPM" the threshold and routed_fraction. One panel per d;
    the three-tier points (one per threshold) are joined by a line, labelled with T."""
    meta = meta or {}
    distances = sorted({r["d"] for r in rows})
    fig, axes = plt.subplots(1, len(distances), figsize=(_width(len(distances), 5.4), 6.0),
                             sharey=False, facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    order = ["PyMatching", "UnionFind3D", "Hierarchical", "Hierarchical + MWPM"]
    for ax, d in zip(axes, distances):
        _axes_style(ax)
        for n in order:
            pts = [r for r in rows if r["d"] == d and r["decoder"] == n]
            if not pts:
                continue
            st = style_for(n)
            pts.sort(key=lambda r: r["us_per_shot"])
            xs = [r["us_per_shot"] for r in pts]
            ys = [100 * r["ler"] for r in pts]
            ax.errorbar(xs, ys, yerr=[[100 * (r["ler"] - r["ci_low"]) for r in pts],
                                      [100 * (r["ci_high"] - r["ler"]) for r in pts]],
                        fmt=("-" if len(pts) > 1 else "") + st.marker, color=st.color, ms=8, lw=1.5,
                        mec=SURFACE, mew=1.2, elinewidth=1.5, capsize=3, zorder=3)
            if n == "Hierarchical + MWPM":
                labels: dict[tuple[float, float], tuple[list[str], float]] = {}
                for r in pts:  # thresholds that route the same shots share one label
                    key = (round(r["us_per_shot"], 2), round(100 * r["ci_high"], 4))
                    ts, frac = labels.get(key, ([], float(r["routed_fraction"])))
                    labels[key] = (ts + [str(int(float(r["threshold"])))], frac)
                for (x, y_top), (ts, frac) in labels.items():
                    share = f"{100 * frac:.0f}%" if frac >= 0.01 else f"{100 * frac:.1f}%"
                    ax.annotate(f"T={','.join(ts)}\n{share}", (x, y_top), textcoords="offset points",
                                xytext=(0, 4), ha="center", va="bottom", fontsize=9, color=INK_2,
                                linespacing=1.1)
        ax.set_xscale("log")
        _log_axis_plain(ax.xaxis)
        ymax = max(100 * r["ci_high"] for r in rows if r["d"] == d)
        ax.set_ylim(0, ymax * 1.25)
        ax.set_title(f"Distance {d}", fontsize=14, color=INK, pad=10)
        ax.set_xlabel("Time per shot (µs, log scale)", fontsize=12, color=INK_2)
    axes[0].set_ylabel("Logical failures (%)", fontsize=12, color=INK_2)
    first = rows[0]
    _header(fig, "A third tier: exact matching for UF's largest clusters",
            _subtitle(meta, first.get("shots"), first.get("p")))
    handles = [Line2D([], [], color=style_for(n).color, marker=style_for(n).marker, ms=8, lw=1.5,
                      label=n, mec=SURFACE) for n in order]
    top = _legend(fig, handles)
    _footer(fig, ["Shots whose largest UnionFind3D cluster has more than T nodes are re-decoded by "
                  "MWPM3D (our exact matching, pure Python); labels give T and the share of shots routed. "
                  "Time = Hierarchical batch time + the routed shots' MWPM time, single thread. "
                  "Whiskers show 95% Wilson intervals.", machine_line(meta)])
    fig.subplots_adjust(left=0.07, right=0.98, top=top, bottom=0.24, wspace=0.2)
    return fig


# ---- ensemble: accuracy vs number of variants ---------------------------------------------
def plot_ensemble_vs_m(rows: list[dict], meta: dict | None = None):
    """rows: decoder in {PyMatching, Correlated PyMatching, UF3D ensemble, UF3D correlated
    ensemble}, d, M ('' for the references), ler, ci_low, ci_high, us_per_shot.
    One panel per d: LER vs M for the ensembles; the references are horizontal lines with
    their 95% band. Labels give single-thread µs/shot."""
    meta = meta or {}
    distances = sorted({int(r["d"]) for r in rows})
    fig, axes = plt.subplots(1, len(distances), figsize=(_width(len(distances), 5.2), 6.0),
                             facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    ens_names = ["UF3D ensemble", "UF3D correlated ensemble"]
    refs = ["PyMatching", "Correlated PyMatching"]
    for ax, d in zip(axes, distances):
        _axes_style(ax)
        Ms = sorted({int(float(r["M"])) for r in rows if r["decoder"] in ens_names and int(r["d"]) == d})
        for n in refs:
            pts = [r for r in rows if r["decoder"] == n and int(r["d"]) == d]
            if not pts:
                continue
            r = pts[0]
            col = style_for(n).color
            ax.axhline(100 * r["ler"], color=col, lw=1.6, ls="--", zorder=2)
            ax.axhspan(100 * r["ci_low"], 100 * r["ci_high"], color=col, alpha=0.12, lw=0, zorder=1)
            ax.annotate(f"{n} ({r['us_per_shot']:.1f} µs)", (1.0, 100 * r["ler"]),
                        xycoords=("axes fraction", "data"), textcoords="offset points",
                        xytext=(-4, 4), ha="right", fontsize=9.5, color=INK_2)
        for n in ens_names:
            pts = sorted((r for r in rows if r["decoder"] == n and int(r["d"]) == d), key=lambda r: int(float(r["M"])))
            if not pts:
                continue
            st = style_for(n)
            xs = [int(float(r["M"])) for r in pts]
            ys = [100 * r["ler"] for r in pts]
            ax.errorbar(xs, ys, yerr=[[100 * (r["ler"] - r["ci_low"]) for r in pts],
                                      [100 * (r["ci_high"] - r["ler"]) for r in pts]],
                        fmt=("-" if n.startswith("UF3D correlated") else "--") + st.marker, color=st.color,
                        lw=2, ms=8, mec=SURFACE, mew=1.2, elinewidth=1.5, capsize=3, zorder=3)
            if n.startswith("UF3D correlated"):
                for x, y, r in zip(xs, ys, pts):
                    ax.annotate(f"{r['us_per_shot']:.0f} µs", (x, y), textcoords="offset points",
                                xytext=(0, -16), ha="center", fontsize=9, color=INK_2)
        ax.set_xticks(Ms)
        ax.set_ylim(bottom=0)
        ax.set_title(f"Distance {d}", fontsize=14, color=INK, pad=10)
        ax.set_xlabel("UF variants per shot (M)", fontsize=12, color=INK_2)
    axes[0].set_ylabel("Logical failures (%)", fontsize=12, color=INK_2)
    first = rows[0]
    _header(fig, "Ensemble UF: more variants per shot, fewer failures",
            _subtitle(meta, first.get("shots"), first.get("p")))
    handles = [Line2D([], [], color=style_for(n).color, marker=style_for(n).marker, ms=8, lw=2,
                      ls="-" if n.startswith("UF3D correlated") else "--", label=n, mec=SURFACE)
               for n in ens_names]
    handles += [Line2D([], [], color=style_for(n).color, lw=1.6, ls="--", label=f"{n} (reference, 95% band)")
                for n in refs]
    top = _legend(fig, handles)
    _footer(fig, ["Lower is better. Each shot is decoded by M UF variants (perturbed weights, mixed growth "
                  "order); the correction that is cheapest under the original weights wins. All times are "
                  "single-thread µs per shot, so they compare directly with PyMatching.", machine_line(meta)])
    fig.subplots_adjust(left=0.07, right=0.98, top=top, bottom=0.22, wspace=0.18)
    return fig


FIGURES: dict[str, Callable] = {
    "accuracy_runtime": plot_accuracy_runtime,
    "accuracy_runtime_sweep": plot_sweep,
    "thread_scaling": plot_thread_scaling,
    "fast_path_fraction": plot_fast_path_fraction,
    "cube_collapse_story": plot_cube_collapse_story,
    "mwpm_tier": plot_mwpm_tier,
    "ensemble_vs_M": plot_ensemble_vs_m,
}


def meta_from_rows(rows: list[dict]) -> dict:
    r = rows[0]
    meta = {k: r.get(k) for k in ("cpu", "power", "logical_cpus", "git_commit", "basis",
                                  "rounds_rule", "noise", "run") if r.get(k) not in (None, "")}
    backends = {x.get("backend") for x in rows}
    meta["backend"] = "numba" if "numba" in backends else r.get("backend")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accuracy", type=Path, help="accuracy_runtime_<ts>.csv")
    ap.add_argument("--sweep", type=Path, help="sweep_<ts>.csv")
    ap.add_argument("--threads", type=Path, help="thread_scaling_<ts>.csv")
    ap.add_argument("--mwpm", type=Path, help="mwpm/mwpm_tier_<ts>.csv (one figure per p)")
    args = ap.parse_args()
    out = []
    if args.accuracy:
        rows = read_rows(args.accuracy)
        meta = meta_from_rows(rows)
        out.append(save_figure(plot_accuracy_runtime(rows, meta), "accuracy_runtime"))
        out.append(save_figure(plot_cube_collapse_story(rows, meta), "cube_collapse_story"))
    if args.sweep:
        rows = read_rows(args.sweep)
        meta = meta_from_rows(rows)
        out.append(save_figure(plot_sweep(rows, meta), "accuracy_runtime_sweep"))
        fp = [r for r in rows if r["decoder"] == "Hierarchical" and r.get("fast_fraction") is not None]
        if fp:
            out.append(save_figure(plot_fast_path_fraction(fp, meta), "fast_path_fraction"))
    if args.threads:
        rows = read_rows(args.threads)
        out.append(save_figure(plot_thread_scaling(rows, meta_from_rows(rows)), "thread_scaling"))
    if args.mwpm:
        rows = read_rows(args.mwpm)
        meta = {**meta_from_rows(rows), "backend": "numba + Python (MWPM)", "threads_note": "1 thread"}
        for p in sorted({r["p"] for r in rows}):
            sub = [r for r in rows if r["p"] == p]
            out.append(save_figure(plot_mwpm_tier(sub, meta), f"mwpm_tier_p{100 * p:g}".replace(".", "_")))
    for stamped, latest in out:
        print(f"wrote {stamped} (and {latest.name})")


if __name__ == "__main__":
    main()
