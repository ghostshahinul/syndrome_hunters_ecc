"""Render the accuracy, timing and operating-point figures from benchmark CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import CenteredNorm, LinearSegmentedColormap, SymLogNorm
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
import numpy as np

from src.evaluate import RESULTS_DIR, wilson

FIGURES_DIR = Path("figures")
STYLES = {
    "fault_join": ("FaultJoin", "#006e67", "o"),
    "pymatching": ("PyMatching", "#4368a9", "s"),
    "pymatching_correlated": ("Correlated PyMatching", "#a94e16", "^"),
    "union_find": ("Weighted union–find", "#795789", "D"),
}
DISTANCE_COLORS = ["#206493", "#bd581b", "#178062", "#84529a", "#614839"]
PLAIN = FuncFormatter(lambda value, _: f"{value:g}")


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    seen = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for k in ("distance", "rounds", "shots", "num_errors"):
                    row[k] = int(row[k])
                for k in ("p", "logical_error_rate", "us_per_shot"):
                    row[k] = float(row[k])
                row["ler_low95"], row["ler_high95"] = wilson(row["num_errors"], row["shots"])
                for k in ("single_p50_us", "single_p95_us"):
                    row[k] = float(row[k]) if row.get(k) else None
                key = row["decoder"], row["distance"], row["p"]
                if key in seen:
                    raise ValueError(f"duplicate operating point {key}; plot independent runs separately")
                seen.add(key)
                rows.append(row)
    if not rows or len({r["basis"] for r in rows}) != 1:
        raise ValueError("supply results for one memory basis per figure set")
    rounds = {r["rounds"] for r in rows}
    if len(rounds) > 1 and not all(r["rounds"] == r["distance"] for r in rows):
        raise ValueError("select either fixed rounds or rounds equal to distance")
    return rows


def _style(name):
    return STYLES.get(name, (name, "#555555", "o"))


def _legend(fig, items):
    handles = [Line2D([], [], color=color, marker=marker, lw=2, markersize=6, label=name)
               for name, color, marker in items]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, 0.855),
               frameon=False, ncol=min(4, len(items)), columnspacing=1.8, handlelength=1.65)


def _header(fig, rows, title, detail):
    basis = rows[0]["basis"].upper()
    rounds = ("rounds = distance" if all(r["rounds"] == r["distance"] for r in rows)
              else f"{rows[0]['rounds']} rounds")
    counts = {r["shots"] for r in rows}
    shots = f"{counts.pop():,} shots per point" if len(counts) == 1 else "shot counts in CSV"
    fig.text(.06, .955, title, fontsize=20, weight="bold", color="#172d39", va="top")
    fig.text(.06, .89, f"Rotated {basis} memory  ·  {rounds}  ·  {shots}  ·  {detail}",
             fontsize=10.5, color="#435864", va="top")


def _axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#b8c4c9")
    ax.tick_params(color="#a8b6bb", labelcolor="#243945")
    ax.grid(axis="y", which="major", alpha=.22)
    ax.set_axisbelow(True)


def _accuracy_points(ax, pts, xvalues, color, marker, connect=True):
    y = np.array([100 * r["logical_error_rate"] for r in pts])
    lo = np.array([100 * r["ler_low95"] for r in pts])
    hi = np.array([100 * r["ler_high95"] for r in pts])
    x = np.asarray(xvalues)
    have = y > 0
    if connect:
        ax.plot(x, np.where(have, y, np.nan), color=color, lw=1.8)
    if have.any():
        ax.errorbar(x[have], y[have], yerr=[y[have] - lo[have], hi[have] - y[have]],
                    fmt=marker, color=color, capsize=2.5, elinewidth=1.1, ms=5.5)
    if (~have).any():
        ax.scatter(x[~have], hi[~have], marker="v", facecolors="none", edgecolors=color, s=54, zorder=3)


def _save(fig, directory, name, footnote):
    directory.mkdir(parents=True, exist_ok=True)
    fig.text(.06, .035, footnote, color="#435864", fontsize=9, va="bottom")
    path = directory / name
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def plot_threshold(rows: list[dict], out_dir: Path = FIGURES_DIR) -> list[Path]:
    names = list(dict.fromkeys(r["decoder"] for r in rows))
    distances = sorted({r["distance"] for r in rows})
    styles = [(f"d = {d}", DISTANCE_COLORS[i % len(DISTANCE_COLORS)], ["o", "s", "D", "^"][i % 4])
              for i, d in enumerate(distances)]
    cols = min(2, len(names))
    nrows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(12, 4.2 + 2.5 * nrows), squeeze=False,
                             sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.97, top=.77, bottom=.14, hspace=.4, wspace=.22)
    _header(fig, rows, "Logical failure rate across noise levels", "95% Wilson intervals")
    _legend(fig, styles)
    for ax, name in zip(axes.flat, names):
        for d, (_, color, marker) in zip(distances, styles):
            pts = sorted((r for r in rows if r["decoder"] == name and r["distance"] == d), key=lambda r: r["p"])
            _accuracy_points(ax, pts, [100 * r["p"] for r in pts], color, marker)
        ax.set_title(_style(name)[0], loc="left", weight="bold", fontsize=12)
        ax.set_yscale("log")
        ax.set_xticks(sorted({100 * r["p"] for r in rows}))
        ax.xaxis.set_major_formatter(PLAIN)
        ax.yaxis.set_major_formatter(PLAIN)
        _axes(ax)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    fig.supxlabel("Physical error probability per noise channel (%)", y=.077, fontsize=11)
    fig.supylabel("Logical failures per memory experiment (%)", x=.013, fontsize=11)
    note = ("Hollow downward triangles mark the 95% upper endpoint when zero failures were observed."
            if any(r["num_errors"] == 0 for r in rows)
            else "Each failure is a shot with an incorrect logical prediction. Lines connect simulated points.")
    return [_save(fig, out_dir, "logical_error_rates.png", note)]


def _point(rows, p):
    choices = sorted({r["p"] for r in rows})
    chosen = min(choices, key=lambda v: abs(v - .005)) if p is None else p
    pts = [r for r in rows if r["p"] == chosen]
    if not pts:
        raise ValueError(f"no rows at p={chosen}; available values are {choices}")
    return pts, chosen


def plot_latency(rows: list[dict], out_dir: Path = FIGURES_DIR, p=None) -> Path:
    pts, p = _point(rows, p)
    names = list(dict.fromkeys(r["decoder"] for r in pts))
    single = all(r["single_p50_us"] is not None for r in pts)
    fig, axes = plt.subplots(1, 1 + int(single), figsize=(12, 5.7), squeeze=False)
    fig.subplots_adjust(left=.08, right=.97, top=.70, bottom=.20, wspace=.27)
    _header(fig, pts, "How decoding time scales with distance", f"physical error {p:.2%}")
    _legend(fig, [_style(n) for n in names])
    for i, ax in enumerate(axes.flat):
        mode = "single_p50_us" if i else "us_per_shot"
        for index, name in enumerate(names):
            series = sorted((r for r in pts if r["decoder"] == name), key=lambda r: r["distance"])
            _, color, marker = _style(name)
            x = np.array([r["distance"] for r in series], float)
            if i:
                x += .06 * (index - (len(names) - 1) / 2)
            y = [r[mode] for r in series]
            ax.plot(x, y, color=color, marker=marker, lw=2, ms=6)
            if i:
                ax.vlines(x, y, [r["single_p95_us"] for r in series], color=color, lw=1.5)
                ax.scatter(x, [r["single_p95_us"] for r in series], c=color, marker="_")
        ax.set(title="Single call per shot" if i else "Batch decoding", xlabel="Code distance",
               ylabel="Time per shot (µs)", yscale="log", xticks=sorted({r["distance"] for r in pts}))
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=[1, 2, 5]))
        ax.yaxis.set_major_formatter(PLAIN)
        ax.yaxis.set_minor_formatter(NullFormatter())
        _axes(ax)
    return _save(fig, out_dir, "runtime_scaling.png",
                 "Batch points show amortized time. Single-call points show medians with whiskers to p95."
                 if single else "Batch points show amortized time per shot.")


def plot_tradeoff(rows, out_dir=FIGURES_DIR, p=None):
    pts, p = _point(rows, p)
    names = list(dict.fromkeys(r["decoder"] for r in pts))
    distances = sorted({r["distance"] for r in pts})
    fig, axes = plt.subplots(1, len(distances), figsize=(max(9, 4.0 * len(distances)), 5.7),
                             sharey=True, squeeze=False)
    fig.subplots_adjust(left=.08, right=.975, top=.70, bottom=.22, wspace=.2)
    _header(fig, pts, "Accuracy and batch time at one operating point", f"physical error {p:.2%}")
    _legend(fig, [_style(n) for n in names])
    for ax, d in zip(axes.flat, distances):
        for r in pts:
            if r["distance"] == d:
                _, color, marker = _style(r["decoder"])
                _accuracy_points(ax, [r], [r["us_per_shot"]], color, marker, connect=False)
        times = [r["us_per_shot"] for r in pts if r["distance"] == d]
        low, high = min(times), max(times)
        ax.set_title(f"Distance {d}")
        if low > 0 and high / low >= 10:
            ax.set_xscale("log")
            ax.set_xlim(low / 1.3, high * 1.3)
            ax.set_xlabel("Batch time per shot (µs, log scale)")
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=[1, 2, 5] if high / low < 80 else [1]))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_xlim(0, high * 1.13)
            ax.set_xlabel("Batch time per shot (µs)")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.xaxis.set_major_formatter(PLAIN)
        _axes(ax)
    axes.flat[0].set_ylabel("Logical failures (%)")
    return _save(fig, out_dir, "accuracy_runtime.png",
                 "Lower and further left is better. Whiskers show 95% Wilson intervals for the failure rate.")


def plot_baseline_comparison(rows, out_dir=FIGURES_DIR):
    """Compare FaultJoin with the best observed control for each metric and point."""
    controls = {r["decoder"] for r in rows} - {"fault_join"}
    if not controls or not any(r["decoder"] == "fault_join" for r in rows):
        return None
    distances = sorted({r["distance"] for r in rows})
    noise = sorted({r["p"] for r in rows})
    by_point = {}
    for row in rows:
        by_point.setdefault((row["distance"], row["p"]), {})[row["decoder"]] = row

    delta = np.full((len(distances), len(noise)), np.nan)
    ratio = np.full_like(delta, np.nan)
    for i, distance in enumerate(distances):
        for j, p in enumerate(noise):
            group = by_point.get((distance, p), {})
            if "fault_join" not in group or not controls.issubset(group):
                continue
            candidate = group["fault_join"]
            best_rate = min(group[name]["logical_error_rate"] for name in controls)
            fastest = min(group[name]["us_per_shot"] for name in controls)
            delta[i, j] = round(10000 * (candidate["logical_error_rate"] - best_rate), 9)
            if fastest > 0 and candidate["us_per_shot"] > 0:
                ratio[i, j] = candidate["us_per_shot"] / fastest
    if not np.isfinite(delta).any():
        return None
    colors = LinearSegmentedColormap.from_list("relative", ["#195777", "#f7f5ed", "#a64a24"])
    colors.set_bad("#e5e9ea")
    span = max(1., np.nanmax(abs(delta)))
    log_ratio = np.log2(ratio)
    log_span = max(1., np.nanmax(abs(log_ratio))) if np.isfinite(log_ratio).any() else 1.
    fig, axes = plt.subplots(1, 2, figsize=(max(12, 1.5 * len(noise)), max(6.2, 3.3 + .65 * len(distances))))
    fig.subplots_adjust(left=.075, right=.965, top=.71, bottom=.275, wspace=.22)
    _header(fig, rows, "FaultJoin across physical noise levels", "observed baseline at each point")
    panels = [
        (delta, delta, SymLogNorm(linthresh=1., vmin=-span, vmax=span),
         "Excess failures per 10,000 shots", "Negative values favor FaultJoin"),
        (log_ratio, ratio, CenteredNorm(vcenter=0, halfrange=log_span),
         "Batch time relative to the fastest baseline", "Values below 1× favor FaultJoin"),
    ]
    for panel, (ax, (heat, labels, norm, title, guide)) in enumerate(zip(axes, panels)):
        ax.imshow(np.ma.masked_invalid(heat), norm=norm, cmap=colors, aspect="auto", interpolation="nearest")
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=10)
        ax.set(xticks=np.arange(len(noise)), xticklabels=[f"{100*p:g}" for p in noise],
               yticks=np.arange(len(distances)), yticklabels=distances,
               xlabel="Physical error probability per noise channel (%)", ylabel="Code distance")
        ax.set_xticks(np.arange(len(noise) + 1) - .5, minor=True)
        ax.set_yticks(np.arange(len(distances) + 1) - .5, minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="both", length=0, pad=5)
        ax.spines[:].set_visible(False)
        for (i, j), value in np.ndenumerate(labels):
            label, ink = "—", "#53656b"
            if np.isfinite(value):
                label = (f"{value:.2f}×" if panel else
                         (f"{value:+.2f}".rstrip("0").rstrip(".") if value else "0"))
                shade = colors(norm(heat[i, j]))
                ink = "white" if np.dot(shade[:3], [.2126, .7152, .0722]) < .48 else "#142d3a"
            ax.text(j, i, label, ha="center", va="center", color=ink, fontsize=11, fontweight="bold")
        ax.text(0, -.265, guide, transform=ax.transAxes, fontsize=10, color="#314954", fontweight="bold")
    return _save(fig, out_dir, "baseline_comparison.png",
                 "Each metric uses its best observed baseline at the same distance and noise. "
                 "A dash marks an incomplete comparison.")


def default_files():
    """Prefer a local benchmark, otherwise combine the published Z-memory sweeps."""
    local = sorted(RESULTS_DIR.glob("bench_*.csv"))
    if local:
        return local[-1:]
    files = []
    for folder in ("confirm", "noise"):
        for path in sorted((RESULTS_DIR / folder).glob("bench_*.csv"), reverse=True):
            with path.open(newline="", encoding="utf-8") as stream:
                first = next(csv.DictReader(stream), {})
            if first.get("basis") == "z":
                files.append(path)
                break
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    ap.add_argument("--p", type=float, help="noise level for the runtime and trade-off figures")
    args = ap.parse_args()
    files = args.files or default_files()
    if not files:
        ap.error("no benchmark CSV found; run python -m src.evaluate first")
    rows = load_rows(files)
    for path in [*plot_threshold(rows, args.out_dir), plot_latency(rows, args.out_dir, args.p),
                 plot_tradeoff(rows, args.out_dir, args.p), plot_baseline_comparison(rows, args.out_dir)]:
        if path is not None:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
