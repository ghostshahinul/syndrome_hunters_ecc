"""Threshold curves and latency-scaling plots from results/*.csv.

CLI:  python -m src.plots                       # newest results file
      python -m src.plots results/bench_X.csv   # specific file(s)
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                for k in ("distance", "rounds", "shots", "num_errors"):
                    r[k] = int(r[k])
                for k in ("p", "logical_error_rate", "ler_stderr", "us_per_shot"):
                    r[k] = float(r[k])
                rows.append(r)
    return rows


def plot_threshold(rows: list[dict], out_dir: Path = FIGURES_DIR) -> list[Path]:
    """One figure per decoder: logical error rate vs p, one line per distance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_decoder = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_decoder[r["decoder"]][r["distance"]].append(r)

    paths = []
    for decoder, by_d in by_decoder.items():
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for d in sorted(by_d):
            pts = sorted(by_d[d], key=lambda r: r["p"])
            ax.errorbar(
                [r["p"] for r in pts], [r["logical_error_rate"] for r in pts],
                yerr=[r["ler_stderr"] for r in pts], marker="o", capsize=3, label=f"d={d}",
            )
        ax.set(xscale="log", yscale="log", xlabel="physical error rate p",
               ylabel="logical error rate per shot", title=f"Threshold: {decoder}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        path = out_dir / f"threshold_{decoder}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_latency(rows: list[dict], out_dir: Path = FIGURES_DIR) -> Path:
    """µs/shot vs distance for every decoder, at the largest p in the data."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p_max = max(r["p"] for r in rows)
    series = defaultdict(list)
    for r in rows:
        if r["p"] == p_max:
            series[r["decoder"]].append((r["distance"], r["us_per_shot"]))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for decoder, pts in sorted(series.items()):
        pts.sort()
        ax.plot([d for d, _ in pts], [t for _, t in pts], marker="o", label=decoder)
    ax.set(yscale="log", xlabel="distance d", ylabel="µs per shot",
           title=f"Decoding latency (p={p_max:g})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    path = out_dir / "latency_vs_distance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    args = ap.parse_args()

    files = args.files or sorted(RESULTS_DIR.glob("bench_*.csv"))[-1:]
    if not files:
        raise SystemExit("no results found; run `python -m src.evaluate` first")
    rows = load_rows(files)
    for path in [*plot_threshold(rows), plot_latency(rows)]:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
