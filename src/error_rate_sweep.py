"""LER vs physical error rate p for the operating-point decoders, drawn two ways.

For each p, runs src.operating_point.measure (same seeds, batches and single-thread rule as
the operating point) and pools everything into one numbered run:
  results/error_rate_sweep/sweep_run<N>.csv         every (p, d, decoder) row
  results/error_rate_sweep/sweep_run<N>_ours.png    our style: LER % vs p (src.figures.plot_sweep)
  results/error_rate_sweep/sweep_run<N>_paper.png   journal style: LER per round vs p
  results/operating_point/operating_point_run<M>.*  our operating-point figure from the rows at
                                                    p = --op-p (if it is in the sweep)
Nothing is overwritten (N and M count up).

CLI:  python -m src.error_rate_sweep --ps 0.002 0.003 0.004 0.005 0.006 0.007 --batches 10
"""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

from src.figures import PAPER_RC, SURFACE, plot_paper_style, plot_sweep, write_rows
from src.operating_point import OPERATING_POINT_DECODERS, default_factories, measure, save_run

SWEEP_DIR = Path("results") / "error_rate_sweep"
SWEEP_STEM = "sweep_run"


def to_figure_row(r) -> dict:
    """OperatingPoint -> the row schema src.figures reads (d, ci_low, ci_high, ...)."""
    row = asdict(r)
    row["d"] = row.pop("distance")
    row["ci_low"] = row.pop("ler_low")
    row["ci_high"] = row.pop("ler_high")
    return row


def claim_run(directory: Path = SWEEP_DIR) -> tuple[int, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    nums = [int(m.group(1)) for f in directory.glob(f"{SWEEP_STEM}*")
            if (m := re.match(rf"{SWEEP_STEM}(\d+)", f.name))]
    n = max(nums, default=0) + 1
    while (directory / f"{SWEEP_STEM}{n}.csv").exists():
        n += 1
    return n, directory / f"{SWEEP_STEM}{n}.csv"


def meta_for(rows: list[dict], run: int) -> dict:
    first = rows[0]
    return {"cpu": first["cpu"], "power": first["power"], "git_commit": first["git_commit"], "run": run,
            "threads_note": "every decoder single-thread (PyMatching too)"}


def save_figures(rows: list[dict], run: int, directory: Path = SWEEP_DIR) -> tuple[Path, Path]:
    meta = meta_for(rows, run)
    ours = directory / f"{SWEEP_STEM}{run}_ours.png"
    paper = directory / f"{SWEEP_STEM}{run}_paper.png"
    fig = plot_sweep(rows, meta)
    fig.savefig(ours, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    with plt.rc_context(PAPER_RC):
        fig = plot_paper_style(rows, meta)
        fig.savefig(paper, dpi=200, facecolor="white")
    plt.close(fig)
    return ours, paper


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ps", type=float, nargs="+", default=[0.002, 0.003, 0.004, 0.005, 0.006, 0.007])
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--shots", type=int, default=50_000, help="shots per batch")
    ap.add_argument("--batches", type=int, default=10, help="independent batches per point")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--op-p", type=float, default=0.005, help="p whose rows also make an operating-point run")
    ap.add_argument("--decoders", nargs="+", default=list(OPERATING_POINT_DECODERS))
    args = ap.parse_args()

    factories = default_factories(args.decoders)
    points = []
    for p in args.ps:
        print(f"--- p = {p}", flush=True)
        points += measure(args.distances, p, args.shots, args.seed, factories, batches=args.batches, threads=1,
                          log=lambda s, p=p: print(f"p={p} {s}", flush=True))
    rows = [to_figure_row(r) for r in points]
    run, csv_path = claim_run()
    write_rows(rows, csv_path)
    ours, paper = save_figures(rows, run)
    print(f"sweep run {run}: wrote {csv_path}, {ours}, {paper}")
    op_rows = [r for r in points if abs(r.p - args.op_p) < 1e-12]
    if op_rows:
        n, op_csv, op_png = save_run(op_rows)
        print(f"operating point run {n}: wrote {op_csv} and {op_png}")


if __name__ == "__main__":
    main()
