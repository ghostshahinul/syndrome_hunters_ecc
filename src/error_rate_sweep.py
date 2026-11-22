"""LER vs physical error rate p for the operating-point decoders, drawn two ways.

For each p, runs src.operating_point.measure (same seeds, batches and single-thread rule as
the operating point) and pools everything into one numbered run:
  results/error_rate_sweep/sweep_run<N>.csv         every (p, d, decoder) row
  results/error_rate_sweep/sweep_run<N>_ours.png    our style: LER % vs p (src.figures.plot_sweep)
  results/error_rate_sweep/sweep_run<N>_paper.png   journal style: LER per round vs p
  results/error_rate_sweep/sweep_run<N>_slides.png  projection: d = 5/7/9, 4 decoders, large type
  results/operating_point/operating_point_run<M>.*  our operating-point figure from the rows at
                                                    p = --op-p (if it is in the sweep)
Nothing is overwritten (N and M count up).

CLI:  python -m src.error_rate_sweep --ps 0.002 0.003 0.004 0.005 0.006 0.007 --batches 10
Replot saved data (no decoding; needs a clean git tree, whose commit goes in the footers;
replaces that run's PNGs) and print the error-suppression table:
      python -m src.error_rate_sweep --replot 1 --op-run 44
Operating point with its uncertainty boxes (operating_point_run<M>_uncertainty.png, clean tree):
      python -m src.error_rate_sweep --uncertainty 44
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import asdict, fields
from pathlib import Path

import matplotlib.pyplot as plt

from src.figures import PAPER_RC, SURFACE, plot_paper_style, plot_slides, plot_sweep, read_rows, write_rows
from src.operating_point import (OPERATING_POINT_DECODERS, RUN_STEM, RUNS_DIR, OperatingPoint, default_factories,
                                 measure, plot_operating_point, plot_operating_point_uncertainty, save_run)

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


def meta_for(rows: list[dict], run: int, git_commit: str | None = None) -> dict:
    first = rows[0]
    return {"cpu": first["cpu"], "power": first["power"], "git_commit": git_commit or first["git_commit"],
            "run": run, "threads_note": "every decoder single-thread (PyMatching too)"}


def save_figures(rows: list[dict], run: int, directory: Path = SWEEP_DIR,
                 git_commit: str | None = None) -> tuple[Path, Path, Path]:
    meta = meta_for(rows, run, git_commit)
    ours = directory / f"{SWEEP_STEM}{run}_ours.png"
    paper = directory / f"{SWEEP_STEM}{run}_paper.png"
    slides = directory / f"{SWEEP_STEM}{run}_slides.png"
    fig = plot_sweep(rows, meta)
    fig.savefig(ours, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    with plt.rc_context(PAPER_RC):
        fig = plot_paper_style(rows, meta)
        fig.savefig(paper, dpi=200, facecolor="white")
    plt.close(fig)
    fig = plot_slides(rows, meta)
    fig.savefig(slides, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return ours, paper, slides


SUPPRESSION_DECODERS = ("PyMatching", "Correlated PyMatching", "UnionFind3D", "Hierarchical")


def suppression(rows: list[dict], decoders=SUPPRESSION_DECODERS) -> dict:
    """{(p, decoder, d): P_L(d) / P_L(d+2)} from per-shot failure rates (None if P_L(d+2) = 0)."""
    ler = {(r["p"], r["decoder"], r["d"]): r["ler"] for r in rows}
    out = {}
    for (p, name, d), v in ler.items():
        if name in decoders and (p, name, d + 2) in ler:
            w = ler[(p, name, d + 2)]
            out[(p, name, d)] = v / w if w > 0 else None
    return out


def suppression_markdown(rows: list[dict], decoders=SUPPRESSION_DECODERS) -> str:
    sup = suppression(rows, decoders)
    ds = sorted({d for (_p, _n, d) in sup})
    ps = sorted({p for (p, _n, _d) in sup})
    out = ["| p | Decoder | " + " | ".join(f"d={d} → {d + 2}" for d in ds) + " |",
           "|---|---|" + "---|" * len(ds)]
    for p in ps:
        for name in decoders:
            cells = [sup.get((p, name, d)) for d in ds]
            out.append(f"| {100 * p:g}% | {name} | " + " | ".join("–" if c is None else f"{c:.2f}" for c in cells)
                       + " |")
    return "\n".join(out)


def load_operating_point(path: Path) -> list[OperatingPoint]:
    types = {f.name: f.type for f in fields(OperatingPoint)}
    conv = {"int": int, "float": float, "str": str}
    with open(path, newline="", encoding="utf-8") as f:
        return [OperatingPoint(**{k: conv[types[k]](v) for k, v in r.items()}) for r in csv.DictReader(f)]


def clean_commit() -> str:
    from src.machine import git_commit
    commit = git_commit()
    if commit.endswith("+dirty") or commit == "unknown":
        raise SystemExit(f"replot needs a clean working tree (git says {commit}); commit first")
    return commit


def uncertainty(op_run: int) -> Path:
    """Draw operating_point_run<op_run>_uncertainty.png from the saved CSV (footer: clean commit)."""
    commit = clean_commit()
    pts = load_operating_point(RUNS_DIR / f"{RUN_STEM}{op_run}.csv")
    for r in pts:
        r.git_commit = commit
    return plot_operating_point_uncertainty(pts, RUNS_DIR / f"{RUN_STEM}{op_run}_uncertainty.png", op_run)


def replot(run: int, op_run: int | None) -> None:
    """Redraw a saved sweep (and operating-point run) from its CSVs; the footer shows the current,
    clean commit (the decoders' code is checked unchanged since the run's recorded commit)."""
    commit = clean_commit()
    rows = read_rows(SWEEP_DIR / f"{SWEEP_STEM}{run}.csv")
    for path in save_figures(rows, run, git_commit=commit):
        print(f"wrote {path}")
    if op_run is not None:
        pts = load_operating_point(RUNS_DIR / f"{RUN_STEM}{op_run}.csv")
        for r in pts:
            r.git_commit = commit
        print(f"wrote {plot_operating_point(pts, RUNS_DIR / f'{RUN_STEM}{op_run}.png', run_number=op_run)}")
    print("\nError suppression per +2 distance, P_L(d) / P_L(d+2) (per-shot rates):\n")
    print(suppression_markdown(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ps", type=float, nargs="+", default=[0.002, 0.003, 0.004, 0.005, 0.006, 0.007])
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--shots", type=int, default=50_000, help="shots per batch")
    ap.add_argument("--batches", type=int, default=10, help="independent batches per point")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--op-p", type=float, default=0.005, help="p whose rows also make an operating-point run")
    ap.add_argument("--decoders", nargs="+", default=list(OPERATING_POINT_DECODERS))
    ap.add_argument("--replot", type=int, default=None, help="redraw sweep run N from its CSV (no decoding)")
    ap.add_argument("--op-run", type=int, default=None, help="with --replot: also redraw this operating-point run")
    ap.add_argument("--uncertainty", type=int, default=None,
                    help="draw operating_point_run<M>_uncertainty.png from operating-point run M (no decoding)")
    args = ap.parse_args()
    if args.uncertainty is not None:
        print(f"wrote {uncertainty(args.uncertainty)}")
        return
    if args.replot is not None:
        replot(args.replot, args.op_run)
        return

    factories = default_factories(args.decoders)
    points = []
    for p in args.ps:
        print(f"--- p = {p}", flush=True)
        points += measure(args.distances, p, args.shots, args.seed, factories, batches=args.batches, threads=1,
                          log=lambda s, p=p: print(f"p={p} {s}", flush=True))
    rows = [to_figure_row(r) for r in points]
    run, csv_path = claim_run()
    write_rows(rows, csv_path)
    ours, paper, slides = save_figures(rows, run)
    print(f"sweep run {run}: wrote {csv_path}, {ours}, {paper}, {slides}")
    op_rows = [r for r in points if abs(r.p - args.op_p) < 1e-12]
    if op_rows:
        n, op_csv, op_png = save_run(op_rows)
        print(f"operating point run {n}: wrote {op_csv} and {op_png}")


if __name__ == "__main__":
    main()
