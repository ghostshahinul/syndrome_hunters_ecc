"""Every team decoder on the same shots (branch integration/all-decoders).

Operating point at circuit-level p = 0.5%, d = 3/5/7/9, `batches` x `shots` (seeds 0, 1, ...;
the same shots for every decoder), every decoder single-thread. Writes the next
results/operating_point/operating_point_run<N>.csv/.png (standard figure), plus
operating_point_run<N>_dark.png (slide version) and operating_point_run<N>_table.md.

    python -m scripts.experiments.compare_all [--batches 5] [--shots 50000]
"""

from __future__ import annotations

import argparse

from src.operating_point import (RUN_STEM, RUNS_DIR, default_factories, markdown_table, measure,
                                 plot_operating_point_dark, save_run)

DECODERS = ("PyMatching", "Correlated PyMatching", "UnionFind3D", "UF3D correlated", "Hierarchical", "FaultJoin")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--shots", type=int, default=50_000, help="shots per batch")
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = measure(args.distances, args.p, args.shots, args.seed, default_factories(DECODERS),
                   batches=args.batches, threads=1)
    n, csv_path, png_path = save_run(rows)
    dark = plot_operating_point_dark(rows, RUNS_DIR / f"{RUN_STEM}{n}_dark.png", run_number=n)
    table = markdown_table(rows)
    (RUNS_DIR / f"{RUN_STEM}{n}_table.md").write_text(table + "\n", encoding="utf-8")
    print(f"run {n}: {csv_path}, {png_path}, {dark}")
    print(table)


if __name__ == "__main__":
    main()
