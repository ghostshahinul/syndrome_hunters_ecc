"""Final round Step 5: Hierarchical on branch opt/overnight vs Hierarchical on main, one run.

Operating point: circuit-level p = 0.5%, d = 3/5/7/9, 5 batches x 50,000 shots (seeds 0-4),
single thread, numba cache cleared. main's union_find / local / hierarchical modules are loaded
from git (--main-rev) next to the branch's, so both are timed in the same process.

    python -m scripts.final_compare [--main-rev REV]
Writes results/optimization/final_step5_<ts>.csv and .png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.opt_b_profile import load_old  # clears the numba cache on import

from src.decoders.cube3d.graph import build_detector_graph  # noqa: E402
from src.decoders.cube3d.hierarchical import Hierarchical  # noqa: E402
from src.figures import timestamp  # noqa: E402
from src.operating_point import DECODER_FACTORIES, measure, plot_operating_point, write_csv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-rev", default="aaf0e88")
    ap.add_argument("--shots", type=int, default=50_000)
    ap.add_argument("--batches", type=int, default=5)
    args = ap.parse_args()
    _, hier_main = load_old(args.main_rev)

    def engine(cls):
        def factory(dem):
            eng = cls(build_detector_graph(dem))
            return lambda dets: eng.run(dets).predictions
        return factory

    from src.decoders.cube3d.union_find import UnionFind3D
    factories = {"PyMatching": DECODER_FACTORIES["PyMatching"],
                 "Correlated PyMatching": DECODER_FACTORIES["Correlated PyMatching"],
                 "UnionFind3D": engine(UnionFind3D),
                 "Hierarchical (main)": engine(hier_main.Hierarchical),
                 "Hierarchical": engine(Hierarchical)}
    rows = measure((3, 5, 7, 9), 0.005, args.shots, 0, factories, batches=args.batches, threads=1)
    out = Path("results") / "optimization"
    stamp = timestamp()
    write_csv(rows, out / f"final_step5_{stamp}.csv")
    plot_operating_point(rows, out / f"final_step5_{stamp}.png")
    print("wrote", out / f"final_step5_{stamp}.csv")


if __name__ == "__main__":
    main()
