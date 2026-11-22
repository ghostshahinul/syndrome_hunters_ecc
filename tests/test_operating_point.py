"""Accuracy vs batch time at one operating point. Every pytest run writes a new
results/operating_point/operating_point_run<N>.png (+ .csv), N counting up per run.

Circuit-level rotated Z memory, rounds = d, p = 0.5%, d = 3, 5, 7, 9.
Each point: 3 independent batches of 50,000 shots (override with `pytest --op-batches B
--op-shots N`); LER pooled over the batches, time = mean batch µs/shot with the fastest
and slowest batch as horizontal whiskers. numba decoders run on 1 thread.
Decoders: PyMatching, Correlated PyMatching, UnionFind3D, UF3D correlated, Hierarchical
(deep tier: UF3D correlated). PyMatching 2D on flat is no longer in the default set.
"""

import pytest

from src.operating_point import OPERATING_POINT_DECODERS, RUNS_DIR, measure, save_run

DISTANCES = (3, 5, 7, 9)
P = 0.005


@pytest.fixture(scope="module")
def operating_points(pytestconfig):
    return measure(DISTANCES, P, pytestconfig.getoption("--op-shots"), seed=0, log=None,
                   batches=pytestconfig.getoption("--op-batches"), threads=1)


def _ler(rows):
    return {(r.decoder, r.distance): r for r in rows}


def test_operating_point(operating_points, report, pytestconfig):
    rows = operating_points
    batches = pytestconfig.getoption("--op-batches")
    assert {r.decoder for r in rows} == set(OPERATING_POINT_DECODERS)
    for r in rows:
        assert 0 <= r.ler_low <= r.ler <= r.ler_high <= 1
        assert 0 < r.us_min <= r.us_per_shot <= r.us_max
        assert r.batches == batches and r.shots == batches * r.shots_per_batch
        report.add("operating_point_ler", r.decoder, f"d={r.distance}", 100 * r.ler,
                   yerr=100 * (r.ler_high - r.ler_low) / 2, ylabel="logical failures (%)")
        report.add("operating_point_time", r.decoder, f"d={r.distance}", r.us_per_shot,
                   ylabel="batch time per shot (µs)", log=True)
    n, csv_path, png_path = save_run(rows, pytestconfig.rootpath / RUNS_DIR)
    assert png_path.exists() and csv_path.exists()

    print(f"\noperating point run {n}: {png_path}")
    for r in rows:
        print(f"d={r.distance} {r.decoder:<24} LER={100 * r.ler:.3f}% "
              f"[{100 * r.ler_low:.3f}, {100 * r.ler_high:.3f}]  {r.us_per_shot:.2f} us/shot "
              f"(batches {r.us_min:.2f}-{r.us_max:.2f})")


def test_reference_decoders_improve_with_distance(operating_points):
    """Below threshold, full-3D PyMatching must improve from d=3 to d=7."""
    ler = _ler(operating_points)
    assert ler[("PyMatching", 7)].ler < ler[("PyMatching", 3)].ler


@pytest.mark.parametrize("decoder", ["PyMatching", "Correlated PyMatching", "UnionFind3D", "UF3D correlated",
                                     "Hierarchical"])
def test_distance_9_improves_on_distance_7(operating_points, decoder):
    """d = 9: every 3D decoder is better than at d = 7, with non-overlapping 95% intervals."""
    ler = _ler(operating_points)
    d7, d9 = ler[(decoder, 7)], ler[(decoder, 9)]
    assert d9.ler_high < d7.ler_low, (decoder, d7.ler, d9.ler)


def test_suppression_table_and_csv_roundtrip(tmp_path):
    """Error-rate sweep helpers: P_L(d)/P_L(d+2) and reloading an operating-point CSV."""
    from src.error_rate_sweep import load_operating_point, suppression, suppression_markdown
    from src.operating_point import OperatingPoint, write_csv
    rows = [{"p": 0.005, "decoder": "PyMatching", "d": d, "ler": v} for d, v in ((3, 0.02), (5, 0.01), (7, 0.0))]
    sup = suppression(rows)
    assert sup[(0.005, "PyMatching", 3)] == 2.0 and sup[(0.005, "PyMatching", 5)] is None
    assert "| 0.5% | PyMatching | 2.00 | – |" in suppression_markdown(rows)
    pt = OperatingPoint("PyMatching", 5, 5, 0.005, 100, 3, 0.03, 0.01, 0.08, 2.5, 2, 50, 2.0, 3.0)
    assert load_operating_point(write_csv([pt], tmp_path / "op.csv")) == [pt]
