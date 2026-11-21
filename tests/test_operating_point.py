"""Accuracy vs batch time at one operating point. Every pytest run writes a new
results/operating_point/operating_point_run<N>.png (+ .csv), N counting up per run.

Circuit-level rotated Z memory, rounds = d, p = 0.5%, d = 3, 5, 7, 50,000 shots per point
(override with `pytest --op-shots N`). Decoders: Cube-Collapse, PyMatching, Correlated
PyMatching, PyMatching 2D on Cube-Collapse's flat syndrome, plus any other registered
decoder once implemented. Asserts only sanity; the figure is the output.
"""

import pytest

from src.operating_point import RUNS_DIR, measure, save_run

DISTANCES = (3, 5, 7)
P = 0.005


@pytest.fixture(scope="module")
def operating_points(pytestconfig):
    return measure(DISTANCES, P, pytestconfig.getoption("--op-shots"), seed=0, log=None)


def test_operating_point(operating_points, report, pytestconfig):
    rows = operating_points
    for r in rows:
        assert 0 <= r.ler_low <= r.ler <= r.ler_high <= 1
        assert r.us_per_shot > 0
        report.add("operating_point_ler", r.decoder, f"d={r.distance}", 100 * r.ler,
                   yerr=100 * (r.ler_high - r.ler_low) / 2, ylabel="logical failures (%)")
        report.add("operating_point_time", r.decoder, f"d={r.distance}", r.us_per_shot,
                   ylabel="batch time per shot (µs)", log=True)
    n, csv_path, png_path = save_run(rows, pytestconfig.rootpath / RUNS_DIR)
    assert png_path.exists() and csv_path.exists()

    print(f"\noperating point run {n}: {png_path}")
    for r in rows:
        print(f"d={r.distance} {r.decoder:<24} LER={100 * r.ler:.3f}% "
              f"[{100 * r.ler_low:.3f}, {100 * r.ler_high:.3f}]  {r.us_per_shot:.2f} us/shot")


def test_reference_decoders_improve_with_distance(operating_points):
    """Below threshold, full-3D PyMatching must improve from d=3 to d=7."""
    ler = {(r.decoder, r.distance): r.ler for r in operating_points}
    assert ler[("PyMatching", 7)] < ler[("PyMatching", 3)]
