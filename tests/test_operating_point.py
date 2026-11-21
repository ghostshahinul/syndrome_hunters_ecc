"""Accuracy vs batch time at one operating point, rendered as
results/tests/<run>/operating_point.png (+ operating_point.csv).

Circuit-level rotated Z memory, rounds = d, p = 0.5%, d = 3, 5, 7, 50,000 shots per point
(override with `pytest --op-shots N`). Decoders: Cube-Collapse, PyMatching, Correlated
PyMatching, PyMatching 2D on Cube-Collapse's flat syndrome, plus any other registered
decoder once implemented. Asserts only sanity; the figure is the output.
"""

import pytest

from src.operating_point import measure, plot_operating_point, write_csv

DISTANCES = (3, 5, 7)
P = 0.005


@pytest.fixture(scope="module")
def operating_points(pytestconfig):
    return measure(DISTANCES, P, pytestconfig.getoption("--op-shots"), seed=0, log=None)


def test_operating_point(operating_points, report):
    rows = operating_points
    for r in rows:
        assert 0 <= r.ler_low <= r.ler <= r.ler_high <= 1
        assert r.us_per_shot > 0
        report.add("operating_point_ler", r.decoder, f"d={r.distance}", 100 * r.ler,
                   yerr=100 * (r.ler_high - r.ler_low) / 2, ylabel="logical failures (%)")
        report.add("operating_point_time", r.decoder, f"d={r.distance}", r.us_per_shot,
                   ylabel="batch time per shot (µs)", log=True)
    report.add_figure("operating_point.png", lambda path: plot_operating_point(rows, path))
    report.add_figure("operating_point.csv", lambda path: write_csv(rows, path))

    print()
    for r in rows:
        print(f"d={r.distance} {r.decoder:<24} LER={100 * r.ler:.3f}% "
              f"[{100 * r.ler_low:.3f}, {100 * r.ler_high:.3f}]  {r.us_per_shot:.2f} us/shot")


def test_reference_decoders_improve_with_distance(operating_points):
    """Below threshold, full-3D PyMatching must improve from d=3 to d=7."""
    ler = {(r.decoder, r.distance): r.ler for r in operating_points}
    assert ler[("PyMatching", 7)] < ler[("PyMatching", 3)]
