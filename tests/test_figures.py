"""Smoke tests for the standard figures (src/figures.py) on tiny synthetic CSVs."""

import csv

import pytest

from src.decoders import DECODERS
from src.figures import (DISPLAY_NAMES, STYLES, display_name, plot_accuracy_runtime,
                         plot_cube_collapse_story, plot_fast_path_fraction, plot_mwpm_tier, plot_sweep,
                         plot_thread_scaling, read_rows, save_figure, style_for, write_rows)

META = {"cpu": "13th Gen Intel(R) Core(TM) i5-13420H", "power": "AC / Balanced", "logical_cpus": "12",
        "backend": "numba", "git_commit": "abc1234", "run": 7}
DECODER_NAMES = ["Cube-Collapse", "PyMatching", "PyMatching 2D on flat", "UnionFind3D", "Hierarchical"]


def _row(decoder, d, p, threads=1, k=10, us=1.0, fast=""):
    shots = 1000
    return {"decoder": decoder, "d": d, "p": p, "rounds": d, "shots": shots, "failures": k,
            "ler": k / shots, "ci_low": max(0, k - 5) / shots, "ci_high": (k + 5) / shots,
            "us_per_shot": us, "threads": threads, "backend": "numba", "fast_fraction": fast,
            "git_commit": "abc1234", "cpu": META["cpu"]}


def _csv(tmp_path, name, rows):
    """Round-trip through a CSV so the plots see exactly what read_rows returns."""
    return read_rows(write_rows(rows, tmp_path / f"{name}.csv"))


@pytest.fixture
def accuracy_rows(tmp_path):
    rows = [_row(n, d, 0.005, us=1.0 + i + d) for d in (3, 5) for i, n in enumerate(DECODER_NAMES)]
    rows += [_row(n, d, 0.005, threads=12, us=0.3 + d / 10) for d in (3, 5) for n in ("UnionFind3D", "Hierarchical")]
    return _csv(tmp_path, "accuracy", rows)


@pytest.fixture
def sweep_rows(tmp_path):
    rows = [_row(n, d, p, k=k, fast=0.9 - d / 20 if n == "Hierarchical" else "")
            for d in (3, 5) for p in (0.001, 0.005) for n, k in
            (("PyMatching", 5), ("UnionFind3D", 6), ("Hierarchical", 0 if p == 0.001 else 7))]
    return _csv(tmp_path, "sweep", rows)


@pytest.fixture
def thread_rows(tmp_path):
    rows = [_row(n, d, 0.005, threads=t, us=10 / t + d) for d in (7, 9) for t in (1, 2, 4, 12)
            for n in ("UnionFind3D", "Hierarchical")]
    rows += [_row("PyMatching", d, 0.005, us=12.0) for d in (7, 9)]
    return _csv(tmp_path, "threads", rows)


def _check(fig, name, tmp_path):
    stamped, latest = save_figure(fig, name, tmp_path / "plots", stamp="20260101_000000")
    assert stamped.exists() and latest.exists() and stamped.stat().st_size > 10_000
    return stamped


def test_accuracy_runtime(accuracy_rows, tmp_path):
    _check(plot_accuracy_runtime(accuracy_rows, META), "accuracy_runtime", tmp_path)


def test_cube_collapse_story(accuracy_rows, tmp_path):
    _check(plot_cube_collapse_story(accuracy_rows, META), "cube_collapse_story", tmp_path)


def test_sweep_with_zero_failure_point(sweep_rows, tmp_path):
    _check(plot_sweep(sweep_rows, META), "accuracy_runtime_sweep", tmp_path)


def test_fast_path_fraction(sweep_rows, tmp_path):
    rows = [r for r in sweep_rows if r["decoder"] == "Hierarchical"]
    _check(plot_fast_path_fraction(rows, META), "fast_path_fraction", tmp_path)


def test_thread_scaling(thread_rows, tmp_path):
    _check(plot_thread_scaling(thread_rows, META), "thread_scaling", tmp_path)


def test_save_never_overwrites(accuracy_rows, tmp_path):
    a = _check(plot_accuracy_runtime(accuracy_rows, META), "x", tmp_path)
    b = _check(plot_accuracy_runtime(accuracy_rows, META), "x", tmp_path)
    assert a != b and a.exists() and b.exists()
    assert (tmp_path / "plots" / "x_latest.png").exists()


def test_csv_roundtrip_types(accuracy_rows):
    r = accuracy_rows[0]
    assert isinstance(r["d"], int) and isinstance(r["ler"], float) and isinstance(r["threads"], int)


def test_style_registry_unique_and_covers_decoders():
    colors = [s.color for s in STYLES.values()]
    markers = [s.marker for s in STYLES.values()]
    assert len(set(colors)) == len(colors) and len(set(markers)) == len(markers)
    for name in ("pymatching", "cube_collapse", "union_find_3d", "hierarchical", "hierarchical_partial"):
        assert name in DECODERS and display_name(name) in STYLES
    assert style_for("union_find_3d") == STYLES["UnionFind3D"]
    assert set(DISPLAY_NAMES.values()) <= set(STYLES)


def test_csv_has_required_columns(tmp_path):
    """The benchmark CSV schema the figures rely on."""
    path = write_rows([_row("UnionFind3D", 3, 0.005)], tmp_path / "schema.csv")
    header = next(csv.reader(open(path, newline="")))
    for col in ("decoder", "d", "p", "rounds", "shots", "failures", "ler", "ci_low", "ci_high",
                "us_per_shot", "threads", "backend", "fast_fraction", "git_commit"):
        assert col in header


def test_mwpm_tier(tmp_path):
    rows = [_row(n, d, 0.005, k=k, us=us) for d in (7, 9)
            for n, k, us in (("PyMatching", 70, 8.0), ("UnionFind3D", 90, 7.0), ("Hierarchical", 90, 7.2))]
    for d in (7, 9):
        for T, k, us in ((d, 75, 3000.0), (2 * d, 82, 1100.0), (4 * d, 88, 240.0)):
            r = _row("Hierarchical + MWPM", d, 0.005, k=k, us=us)
            r.update(threshold=T, routed_fraction=0.4 / T)
            rows.append(r)
    _check(plot_mwpm_tier(_csv(tmp_path, "mwpm", rows), META), "mwpm_tier", tmp_path)
