"""Pytest plugin: after every test run, write plots + raw data to results/tests/<timestamp>/.

Always produced:
  outcomes.png    passed / failed / skipped / error counts per test module
  durations.png   slowest tests
  tests.csv       one row per test (outcome, duration)
Produced from metrics that tests record with the `report` fixture:
  <plot>.png      grouped bar chart per plot name (series = decoder, x = case)
  metrics.csv     every recorded value

Disable with `pytest --no-plots`.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

RESULTS_ROOT = Path("results") / "tests"
OUTCOMES = ("passed", "failed", "skipped", "error")
COLORS = {"passed": "#2e9e5b", "failed": "#d1453b", "skipped": "#b8b8b8", "error": "#8a3ffc"}


@dataclass
class Metric:
    plot: str
    series: str
    x: str
    y: float
    yerr: float | None
    ylabel: str
    log: bool
    test: str


class ResultsReport:
    def __init__(self, rootdir: Path):
        self.rootdir = rootdir
        self.tests: dict[str, dict] = {}
        self.metrics: list[Metric] = []
        self.out_dir: Path | None = None
        self._current_test = ""

    # ---- API used by tests via the `report` fixture -------------------------------
    def add(self, plot: str, series: str, x: str, y: float, yerr: float | None = None,
            ylabel: str = "", log: bool = False) -> None:
        """Record one value. All values sharing `plot` end up on the same chart."""
        self.metrics.append(Metric(plot, series, str(x), float(y),
                                   None if yerr is None else float(yerr),
                                   ylabel, log, self._current_test))

    # ---- pytest hooks --------------------------------------------------------------
    def pytest_runtest_setup(self, item):
        self._current_test = item.nodeid

    def pytest_runtest_logreport(self, report):
        t = self.tests.setdefault(report.nodeid, {"outcome": "passed", "duration": 0.0})
        t["duration"] += report.duration
        if report.skipped:
            t["outcome"] = "skipped"
        elif report.failed:
            t["outcome"] = "failed" if report.when == "call" else "error"

    def pytest_sessionfinish(self, session):
        if not self.tests:
            return
        import matplotlib
        matplotlib.use("Agg")

        out = self.rootdir / RESULTS_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        out.mkdir(parents=True, exist_ok=False)
        self.out_dir = out
        self._write_tests_csv(out / "tests.csv")
        self._plot_outcomes(out / "outcomes.png")
        self._plot_durations(out / "durations.png")
        if self.metrics:
            self._write_metrics_csv(out / "metrics.csv")
            by_plot = defaultdict(list)
            for m in self.metrics:
                by_plot[m.plot].append(m)
            for name, ms in by_plot.items():
                self._plot_metric(name, ms, out / f"{name}.png")

    def pytest_terminal_summary(self, terminalreporter):
        if self.out_dir is not None:
            rel = self.out_dir.relative_to(self.rootdir)
            terminalreporter.write_line(f"test plots written to {rel}")

    # ---- writers -------------------------------------------------------------------
    def _write_tests_csv(self, path: Path):
        with open(path, "x", newline="") as f:
            w = csv.writer(f)
            w.writerow(["test", "outcome", "duration_s"])
            for nodeid, t in self.tests.items():
                w.writerow([nodeid, t["outcome"], f"{t['duration']:.6f}"])

    def _write_metrics_csv(self, path: Path):
        with open(path, "x", newline="") as f:
            w = csv.writer(f)
            w.writerow(["plot", "series", "x", "y", "yerr", "test"])
            for m in self.metrics:
                w.writerow([m.plot, m.series, m.x, m.y, "" if m.yerr is None else m.yerr, m.test])

    def _plot_outcomes(self, path: Path):
        import matplotlib.pyplot as plt

        counts = defaultdict(lambda: dict.fromkeys(OUTCOMES, 0))
        for nodeid, t in self.tests.items():
            counts[nodeid.split("::")[0]][t["outcome"]] += 1
        modules = sorted(counts)

        fig, ax = plt.subplots(figsize=(7, 0.6 * len(modules) + 1.5))
        left = [0] * len(modules)
        for outcome in OUTCOMES:
            vals = [counts[m][outcome] for m in modules]
            if any(vals):
                ax.barh(modules, vals, left=left, color=COLORS[outcome], label=outcome)
                left = [a + b for a, b in zip(left, vals)]
        total = {o: sum(c[o] for c in counts.values()) for o in OUTCOMES}
        ax.set(xlabel="tests", title="Test outcomes  " +
               "  ".join(f"{o}={n}" for o, n in total.items() if n))
        ax.invert_yaxis()
        ax.legend(loc="lower right")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_durations(self, path: Path, top: int = 15):
        import matplotlib.pyplot as plt

        ranked = sorted(self.tests.items(), key=lambda kv: kv[1]["duration"], reverse=True)[:top]
        labels = [nodeid.split("::", 1)[-1] for nodeid, _ in ranked]
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(ranked) + 1.5))
        ax.barh(labels, [t["duration"] for _, t in ranked],
                color=[COLORS[t["outcome"]] for _, t in ranked])
        ax.set(xlabel="seconds", title=f"Slowest {len(ranked)} tests (colour = outcome)")
        ax.invert_yaxis()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_metric(self, name: str, ms: list[Metric], path: Path):
        import matplotlib.pyplot as plt

        xs = list(dict.fromkeys(m.x for m in ms))
        series = list(dict.fromkeys(m.series for m in ms))
        width = 0.8 / len(series)
        fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(xs) * len(series) + 2), 4.5))
        for i, s in enumerate(series):
            pts = {m.x: m for m in ms if m.series == s}
            pos = [j + (i - (len(series) - 1) / 2) * width for j, x in enumerate(xs) if x in pts]
            vals = [pts[x].y for x in xs if x in pts]
            errs = [pts[x].yerr or 0 for x in xs if x in pts]
            ax.bar(pos, vals, width, yerr=errs if any(errs) else None, capsize=3, label=s)
        ax.set_xticks(range(len(xs)), xs)
        ax.set(ylabel=ms[0].ylabel or name, title=name.replace("_", " "))
        if ms[0].log:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
