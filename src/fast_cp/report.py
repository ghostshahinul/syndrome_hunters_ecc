"""Audit paired predictions and render the comparison from measured CSV rows."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.plots import DISTANCE_COLORS, _accuracy_points, _axes


def write_rows(path, rows):
    with path.open("x", newline="", encoding="utf8") as stream:
        w = csv.DictWriter(stream, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


BASE_STYLES = {
    "mwpm": ("Ordinary MWPM", "#555a60", "s"),
    "correlated": ("Original correlated", "#9b3f13", "o"),
    "packed": ("Packed wheel", "#c8800b", "^"),
    "native_control_parallel": ("Native control · {workers} workers", "#526581", "<"),
    "native_control": ("Native upstream control", "#374b8c", "D"),
    "fast_serial": ("Fast · 1 worker", "#005b80", "v"),
    "fast_parallel": ("Fast · {workers} workers", "#00886f", "o"),
}
COLORS = ["#713d91", "#bd3865", "#006c67", "#90352b"]


def styles(names, rows):
    result = {}
    for key in names:
        if key in BASE_STYLES:
            label, color, marker = BASE_STYLES[key]
            result[key] = (
                label.format(
                    workers=next(r["workers"] for r in rows if r["decoder"] == key)
                ),
                color,
                marker,
            )
        else:
            i = sum(n not in BASE_STYLES for n in result)
            result[key] = (
                f"Fast · {next(r['workers'] for r in rows if r['decoder'] == key)} workers",
                COLORS[i % len(COLORS)],
                ("P", "X", "*", "h")[i % 4],
            )
    return result


def load(run):
    files = [run / "summary.csv", *sorted(run.glob("scaling_*/summary.csv"))]
    rows = []
    for path in files:
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                for key in (
                    "distance",
                    "shots",
                    "num_errors",
                    "rounds",
                    "workers",
                    "different_from_correlated",
                ):
                    r[key] = int(r[key])
                for key in (
                    "p",
                    "logical_error_rate",
                    "ler_low95",
                    "ler_high95",
                    "us_per_shot",
                    "decode_s",
                    "shots_per_s",
                    "speedup",
                    "single_p50_us",
                    "single_p95_us",
                    "warm_rss_delta_mib",
                    "process_peak_mib",
                    "peak_growth_mib",
                    "graph_s",
                    "warmup_s",
                ):
                    r[key] = float(r[key])
                rows.append(r)
    return rows


def validate(rows, cfg):
    names = list(dict.fromkeys(r["decoder"] for r in rows))
    for d in cfg["distances"]:
        for p in cfg["ps"]:
            point = [r for r in rows if r["distance"] == d and r["p"] == p]
            assert len(point) == len(names) and {r["decoder"] for r in point} == set(
                names
            )
            assert len({r["sample_sha256"] for r in point}) == 1
            ref = next(r for r in point if r["decoder"] == "correlated")
            for r in point:
                assert (
                    r["shots"] == cfg["shots"]
                    and r["rounds"] == d
                    and int(r["seed"]) == cfg["seed"]
                )
                if r["decoder"] != "mwpm":
                    assert (
                        r["different_from_correlated"] == 0
                        and r["prediction_sha256"] == ref["prediction_sha256"]
                    )
                    assert r["num_errors"] == ref["num_errors"]
    return names


def accuracy(rows, cfg, dest):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.73, bottom=0.18, wspace=0.12)
    fig.suptitle(
        "Logical failures across the physical noise sweep",
        y=0.97,
        x=0.06,
        ha="left",
        weight="bold",
        fontsize=18,
    )
    fig.text(
        0.06,
        0.875,
        f"Rotated {cfg['basis'].upper()} memory  ·  rounds = distance  ·  95% Wilson intervals",
    )
    handles = []
    for ax, key, title in zip(
        axes,
        ("mwpm", "correlated"),
        ("Ordinary MWPM", "All correlated implementations"),
    ):
        for i, d in enumerate(cfg["distances"]):
            pts = sorted(
                (r for r in rows if r["decoder"] == key and r["distance"] == d),
                key=lambda r: r["p"],
            )
            color = DISTANCE_COLORS[i % len(DISTANCE_COLORS)]
            marker = ("o", "s", "^", "D", "v")[i % 5]
            _accuracy_points(ax, pts, [100 * r["p"] for r in pts], color, marker)
            if key == "mwpm":
                handles.append(
                    Line2D([], [], label=f"d = {d}", color=color, marker=marker)
                )
        ax.set(
            title=title,
            yscale="log",
            xticks=[p * 100 for p in cfg["ps"]],
            xlabel="Physical noise (%)",
        )
        _axes(ax)
    axes[0].set_ylabel("Logical failures per shot (%)")
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.85),
        frameon=False,
        ncol=5,
    )
    fig.text(
        0.06,
        0.035,
        "Correlated curves coincide on every shot. Open downward markers show upper endpoints for zero observed failures.",
        fontsize=9,
    )
    fig.savefig(dest / "A_logical_error_rate.png", dpi=240)
    plt.close(fig)


def curves(rows, cfg, names, style, key, title, unit, filename, foot, dest):
    n = len(cfg["ps"])
    fig, axes = plt.subplots(
        (n + 1 + 2) // 3,
        3,
        figsize=(13.6, 7.8),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.075, right=0.98, top=0.81, bottom=0.15, hspace=0.42, wspace=0.2
    )
    fig.suptitle(title, x=0.056, y=0.97, ha="left", fontsize=19, weight="bold")
    fig.text(
        0.056,
        0.89,
        f"Rotated {cfg['basis'].upper()} memory  ·  {cfg['shots']:,} shots per point  ·  rounds = distance",
    )
    for ax, p in zip(axes.flat, cfg["ps"]):
        for name in names:
            pts = sorted(
                (r for r in rows if r["decoder"] == name and r["p"] == p),
                key=lambda r: r["distance"],
            )
            label, color, marker = style[name]
            ax.plot(
                [r["distance"] for r in pts],
                [r[key] for r in pts],
                color=color,
                marker=marker,
                ms=4,
                linewidth=1.6,
                label=label,
            )
        ax.set(title=f"Physical noise {p:.1%}", xticks=cfg["distances"])
        if key != "process_peak_mib":
            ax.set_yscale("log")
        if key == "speedup":
            ax.axhline(1, color="#777", lw=0.8, ls=":")
        _axes(ax)
    legend_ax = axes.flat[n]
    legend_ax.axis("off")
    legend_ax.legend(
        *axes.flat[0].get_legend_handles_labels(),
        loc="center left",
        frameon=False,
        fontsize=9.5,
    )
    for ax in list(axes.flat)[n + 1 :]:
        ax.axis("off")
    fig.supxlabel("Code distance", y=0.075)
    fig.supylabel(unit, x=0.009)
    fig.text(0.056, 0.028, foot, fontsize=9)
    fig.savefig(dest / filename, dpi=240)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    cfg = json.loads((args.run / "config.json").read_text())
    rows = load(args.run)
    names = validate(rows, cfg)
    originals = {(r["distance"], r["p"], r["decoder"]): r for r in rows}
    for r in rows:
        r["measurement_source"] = "initial"
    for rerun in sorted(args.run.glob("replication/d*/20*")):
        repeated = load(rerun)
        validate(repeated, json.loads((rerun / "config.json").read_text()))
        for r in repeated:
            key = (r["distance"], r["p"], r["decoder"])
            first = originals[key]
            if (
                first["sample_sha256"] != r["sample_sha256"]
                or first["prediction_sha256"] != r["prediction_sha256"]
            ):
                raise ValueError("replication used different samples or predictions")
            r["measurement_source"] = str(rerun.relative_to(args.run)).replace(
                chr(92), "/"
            )
            originals[key] = r
    rows = list(originals.values())
    for r in rows:
        r["speedup"] = (
            originals[r["distance"], r["p"], "correlated"]["decode_s"] / r["decode_s"]
        )
    dest = args.out or args.run / "report"
    dest.mkdir(parents=True, exist_ok=False)
    write_rows(dest / "all_points.csv", rows)
    ranking = []
    for name in names:
        group = [r for r in rows if r["decoder"] == name]
        n = sum(r["shots"] for r in group)
        seconds = sum(r["decode_s"] for r in group)
        last = next(
            r
            for r in group
            if r["distance"] == max(cfg["distances"])
            and r["p"] == min(cfg["ps"], key=lambda p: abs(p - 0.005))
        )
        ranking.append(
            {
                "decoder": name,
                "workers": group[0]["workers"],
                "shots": n,
                "failures": sum(r["num_errors"] for r in group),
                "aggregate_ler": sum(r["num_errors"] for r in group) / n,
                "grid_decode_s": seconds,
                "grid_shots_s": n / seconds,
                "grid_us_shot": 1e6 * seconds / n,
                "largest_distance": last["distance"],
                "scalability_p": last["p"],
                "largest_us_shot": last["us_per_shot"],
                "largest_p50_us": last["single_p50_us"],
                "largest_p95_us": last["single_p95_us"],
                "largest_peak_mib": last["process_peak_mib"],
                "largest_setup_s": last["graph_s"],
                "mismatched_shots": sum(r["different_from_correlated"] for r in group),
            }
        )
    ref = next(r for r in ranking if r["decoder"] == "correlated")
    for r in ranking:
        r["speedup"] = ref["grid_decode_s"] / r["grid_decode_s"]
        for col, rank in (
            ("grid_decode_s", "speed_rank"),
            ("failures", "accuracy_rank"),
            ("largest_us_shot", "large_distance_speed_rank"),
        ):
            r[rank] = 1 + sum(other[col] < r[col] for other in ranking)
    ranking.sort(key=lambda r: r["speed_rank"])
    write_rows(dest / "ranking.csv", ranking)
    style = styles(names, rows)
    accuracy(rows, cfg, dest)
    args = (rows, cfg, names, style)
    curves(
        *args,
        "us_per_shot",
        "Amortized batch decoding time",
        "Microseconds per shot",
        "B_time.png",
        "Median of three warmed passes. Sampling and graph construction are timed separately.",
        dest,
    )
    curves(
        *args,
        "shots_per_s",
        "Decoding throughput",
        "Shots per second",
        "C_throughput.png",
        "Each point uses the same input samples for every decoder.",
        dest,
    )
    curves(
        *args,
        "speedup",
        "Speed relative to original correlated PyMatching",
        "Speedup (×)",
        "D_speedup.png",
        "Above 1× means faster than the original correlated wheel. Ordinary MWPM has different predictions.",
        dest,
    )
    curves(
        *args,
        "process_peak_mib",
        "Peak resident memory during batch decoding",
        "Process peak (MiB)",
        "E_memory.png",
        "Fresh process per point and strategy. Includes Python, native allocations, graph workspaces and input arrays.",
        dest,
    )
    print(dest)


if __name__ == "__main__":
    main()
