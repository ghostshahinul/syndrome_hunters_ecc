"""Compare Fast R9 devices, precision and batch sizes on one saved operating point."""

import argparse
from pathlib import Path

from src import ising_data as nv
from src.decoders.ising import REFERENCE_HYBRIDS
from src.evaluate import benchmark, write_results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distance", type=int, default=9)
    ap.add_argument("--p", type=float, default=.005)
    ap.add_argument("--basis", choices=("x", "z"), default="z")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 16, 32])
    ap.add_argument("--devices", nargs="+", choices=("cpu", "cuda"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--latency-shots", type=int, default=32)
    ap.add_argument("--ising-repo", type=Path, default=nv.DEFAULT_REPO)
    ap.add_argument("--checkpoint", type=Path, default=nv.DEFAULT_CHECKPOINT)
    ap.add_argument("--data-dir", type=Path, default=Path("data/ising"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/ising/profile"))
    a = ap.parse_args()
    nv.verify_checkpoint(a.checkpoint)
    nv.activate(a.ising_repo)
    import torch
    devices = a.devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    if "cuda" in devices and not torch.cuda.is_available():
        ap.error("CUDA requested but unavailable")
    reference_predictions = {}
    expected_samples = None
    for device in devices:
        for precision in (["fp32", "fp16"] if device == "cuda" else ["fp32"]):
            for batch_size in a.batch_sizes:
                rows = benchmark(["pymatching_correlated", *REFERENCE_HYBRIDS], [a.distance], [a.p], a.shots,
                                 a.basis, a.seed, source="nvidia", ising_repo=a.ising_repo, checkpoint=a.checkpoint,
                                 device=device, precision=precision, data_dir=a.data_dir, batch_size=batch_size,
                                 repeats=a.repeats, latency_shots=a.latency_shots)
                expected_samples = expected_samples or rows[0]["sample_sha256"]
                if any(row["sample_sha256"] != expected_samples for row in rows):
                    raise ValueError("Profiling variants used different samples")
                agreement = {}
                for row in rows:
                    first = reference_predictions.setdefault(row["decoder"], row["prediction_sha256"])
                    agreement[row["decoder"]] = row["prediction_sha256"] == first
                config = {**vars(a), "device": device, "precision": precision, "batch_size": batch_size,
                          "predictions_identical_to_first_variant": agreement}
                path = write_results(rows, a.out_dir, config)
                print(f"{device} {precision} batch {batch_size}: {agreement}, saved {path}", flush=True)


if __name__ == "__main__":
    main()
