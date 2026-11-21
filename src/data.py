"""Sample detection events + observable flips, and save/load them.

CLI:  python -m src.data --distances 3 5 7 --ps 0.001 0.005 --shots 100000
Writes one .npz per (d, p) into data/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import stim

from src.circuits import surface_code_circuit

DATA_DIR = Path("data")


@dataclass
class Samples:
    detection_events: np.ndarray  # bool, shape (shots, num_detectors)
    observable_flips: np.ndarray  # bool, shape (shots, num_observables)
    meta: dict = field(default_factory=dict)

    @property
    def shots(self) -> int:
        return self.detection_events.shape[0]


def sample(circuit: stim.Circuit, shots: int, seed: int | None = None) -> Samples:
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    return Samples(dets, obs, {"shots": shots, "seed": seed})


def sample_surface_code(
    distance: int, p: float, shots: int, rounds: int | None = None,
    basis: str = "z", seed: int | None = None,
) -> Samples:
    circuit = surface_code_circuit(distance, p, rounds, basis)
    s = sample(circuit, shots, seed)
    s.meta.update(distance=distance, p=p, rounds=rounds or distance, basis=basis)
    return s


def save(samples: Samples, path: str | Path) -> Path:
    """Bit-packed, compressed .npz. Refuses to overwrite an existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as archive:
        np.savez_compressed(
            archive,
            dets=np.packbits(samples.detection_events, axis=1),
            obs=np.packbits(samples.observable_flips, axis=1),
            num_detectors=samples.detection_events.shape[1],
            num_observables=samples.observable_flips.shape[1],
            meta=json.dumps(samples.meta),
        )
    return path


def load(path: str | Path) -> Samples:
    with np.load(path) as f:
        nd, no = int(f["num_detectors"]), int(f["num_observables"])
        dets = np.unpackbits(f["dets"], axis=1, count=nd).astype(bool)
        obs = np.unpackbits(f["obs"], axis=1, count=no).astype(bool)
        meta = json.loads(str(f["meta"]))
    return Samples(dets, obs, meta)


def default_path(distance: int, p: float, rounds: int, basis: str, seed: int | None,
                 shots: int | None = None) -> Path:
    if shots is None:  # Original scaffold filenames.
        return DATA_DIR / f"d{distance}_r{rounds}_p{p:g}_{basis}_s{seed}.npz"
    return DATA_DIR / f"d{distance}_r{rounds}_p{p}_{basis}_s{seed}_n{shots}.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.001, 0.003, 0.005, 0.007, 0.01])
    ap.add_argument("--shots", type=int, default=50_000)
    ap.add_argument("--basis", choices=("x", "z"), default="z")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20261121)
    args = ap.parse_args()
    if args.shots < 1:
        ap.error("--shots must be positive")

    for d in args.distances:
        for p in args.ps:
            rounds = d if args.rounds is None else args.rounds
            path = default_path(d, p, rounds, args.basis, args.seed, args.shots)
            if path.exists():
                print(f"skip  {path} (exists)")
                continue
            s = sample_surface_code(d, p, args.shots, rounds, args.basis, args.seed)
            save(s, path)
            print(f"wrote {path}  shots={s.shots}  detectors={s.detection_events.shape[1]}")


if __name__ == "__main__":
    main()
