"""Decode a Stim b8 detector stream to a new b8 logical-prediction file."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import stim

from src.decoders.fast_correlated import NativeCorrelated, PackedCorrelated


def decode_file(
    dem_path,
    detectors,
    output,
    *,
    backend="packed",
    workers=1,
    batch_size=4096,
    cache_zero=None,
):
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    dem = stim.DetectorErrorModel.from_file(dem_path)
    width = (dem.num_detectors + 7) // 8
    if not width or Path(detectors).stat().st_size % width:
        raise ValueError("invalid b8 input size for this detector model")
    with ExitStack() as stack:
        if backend == "packed":
            if workers != 1 or cache_zero is not None:
                raise ValueError("worker and cache settings require a native backend")
            decoder = PackedCorrelated(dem)
        elif backend in ("native", "native-control"):
            decoder = NativeCorrelated(
                dem, workers=workers, fast=backend == "native", cache_zero=cache_zero
            )
            stack.callback(decoder.close)
        else:
            raise ValueError(f"unknown backend {backend}")
        source = stack.enter_context(Path(detectors).open("rb"))
        dest = stack.enter_context(Path(output).open("xb"))
        count = 0
        try:
            while buf := source.read(width * batch_size):
                rows = np.frombuffer(buf, np.uint8).reshape(-1, width)
                dest.write(decoder.decode_packed(rows))
                count += len(rows)
        except BaseException:
            dest.close()
            Path(output).unlink()
            raise
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dem", type=Path, required=True)
    ap.add_argument("--dets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--backend", choices=["packed", "native", "native-control"], default="packed"
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--cache-zero", action=argparse.BooleanOptionalAction, default=None)
    a = ap.parse_args()
    print(
        decode_file(
            a.dem,
            a.dets,
            a.out,
            backend=a.backend,
            workers=a.workers,
            batch_size=a.batch_size,
            cache_zero=a.cache_zero,
        ),
        "shots decoded",
    )


if __name__ == "__main__":
    main()
