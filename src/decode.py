"""Predict logical flips from an unlabeled Stim b8 file of detector shots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import stim

from src.decoders import DECODERS, get_decoder


def decode_file(dem_path: Path, detectors: Path, output: Path, decoder_name="free_energy_matching", batch_size=4096) -> int:
    dem = stim.DetectorErrorModel.from_file(dem_path)
    width = (dem.num_detectors + 7) // 8
    if width == 0 or batch_size < 1:
        raise ValueError("b8 input requires at least one detector and a positive batch size")
    if detectors.stat().st_size % width:
        raise ValueError("truncated b8 input: the last shot has too few bytes")
    decoder = get_decoder(decoder_name)(dem)
    shots = 0
    with detectors.open("rb") as source, output.open("xb") as dest:
        try:
            while chunk := source.read(batch_size * width):
                packed = np.frombuffer(chunk, np.uint8).reshape(-1, width)
                dets = np.unpackbits(packed, axis=1, count=dem.num_detectors, bitorder="little").view(bool)
                obs = decoder.decode_batch(dets)
                dest.write(np.packbits(obs, axis=1, bitorder="little").tobytes())
                shots += len(dets)
        except BaseException:
            dest.close()
            output.unlink()
            raise
    return shots


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dem", type=Path, required=True, help="decomposed Stim detector error model")
    ap.add_argument("--dets", type=Path, required=True, help="detector shots in b8 format")
    ap.add_argument("--out", type=Path, required=True, help="new b8 file for logical predictions")
    ap.add_argument("--decoder", choices=[*DECODERS, "experimental"], default="free_energy_matching")
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()
    count = decode_file(args.dem, args.dets, args.out, args.decoder, args.batch_size)
    print(f"wrote {count} predictions to {args.out}")


if __name__ == "__main__":
    main()
