"""Vary packed input, packed output, core updates and zero caching independently."""

import argparse
import csv
import hashlib
from functools import partial
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from src.circuits import detector_error_model, surface_code_circuit
from src.decoders.baseline import CorrelatedMatchingDecoder
from src.decoders.fast_correlated import NativeCorrelated, PackedCorrelated


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with args.out.open("x", newline="") as f:
        writer = None
        for d in (3, 7, 11):
            c = surface_code_circuit(d, 0.005)
            dem = detector_error_model(c)
            x, truth = c.compile_detector_sampler(seed=20261121).sample(
                8192, separate_observables=True, bit_packed=True
            )
            dense = np.unpackbits(
                x, axis=1, count=dem.num_detectors, bitorder="little"
            ).view(bool)
            ref = np.packbits(
                CorrelatedMatchingDecoder(dem).decode_batch(dense),
                axis=1,
                bitorder="little",
            )
            empty = int(np.count_nonzero(~x.any(axis=1)))
            digest = hashlib.sha256(x.tobytes() + truth.tobytes()).hexdigest()
            for family in ("wheel", "native"):
                for a in (False, True):
                    for b in (False, True):
                        if family == "wheel":
                            decoder = PackedCorrelated(dem)
                            rows = x if a else dense

                            decode = partial(
                                decoder.predict, packed_input=a, packed_output=b
                            )
                            packed_output = b
                        else:
                            decoder = NativeCorrelated(dem, fast=a, cache_zero=b)
                            rows = x
                            decode = decoder.decode_packed
                            packed_output = True
                        try:
                            decode(rows[:256])
                            for rep in range(3):
                                elapsed = 0
                                for j in range(0, len(rows), 4096):
                                    batch = rows[j : j + 4096]
                                    start = perf_counter_ns()
                                    out = decode(batch)
                                    elapsed += perf_counter_ns() - start
                                    bits = (
                                        out
                                        if packed_output
                                        else np.packbits(out, axis=1, bitorder="little")
                                    )
                                    np.testing.assert_array_equal(
                                        bits, ref[j : j + len(batch)]
                                    )
                                row = {
                                    "distance": d,
                                    "p": 0.005,
                                    "rounds": d,
                                    "seed": 20261121,
                                    "shots": len(rows),
                                    "family": family,
                                    "packed_input": a if family == "wheel" else True,
                                    "packed_output": packed_output,
                                    "fast_core": a if family == "native" else "",
                                    "cache_zero": b if family == "native" else "",
                                    "repeat": rep + 1,
                                    "us_per_shot": elapsed / len(rows) / 1000,
                                    "empty_syndromes": empty,
                                    "logical_failures": int(
                                        (ref != truth).any(axis=1).sum()
                                    ),
                                    "mismatches": 0,
                                    "little_endian_sample_sha256": digest,
                                }
                                if writer is None:
                                    writer = csv.DictWriter(f, fieldnames=row)
                                    writer.writeheader()
                                writer.writerow(row)
                        finally:
                            if isinstance(decoder, NativeCorrelated):
                                decoder.close()


if __name__ == "__main__":
    main()
