"""Profile the original two C++ passes using a separate instrumented binary."""

import argparse
import csv
import ctypes as ct
import os
from pathlib import Path

import numpy as np
import pymatching

from src.circuits import detector_error_model, surface_code_circuit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    path = Path(
        "external/fast_cp_build",
        "cp_detail.dll" if os.name == "nt" else "libcp_detail.so",
    )
    lib = ct.CDLL(str(path.resolve()))
    lib.cp_detail.argtypes = [
        ct.c_char_p,
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_void_p,
        ct.c_void_p,
        ct.c_void_p,
    ]
    lib.cp_detail.restype = ct.c_int
    lib.cp_detail_error.restype = ct.c_char_p
    with args.out.open("x", newline="") as f:
        writer = None
        for d in (3, 7, 11):
            c = surface_code_circuit(d, 0.005)
            dem = detector_error_model(c)
            shots = c.compile_detector_sampler(seed=20261121).sample(
                8192, bit_packed=True
            )
            ref = pymatching.Matching.from_detector_error_model(
                dem, enable_correlations=True
            ).decode_batch(
                shots,
                enable_correlations=True,
                bit_packed_shots=True,
                bit_packed_predictions=True,
            )
            for rep in range(3):
                times, counts = np.zeros(5), np.zeros(4, np.uint64)
                out = np.empty_like(ref)
                status = lib.cp_detail(
                    str(dem).encode(),
                    shots.ctypes.data,
                    len(shots),
                    shots.shape[1],
                    out.ctypes.data,
                    times.ctypes.data,
                    counts.ctypes.data,
                )
                if status:
                    raise ValueError(lib.cp_detail_error().decode())
                np.testing.assert_array_equal(out, ref)
                row = {
                    "distance": d,
                    "rounds": d,
                    "p": 0.005,
                    "seed": 20261121,
                    "shots": len(shots),
                    "repeat": rep + 1,
                    "first_pass_and_paths_s": times[0],
                    "matching_graph_update_s": times[1],
                    "search_graph_update_s": times[2],
                    "second_pass_s": times[3],
                    "undo_and_counters_s": times[4],
                    "first_pass_edges": counts[0],
                    "matching_weight_writes": counts[1],
                    "search_weight_writes": counts[2],
                    "shots_without_weight_changes": counts[3],
                }
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)


if __name__ == "__main__":
    main()
