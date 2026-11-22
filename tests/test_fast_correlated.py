"""Regression cases for packed I/O, the two-pass shortcut and independent workspaces."""

import os
from pathlib import Path

import numpy as np
import pytest
import stim

from src.circuits import detector_error_model, surface_code_circuit
from src.decoders.baseline import CorrelatedMatchingDecoder, PyMatchingDecoder
from src.decoders.fast_correlated import NativeCorrelated, PackedCorrelated

DLL = (
    Path(__file__).resolve().parents[1]
    / "external/fast_cp_build"
    / ("fast_cp.dll" if os.name == "nt" else "libfast_cp.so")
)
native = pytest.mark.skipif(
    not DLL.is_file(), reason="build the optional pinned native library"
)


def pack(bits):
    return np.packbits(bits, axis=1, bitorder="little")


def test_independent_packing_flags_and_correlations():
    c = surface_code_circuit(3, 0.02)
    dem = detector_error_model(c)
    dets = c.compile_detector_sampler(seed=317).sample(4000)
    ref = CorrelatedMatchingDecoder(dem).decode_batch(dets)
    assert np.any(ref != PyMatchingDecoder(dem).decode_batch(dets))
    decoder = PackedCorrelated(dem)
    for packed_in in (False, True):
        for packed_out in (False, True):
            actual = decoder.predict(
                pack(dets) if packed_in else dets,
                packed_input=packed_in,
                packed_output=packed_out,
            )
            np.testing.assert_array_equal(actual, pack(ref) if packed_out else ref)


@native
@pytest.mark.parametrize(
    "workers,fast,cache", [(1, False, False), (1, True, False), (2, True, True)]
)
@pytest.mark.parametrize("basis", ["x", "z"])
def test_native_repeated_batches_and_worker_boundaries(workers, fast, cache, basis):
    c = surface_code_circuit(4, 0.018, rounds=3, basis=basis)
    dem = detector_error_model(c)
    rows = c.compile_detector_sampler(seed=431).sample(769)
    rows[::11] = False
    ref = CorrelatedMatchingDecoder(dem).decode_batch(rows)
    decoder = NativeCorrelated(dem, workers=workers, fast=fast, cache_zero=cache)
    try:
        for select in (slice(None), slice(None, None, -2), slice(13, 14), slice(0, 0)):
            np.testing.assert_array_equal(
                decoder.decode_packed(pack(rows)[select]), pack(ref[select])
            )
        np.testing.assert_array_equal(decoder.decode_batch(rows), ref)
    finally:
        decoder.close()
    with pytest.raises(RuntimeError, match="closed"):
        decoder.decode_packed(pack(rows))


@native
@pytest.mark.parametrize("last", [1, 8, 63, 64])
def test_observable_bit_order_and_large_observable_fallback(last):
    dem = stim.DetectorErrorModel(
        f"error(.1) D0 L0\nerror(.12) D1 L{last}\nerror(.03) D0 ^ D1 L{last}"
    )
    dets = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 17, dtype=bool)
    ref = CorrelatedMatchingDecoder(dem).decode_batch(dets)
    decoder = NativeCorrelated(dem)
    try:
        np.testing.assert_array_equal(decoder.decode_batch(dets), ref)
    finally:
        decoder.close()


def test_padding_and_shape_validation():
    dec = PackedCorrelated(stim.DetectorErrorModel("error(.1) D0 L0"))
    with pytest.raises(ValueError, match="padding"):
        dec.decode_packed(np.array([[2]], dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        dec.decode_packed(np.ones((2, 1), dtype=bool))


def test_packed_streaming_entry_point(tmp_path):
    from src.fast_cp.decode import decode_file

    circuit = surface_code_circuit(3, 0.013)
    dem = detector_error_model(circuit)
    dem_path, inputs, outputs = [
        tmp_path / n for n in ("memory.dem", "dets.b8", "pred.b8")
    ]
    dem.to_file(dem_path)
    dets = circuit.compile_detector_sampler(seed=834).sample(59)
    pack(dets).tofile(inputs)
    assert decode_file(dem_path, inputs, outputs, batch_size=7) == len(dets)
    assert (
        outputs.read_bytes()
        == pack(CorrelatedMatchingDecoder(dem).decode_batch(dets)).tobytes()
    )
    with pytest.raises(FileExistsError):
        decode_file(dem_path, inputs, outputs)
