"""Exact correlated prediction with packed batches and an isolated native backend."""

import ctypes as ct
import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pymatching

from src.decoders.base import Decoder


def _packed(shots, count):
    rows = np.asarray(shots)
    if rows.dtype != np.uint8 or rows.ndim != 2 or rows.shape[1] != (count + 7) // 8:
        raise ValueError(
            "expected a two-dimensional uint8 array of packed detector bytes"
        )
    if count % 8 and np.any(rows[:, -1] >> (count % 8)):
        raise ValueError("nonzero detector padding")
    return rows


def _unpack(rows, width):
    return np.unpackbits(rows, axis=1, count=width, bitorder="little").view(bool)


class PackedCorrelated(Decoder):
    """The wheel's original two passes. Input and output packing are independent."""

    name = "correlated_packed"

    def __init__(self, dem):
        super().__init__(dem)
        self.matching = pymatching.Matching.from_detector_error_model(
            dem, enable_correlations=True
        )

    def predict(self, shots, *, packed_input, packed_output):
        rows = (
            _packed(shots, self.num_detectors)
            if packed_input
            else self._check_input(shots).view(np.uint8)
        )
        out = self.matching.decode_batch(
            rows,
            bit_packed_shots=packed_input,
            bit_packed_predictions=packed_output,
            enable_correlations=True,
        )
        return out if packed_output else out.view(bool)

    def decode_packed(self, rows):
        return self.predict(rows, packed_input=True, packed_output=True)

    def decode_batch(self, rows):
        return self.predict(rows, packed_input=False, packed_output=False)


def _library(path):
    lib = ct.CDLL(str(path))
    lib.cp_create.argtypes = [ct.c_char_p]
    lib.cp_create.restype = ct.c_void_p
    lib.cp_destroy.argtypes = [ct.c_void_p]
    lib.cp_destroy.restype = None
    lib.cp_decode.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_void_p,
        ct.c_size_t,
        ct.c_int,
        ct.c_int,
    ]
    lib.cp_decode.restype = ct.c_int
    lib.cp_error.argtypes = []
    lib.cp_error.restype = ct.c_char_p
    return lib


def _release(lib, graphs, pool):
    if pool is not None:
        pool.shutdown(wait=True)
    for graph in graphs:
        lib.cp_destroy(graph)


class NativeCorrelated(Decoder):
    """One mutable native matching workspace per concurrent worker."""

    name = "correlated_native"

    def __init__(self, dem, *, fast=True, workers=1, cache_zero=None, library=None):
        super().__init__(dem)
        if workers < 1:
            raise ValueError("workers must be positive")
        default = Path(__file__).resolve().parents[2] / "external" / "fast_cp_build"
        self.lib = _library(
            library or default / ("fast_cp.dll" if os.name == "nt" else "libfast_cp.so")
        )
        self.fast = fast
        self.cache_zero = fast if cache_zero is None else cache_zero
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(workers) if workers > 1 else None
        self.graphs = []
        self._finalizer = weakref.finalize(
            self, _release, self.lib, self.graphs, self.pool
        )
        encoded = str(dem).encode()
        try:
            for _ in range(workers):
                graph = self.lib.cp_create(encoded)
                if not graph:
                    raise ValueError(self.lib.cp_error().decode())
                self.graphs.append(graph)
        except BaseException:
            self.close()
            raise

    def close(self):
        with self.lock:
            self._finalizer()

    def _run(self, graph, rows, output):
        status = self.lib.cp_decode(
            graph,
            rows.ctypes.data,
            len(rows),
            rows.shape[1],
            output.ctypes.data,
            output.shape[1],
            self.fast,
            self.cache_zero,
        )
        if status:
            raise ValueError(self.lib.cp_error().decode())

    def decode_packed(self, shots):
        rows = np.ascontiguousarray(_packed(shots, self.num_detectors))
        output = np.empty((len(rows), (self.num_observables + 7) // 8), dtype=np.uint8)
        with self.lock:
            if not self._finalizer.alive:
                raise RuntimeError("decoder closed")
            # Keep the single-call path free of thread scheduling.
            if self.pool is None or len(rows) < len(self.graphs) * 8:
                self._run(self.graphs[0], rows, output)
            else:
                futures = []
                for i, graph in enumerate(self.graphs):
                    chunk = slice(
                        len(rows) * i // len(self.graphs),
                        len(rows) * (i + 1) // len(self.graphs),
                    )
                    futures.append(
                        self.pool.submit(self._run, graph, rows[chunk], output[chunk])
                    )
                wait(futures)
                for future in futures:
                    future.result()
        return output

    def decode_batch(self, rows):
        packed = np.packbits(self._check_input(rows), axis=1, bitorder="little")
        return _unpack(self.decode_packed(packed), self.num_observables)
