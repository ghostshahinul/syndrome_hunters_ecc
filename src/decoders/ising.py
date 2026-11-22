"""NVIDIA Fast R9 correction frame followed by ordinary or correlated matching."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.decoders.base import Decoder
from src.decoders.baseline import CorrelatedMatchingDecoder, PyMatchingDecoder
from src.ising_data import (DEFAULT_CHECKPOINT, DEFAULT_REPO, MODEL_ID, MODEL_REV, MODEL_SHA,
                            activate, verify_checkpoint)

MATCHERS = {"ising_pymatching": PyMatchingDecoder, "ising_pymatching_correlated": CorrelatedMatchingDecoder}
REFERENCE_HYBRIDS = tuple(MATCHERS)
BOUNDARY_HYBRID = "ising_boundary_correlated"
MATCHERS[BOUNDARY_HYBRID] = CorrelatedMatchingDecoder


class NvidiaPredecoder:
    def __init__(self, exp, repo=DEFAULT_REPO, checkpoint=DEFAULT_CHECKPOINT, device="auto", precision="auto"):
        if exp.rounds < 2:
            raise ValueError("NVIDIA's predecoder requires at least two syndrome rounds")
        activate(Path(repo))
        checked = verify_checkpoint(checkpoint)
        import torch
        from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps
        from export.safetensors_utils import load_safetensors

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
            raise ValueError(f"Unavailable inference device: {device}")
        if precision == "auto":
            precision = "fp16" if device == "cuda" else "fp32"
        if precision not in ("fp16", "fp32") or (device == "cpu" and precision == "fp16"):
            raise ValueError("Use fp32 on CPU and fp16 or fp32 on CUDA")
        self.device = torch.device(device)
        self.torch = torch
        free, total = torch.cuda.mem_get_info(self.device) if device == "cuda" else (None, None)
        start = time.perf_counter()
        model, meta = load_safetensors(str(checked), device=device)
        if str(meta.get("model_id")) != "1" or meta.get("quant_format") != "fp16":
            raise ValueError("Unexpected Fast R9 safetensors model metadata")
        model = model.half() if precision == "fp16" else model.float()
        model = model.to(memory_format=torch.channels_last_3d)
        self.synchronize()
        elapsed = time.perf_counter() - start
        cfg = SimpleNamespace(distance=exp.distance, enable_fp16=precision == "fp16",
                              data=SimpleNamespace(code_rotation="XV"),
                              test=SimpleNamespace(meas_basis_test=exp.basis, n_rounds=exp.rounds,
                                                   th_data=0., th_syn=0., sampling_mode="threshold"))
        maps = _build_stab_maps(exp.distance, "XV")
        self.module = PreDecoderMemoryEvalModule(model.eval(), cfg, maps, self.device).to(self.device).eval()
        self.synchronize()
        self.info = dict(device=device, precision=precision, model=MODEL_ID, model_revision=MODEL_REV,
                         checkpoint_sha256=MODEL_SHA, model_load_s=elapsed, torch_version=str(torch.__version__),
                         gpu=torch.cuda.get_device_name(self.device) if device == "cuda" else None,
                         gpu_total_mib=total / 2**20 if total is not None else None,
                         gpu_free_before_model_mib=free / 2**20 if free is not None else None,
                         cuda_runtime=torch.version.cuda if device == "cuda" else None)

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def reset_peak(self):
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def peak_mib(self):
        if self.device.type == "cuda":
            return self.torch.cuda.max_memory_allocated(self.device) / 2**20
        return None

    def predecode(self, dets):
        with self.torch.inference_mode():
            self.synchronize()
            tensor = self.torch.from_numpy(np.ascontiguousarray(dets, dtype=np.uint8)).to(self.device)
            packed = self.module(tensor).cpu().numpy()
            self.synchronize()
        if packed.dtype != np.uint8 or packed.shape != (len(dets), dets.shape[1] + 1):
            raise ValueError("Invalid NVIDIA output layout")
        # NVIDIA returns [pre_L | residual detectors] with the original detector order.
        return np.ascontiguousarray(packed[:, 1:], dtype=bool), packed[:, :1].astype(bool)


class IsingHybrid(Decoder):
    @classmethod
    def from_experiment(cls, name, exp, repo=DEFAULT_REPO, checkpoint=DEFAULT_CHECKPOINT,
                        device="auto", precision="auto"):
        """Construct both stages from the same audited NVIDIA experiment."""
        backend = NvidiaPredecoder(exp, repo, checkpoint, device, precision)
        block = None
        if name == BOUNDARY_HYBRID:
            half = (exp.distance**2 - 1) // 2
            index = 2 * exp.rounds - (2 if exp.basis == "X" else 3)
            block = slice(index * half, (index + 1) * half)
        return cls(name, exp.dem, backend, block)

    def __init__(self, name, dem, predecoder, complementary_block=None):
        super().__init__(dem)
        if self.num_observables != 1:
            raise ValueError("NVIDIA surface memory supports one logical observable")
        self.name = name
        self.predecoder = predecoder
        self.matcher = MATCHERS[name](dem)
        if (name == BOUNDARY_HYBRID) != (complementary_block is not None):
            raise ValueError("Boundary restoration requires the audited complementary-check indices")
        self.complementary_block = complementary_block

    def decode_batch(self, detection_events):
        dets = self._check_input(detection_events)
        if len(dets) == 0:
            return np.empty((0, self.num_observables), dtype=bool)
        start = time.perf_counter_ns()
        residual, frame = self.predecoder.predecode(dets)
        if self.complementary_block is not None:
            # The upstream transform uses a zero sentinel for this raw syndrome
            # block. Restore the omitted input term in R = input XOR correction.
            block = self.complementary_block
            residual[:, block] ^= dets[:, block]
        split = time.perf_counter_ns()
        if residual.shape != dets.shape or frame.shape != (len(dets), self.num_observables):
            raise ValueError("Invalid predecoder detector or logical-frame dimensions")
        if residual.dtype != bool or frame.dtype != bool:
            raise ValueError("Expected binary boolean residuals and logical frame")
        predictions = self.matcher.decode_batch(residual) ^ frame
        end = time.perf_counter_ns()
        self.last_ai_s = (split - start) * 1e-9
        self.last_global_s = (end - split) * 1e-9
        self.last_residual = residual
        return predictions

    def residual_events(self):
        return int(np.count_nonzero(self.last_residual))
