"""Save golden decoder outputs, so later speedups can be checked to be bit-identical.

Writes tests/golden/golden_outputs.npz: for each case (noise tier, d, p) the packed detection
events, and for each decoder config the predictions (packed observable bits), ok,
max_cluster, and an order-independent hash of the correction (when the engine can record
one). Regenerate ONLY when an output change is intended, and say so in the commit.

    python -m scripts.experiments.make_golden
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.circuits import tier_circuit
from src.decoders.cube3d.ensemble import EnsembleConfig, UFEnsemble
from src.decoders.cube3d.graph import detector_graph
from src.decoders.cube3d.hierarchical import Hierarchical, HierarchicalConfig
from src.decoders.cube3d.union_find import UnionFind3D, UnionFind3DConfig

OUT = Path("tests") / "golden" / "golden_outputs.npz"
TIER_P = {"code_capacity": 0.05, "phenomenological": 0.01, "circuit": 0.005}
CASES = [(t, d) for t in TIER_P for d in (3, 5, 7)] + [("circuit", 9)]
SHOTS = 600
_PRIME = np.uint64(0x9E3779B97F4A7C15)


def correction_hash(corr: np.ndarray) -> np.ndarray:
    """(shots, E) uint8 -> (shots,) uint64: XOR over flipped edges of a mixed edge id."""
    E = corr.shape[1]
    ids = (np.arange(1, E + 1, dtype=np.uint64) * _PRIME) ^ (np.arange(E, dtype=np.uint64) << np.uint64(17))
    out = np.zeros(corr.shape[0], np.uint64)
    for e in np.flatnonzero(corr.any(axis=0)):
        out ^= np.where(corr[:, e] != 0, ids[e], np.uint64(0))
    return out


def pack_bits(pred: np.ndarray) -> np.ndarray:
    return (pred.astype(np.uint64) << np.arange(pred.shape[1], dtype=np.uint64)).sum(axis=1, dtype=np.uint64)


def engines(g):
    yield "uf", UnionFind3D(g, UnionFind3DConfig())
    yield "uf_corr", UnionFind3D(g, UnionFind3DConfig(correlated=True))
    corr = UnionFind3DConfig(correlated=True)  # pinned: no pass-2 gate (the pre-gate behaviour)
    yield "hier_whole_corr", Hierarchical(g, HierarchicalConfig(gate="whole_shot", table=False, uf=corr))
    yield "hier_partial_corr", Hierarchical(g, HierarchicalConfig(gate="partial", table=False, uf=corr))
    yield "hier_whole_plain", Hierarchical(g, HierarchicalConfig(gate="whole_shot", uf=UnionFind3DConfig(),
                                                                 table=False))
    yield "ens4_corr", UFEnsemble(g, EnsembleConfig(M=4))


def run_engine(name, eng, dets):
    out = {}
    if name.startswith("uf") or name.startswith("ens"):
        r = eng.run(dets, record_correction=True)
        out["hash"] = correction_hash(r.correction)
    else:
        r = eng.run(dets)
    out["pred"] = pack_bits(r.predictions)
    out["ok"] = r.ok
    if getattr(r, "max_cluster", None) is not None:
        out["size"] = r.max_cluster
    if getattr(r, "chosen", None) is not None:
        out["chosen"] = r.chosen
    return out


def main() -> None:
    data = {}
    for tier, d in CASES:
        c = tier_circuit(tier, d, TIER_P[tier])
        dets = c.compile_detector_sampler(seed=1234 + d).sample(SHOTS, separate_observables=True)[0]
        key = f"{tier}_{d}"
        data[f"{key}__dets"] = np.packbits(dets, axis=1)
        data[f"{key}__ndet"] = np.array(dets.shape[1])
        g = detector_graph(tier, d, TIER_P[tier])
        for name, eng in engines(g):
            for field, arr in run_engine(name, eng, dets).items():
                data[f"{key}__{name}__{field}"] = arr
        print("golden", key, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **data)
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
