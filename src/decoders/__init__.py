"""Decoder registry. Add new decoders to DECODERS so evaluate.py and the tests see them."""

from src.decoders.base import Decoder
from src.decoders.baseline import PyMatchingDecoder
from src.decoders.cube3d.hierarchical import HierarchicalDecoder, HierarchicalPartialDecoder
from src.decoders.cube3d.union_find import UnionFind3DDecoder
from src.decoders.cube_collapse import CubeCollapseDecoder
from src.decoders.experimental import ExperimentalDecoder
from src.decoders.union_find import UnionFindDecoder

DECODERS: dict[str, type[Decoder]] = {
    cls.name: cls
    for cls in (PyMatchingDecoder, UnionFindDecoder, ExperimentalDecoder, CubeCollapseDecoder,
                UnionFind3DDecoder, HierarchicalDecoder, HierarchicalPartialDecoder)
}


def get_decoder(name: str) -> type[Decoder]:
    try:
        return DECODERS[name]
    except KeyError:
        raise KeyError(f"unknown decoder {name!r}; choose from {sorted(DECODERS)}") from None


__all__ = ["DECODERS", "Decoder", "get_decoder"]
