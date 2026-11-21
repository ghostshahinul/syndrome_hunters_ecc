"""Decoder registry. Add new decoders to DECODERS so evaluate.py and the tests see them."""

from src.decoders.base import Decoder
from src.decoders.baseline import CorrelatedMatchingDecoder, PyMatchingDecoder
from src.decoders.experimental import FaultJoinDecoder
from src.decoders.union_find import UnionFindDecoder

DECODERS: dict[str, type[Decoder]] = {
    cls.name: cls for cls in (FaultJoinDecoder, PyMatchingDecoder, CorrelatedMatchingDecoder, UnionFindDecoder)
}


def get_decoder(name: str) -> type[Decoder]:
    if name == "experimental":
        return FaultJoinDecoder
    try:
        return DECODERS[name]
    except KeyError:
        raise KeyError(f"unknown decoder {name!r}; choose from {sorted(DECODERS)}") from None


__all__ = ["DECODERS", "Decoder", "get_decoder"]
