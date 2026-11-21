"""Union-Find decoder (Delfosse & Nickerson, arXiv:1709.06218). TODO: implement.

Suggested plan:
  1. In __init__, turn the DEM into a graph: one node per detector plus one virtual
     boundary node; one edge per error mechanism, carrying its observable mask.
  2. Per shot: start a cluster at every flipped detector, then grow odd clusters by
     half-edges and merge them (union-find with path compression + union by size)
     until every cluster is even or touches the boundary.
  3. Peeling: build a spanning forest of each cluster and peel leaves to get a
     correction; XOR the observable masks of the chosen edges.
If the hot loop is too slow in Python, move it into native/ and keep this class as
the wrapper.
"""

from __future__ import annotations

import numpy as np
import stim

from src.decoders.base import Decoder


class UnionFindDecoder(Decoder):
    name = "union_find"

    def __init__(self, dem: stim.DetectorErrorModel):
        super().__init__(dem)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        raise NotImplementedError("UnionFindDecoder is not implemented yet")
