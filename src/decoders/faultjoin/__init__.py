"""FaultJoin (teammate branch shahi_2, by ghostshahinul): two-fault logical voting with
syndrome-conditioned union-find. Integrated on branch integration/all-decoders.

The modules keep their shahi_2 contents; only imports moved into this package:
  faults.py      reads both views of a decomposed DEM (whole faults and graph edges)
  join.py        exact syndrome join for explanations with at most two DEM faults
  condition.py   growth costs conditioned on clicks of a correlated partner
  union_find.py  shahi_2's own union-find kernel (private here: main deleted
                 src/decoders/union_find.py, so it is not restored at that path)
  decoder.py     FaultJoinDecoder (registry name "fault_join")
"""

from src.decoders.faultjoin.decoder import FaultJoinDecoder

__all__ = ["FaultJoinDecoder"]
