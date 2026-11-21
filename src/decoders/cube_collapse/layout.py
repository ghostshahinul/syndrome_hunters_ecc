"""Stage 0: map detectors to (round, site) and build the raw syndrome cube.

Stim's rotated memory_{basis} experiment has T = rounds + 1 detector layers:
  t = 0          only the basis-type checks (the other type starts in a random state)
  t = 1..r-1     both check types (we keep only the basis-type sites)
  t = r          final data-readout layer: data parity XOR last ancilla measurement
The selected sites are exactly those with a detector at t = 0. CubeLayout checks this
structure and raises LayoutError with an explanation if a circuit doesn't match it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

Site = tuple[float, float]


class LayoutError(ValueError):
    """The detector layout doesn't match the rotated memory-experiment structure."""


@dataclass(frozen=True, eq=False)
class CubeLayout:
    """sites[i] = (x, y) of node i (sorted row-major by (y, x)).
    det_index[t, i] = detector id of site i in layer t; the last layer is the final
    data-readout layer."""

    sites: tuple[Site, ...]
    det_index: np.ndarray  # (T_full, n_sites) int64

    @property
    def n_sites(self) -> int:
        return len(self.sites)

    @property
    def n_layers(self) -> int:
        return self.det_index.shape[0]

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> "CubeLayout":
        return cls.from_coords(dem.get_detector_coordinates())

    @classmethod
    def from_circuit(cls, circuit: stim.Circuit) -> "CubeLayout":
        return cls.from_coords(circuit.get_detector_coordinates())

    @classmethod
    def from_coords(cls, coords: dict[int, list[float]]) -> "CubeLayout":
        if not coords:
            raise LayoutError("circuit has no detectors")
        bad = [d for d, c in coords.items() if len(c) < 3]
        if bad:
            raise LayoutError(f"detectors {bad[:5]} lack (x, y, t) coordinates")

        ts = sorted({c[2] for c in coords.values()})
        if ts != [float(i) for i in range(len(ts))]:
            raise LayoutError(f"expected time coordinates 0..T-1, got {ts}")

        sites = sorted({(c[0], c[1]) for c in coords.values() if c[2] == 0},
                       key=lambda s: (s[1], s[0]))
        site_id = {s: i for i, s in enumerate(sites)}
        index = np.full((len(ts), len(sites)), -1, dtype=np.int64)
        for det, c in coords.items():
            i = site_id.get((c[0], c[1]))
            if i is None:
                continue  # other check type (e.g. X checks in memory_z): not used
            t = int(c[2])
            if index[t, i] != -1:
                raise LayoutError(f"site {sites[i]} has two detectors in layer t={t}")
            index[t, i] = det
        missing = np.argwhere(index < 0)
        if missing.size:
            t, i = missing[0]
            raise LayoutError(
                f"site {sites[i]} has no detector in layer t={t}; expected every t=0 "
                f"site in every layer (rotated memory experiment)"
            )
        return cls(tuple(sites), index)


def build_cube(
    detection_events: np.ndarray, layout: CubeLayout, include_final_layer: bool = True
) -> np.ndarray:
    """Detection events (shots, num_detectors) -> raw ancilla values (shots, T, n_sites).

    A detection event is m_t XOR m_{t-1} (with m_{-1} = 0), so a cumulative XOR over
    time recovers the raw ancilla value m_t relative to the noiseless outcome. The last
    slice is the data-derived parity when include_final_layer is True; otherwise it is
    dropped and T = rounds.
    """
    events = np.asarray(detection_events, dtype=np.uint8)[:, layout.det_index]
    raw = np.bitwise_xor.accumulate(events, axis=1)
    return raw if include_final_layer else raw[:, :-1]
