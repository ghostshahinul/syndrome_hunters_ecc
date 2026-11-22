"""Cube3D decoders on the full 3D detector graph. See docs/cube3d_spec.md."""

from src.decoders.cube3d.graph import DetectorGraph3D, build_detector_graph, detector_graph

__all__ = ["DetectorGraph3D", "build_detector_graph", "detector_graph"]
