"""Cube-Collapse Stage 2: check graph structure and precomputed shortest paths."""

import numpy as np
import pytest

from src.circuits import phenomenological_circuit, surface_code_circuit
from src.decoders.cube_collapse.clique import color_cliques
from src.decoders.cube_collapse.graph import BOUNDARY, build_check_graph

from cc_helpers import cc_decoder


@pytest.mark.parametrize("basis", ["z", "x"])
@pytest.mark.parametrize("d", [3, 5, 7])
def test_graph_matches_code_capacity(d, basis):
    ref = cc_decoder(d, basis).graph
    for circuit in (surface_code_circuit(d, 0.003, basis=basis),
                    phenomenological_circuit(d, 0.01, basis=basis)):
        g = build_check_graph(circuit.detector_error_model(decompose_errors=True))
        assert g.n_nodes == ref.n_nodes == (d * d - 1) // 2
        assert g.n_edges == ref.n_edges
        assert g.signature() == ref.signature()   # endpoints + observable masks


@pytest.mark.parametrize("d", [3, 5, 7])
def test_edge_counts(d):
    g = cc_decoder(d).graph
    # d^2 data qubits; the 2d qubits on the two relevant boundaries merge into d+1 edges.
    assert g.n_edges == d * d - (d - 1)
    assert int((g.edge_v == BOUNDARY).sum()) == d + 1
    assert (g.edge_weight > 0).all() and np.isfinite(g.edge_weight).all()


@pytest.mark.parametrize("d", [3, 5, 7, 9])
def test_every_node_reaches_boundary(d):
    g = cc_decoder(d).graph
    assert np.isfinite(g.dist[:, g.n_nodes]).all()
    assert (g.hops[:, g.n_nodes] >= 1).all()


@pytest.mark.parametrize("d", [3, 5, 7])
def test_paths_are_consistent(d):
    """Flipping a precomputed path lights exactly its endpoints; dist is symmetric."""
    g = cc_decoder(d).graph
    n = g.n_nodes
    np.testing.assert_allclose(g.dist[:, :n], g.dist[:, :n].T)
    for u in range(n):
        for t in range(n + 1):
            corr = np.zeros((1, g.n_edges), dtype=np.uint8)
            corr[0, g.path_edges[u, t, : g.path_len[u, t]]] ^= 1
            expect = np.zeros(n, dtype=np.uint8)
            if t != u:
                expect[u] = 1
                if t < n:
                    expect[t] = 1
            np.testing.assert_array_equal(g.syndrome(corr)[0], expect)
            assert g.path_len[u, t] == (g.hops[u, t] if t != u else 0)
            np.testing.assert_allclose(g.edge_weight[g.path_edges[u, t, : g.path_len[u, t]]].sum(), g.dist[u, t])


@pytest.mark.parametrize("d", [3, 5, 7, 9])
def test_clique_coloring_is_conflict_free(d, report):
    g = cc_decoder(d).graph
    groups = color_cliques(g)
    closed = [{u, *g.nbr[u][g.nbr[u] >= 0].tolist()} for u in range(g.n_nodes)]
    assert sorted(np.concatenate(groups).tolist()) == list(range(g.n_nodes))
    for centers in groups:
        for i, a in enumerate(centers):
            for b in centers[i + 1:]:
                assert not closed[a] & closed[b]
    report.add("cube_collapse_colors", "cube_collapse", f"d={d}", len(groups), ylabel="clique colors")
