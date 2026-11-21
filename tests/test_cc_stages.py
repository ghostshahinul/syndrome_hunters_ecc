"""Cube-Collapse Stage 3 (Clique S1/S2/S3) and Stage 4 (greedy completion) fixtures.

Geometry: the basis checks of the rotated code form a diagonal grid; a bulk node's
neighbors sit at (+-2, +-2). "Opposite" neighbor pairs (a + b == 2c) are the straight
(horizontal/vertical analogue) chains, "bent" pairs are the diagonal analogues.
"""

import itertools

import numpy as np
import pytest

from src.decoders.cube_collapse.graph import BOUNDARY

from cc_helpers import (
    bulk_nodes, cc_decoder, decode_errors, edge_between, errors_to_case, far, path_edges,
)


def cleared(result):
    return {st: int(result.cleared[st].sum()) for st in ("s1", "s2", "s3", "s4")}


# ---- Stage 3: Clique ---------------------------------------------------------------
def test_single_bulk_error():
    dec = cc_decoder(7)
    g = dec.graph
    u = int(bulk_nodes(g)[0])
    v = int(g.nbr[u, 0])
    e = edge_between(g, u, v)
    r, truth = decode_errors(dec, [e])
    assert cleared(r) == {"s1": 2, "s2": 0, "s3": 0, "s4": 0}
    assert np.flatnonzero(r.correction[0]).tolist() == [e]
    np.testing.assert_array_equal(r.predictions, truth)


@pytest.mark.parametrize("kind", ["boundary", "corner"])
def test_boundary_and_corner_error(kind):
    dec = cc_decoder(7)
    g = dec.graph
    has_b = np.flatnonzero(g.boundary_edge >= 0)
    deg = g.degree[has_b]
    u = int(has_b[np.argmin(deg)] if kind == "corner" else has_b[np.argmax(deg)])
    e = int(g.boundary_edge[u])
    r, truth = decode_errors(dec, [e])
    assert cleared(r) == {"s1": 0, "s2": 0, "s3": 1, "s4": 0}
    assert np.flatnonzero(r.correction[0]).tolist() == [e]
    np.testing.assert_array_equal(r.predictions, truth)


def _neighbor_pairs(g, c):
    nbs = [int(x) for x in g.nbr[c] if x >= 0]
    for a, b in itertools.combinations(nbs, 2):
        sa, sb, sc = (np.array(g.sites[i]) for i in (a, b, c))
        yield a, b, ("straight" if np.allclose(sa + sb, 2 * sc) else "bent")


@pytest.mark.parametrize("shape", ["straight", "bent"])
def test_length2_chain_through_unlit_center(shape):
    dec = cc_decoder(7)
    g = dec.graph
    n_cases = 0
    for c in bulk_nodes(g):
        for a, b, kind in _neighbor_pairs(g, int(c)):
            if kind != shape:
                continue
            r, truth = decode_errors(dec, [edge_between(g, c, a), edge_between(g, c, b)])
            assert cleared(r) == {"s1": 0, "s2": 2, "s3": 0, "s4": 0}, (c, a, b)
            np.testing.assert_array_equal(r.predictions, truth)
            n_cases += 1
    assert n_cases >= (2 if shape == "straight" else 4)


def test_four_lit_around_center():
    dec = cc_decoder(7)
    g = dec.graph
    c = int(bulk_nodes(g)[0])
    edges = [int(e) for e in g.nbr_edge[c] if e >= 0]
    r, truth = decode_errors(dec, edges)
    assert cleared(r)["s2"] == 4 and cleared(r)["s4"] == 0
    np.testing.assert_array_equal(r.predictions, truth)


def test_two_separated_errors():
    dec = cc_decoder(9)
    g = dec.graph
    internal = [e for e in range(g.n_edges) if g.edge_v[e] != BOUNDARY]
    for e1, e2 in itertools.combinations(internal, 2):
        n1, n2 = (g.edge_u[e1], g.edge_v[e1]), (g.edge_u[e2], g.edge_v[e2])
        if far(g, n1, n2, 3):
            break
    r, truth = decode_errors(dec, [e1, e2])
    assert cleared(r) == {"s1": 4, "s2": 0, "s3": 0, "s4": 0}
    np.testing.assert_array_equal(r.predictions, truth)


def test_errors_resolved_in_different_stages():
    """One S1 pair, one S2 chain and one S3 boundary error in the same shot (d=9)."""
    dec = cc_decoder(9)
    g = dec.graph
    bulk = bulk_nodes(g)
    c2 = int(bulk[len(bulk) // 2])
    a, b, _ = next(_neighbor_pairs(g, c2))
    s2_edges, s2_nodes = [edge_between(g, c2, a), edge_between(g, c2, b)], [a, b, c2]
    s1 = next((int(u), int(v)) for u in bulk for v in g.nbr[u]
              if v >= 0 and far(g, (u, v), s2_nodes, 3))
    s3 = next(int(u) for u in np.flatnonzero(g.boundary_edge >= 0)
              if far(g, (u,), s2_nodes + list(s1), 3))
    edges = s2_edges + [edge_between(g, *s1), int(g.boundary_edge[s3])]
    r, truth = decode_errors(dec, edges)
    assert cleared(r) == {"s1": 2, "s2": 2, "s3": 1, "s4": 0}
    np.testing.assert_array_equal(r.predictions, truth)


def test_overlapping_cliques_three_in_a_row():
    """a-b-c all lit in a straight line: the cliques at a and c both want b. Exactly one
    S1 fires; the leftover node is resolved later; the syndrome is still satisfied."""
    dec = cc_decoder(9)
    g = dec.graph
    b = int(bulk_nodes(g)[len(bulk_nodes(g)) // 2])
    a, c, _ = next(p for p in _neighbor_pairs(g, b) if p[2] == "straight")
    flat = np.zeros((1, g.n_nodes), dtype=np.uint8)
    flat[0, [a, b, c]] = 1
    r = dec.decode_flat(flat)
    assert (g.syndrome(r.correction) == flat).all()
    st = cleared(r)
    assert st["s1"] == 2 and st["s2"] == 0 and st["s3"] + st["s4"] == 1


# ---- Stage 4: growing-radius greedy completion --------------------------------------
@pytest.mark.parametrize("length,d", [(3, 9), (4, 11)])
def test_chain_resolved_in_stage4(length, d):
    dec = cc_decoder(d)
    g = dec.graph
    n = g.n_nodes
    u, v = next((int(u), int(v)) for u in range(n) for v in range(u + 1, n)
                if g.hops[u, v] == length and g.hops[u, n] > length and g.hops[v, n] > length)
    r, truth = decode_errors(dec, path_edges(g, u, v))
    assert cleared(r) == {"s1": 0, "s2": 0, "s3": 0, "s4": 2}
    assert int(r.radius[0]) == length
    np.testing.assert_array_equal(r.predictions, truth)


def test_lone_node_goes_to_boundary():
    dec = cc_decoder(9)
    g = dec.graph
    n = g.n_nodes
    u = int(np.argmax(g.hops[:, n]))
    far_pair = next((int(x), int(y)) for x in bulk_nodes(g) for y in g.nbr[x]
                    if y >= 0 and far(g, (x, y), (u,), 4))
    edges = path_edges(g, u, BOUNDARY) + [edge_between(g, *far_pair)]
    r, truth = decode_errors(dec, edges)
    assert cleared(r) == {"s1": 2, "s2": 0, "s3": 0, "s4": 1}
    assert int(r.radius[0]) == int(g.hops[u, n])
    np.testing.assert_array_equal(r.predictions, truth)


@pytest.mark.parametrize("d", [5, 7, 9])
def test_dense_clusters_always_resolve(d):
    dec = cc_decoder(d)
    g = dec.graph
    n = g.n_nodes
    rng = np.random.default_rng(d)
    blobs = []
    for center in range(n):  # every node's 2-hop ball fully lit
        f = np.zeros(n, dtype=np.uint8)
        f[g.hops[center, :n] <= 2] = 1
        blobs.append(f)
    flat = np.vstack(blobs + [rng.integers(0, 2, size=(500, n), dtype=np.uint8),
                              np.ones((1, n), dtype=np.uint8)])
    r = dec.decode_flat(flat)
    np.testing.assert_array_equal(g.syndrome(r.correction), flat)
    total = sum(r.cleared[st] for st in ("s1", "s2", "s3", "s4"))
    np.testing.assert_array_equal(total, flat.sum(axis=1))
    assert r.radius.max() <= g.hops[:, n].max()
