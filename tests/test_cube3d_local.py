"""Step 2: Cube3D-Local fast path."""

import numpy as np
import pytest

from src.circuits import NOISE_TIERS, tier_circuit
from src.decoders.cube3d.graph import T_DIAGONAL, T_TIME, _components, detector_graph
from src.decoders.cube3d.local import Cube3DLocal

TIER_P = {"code_capacity": 0.05, "phenomenological": 0.01, "circuit": 0.005}
CASES = [(tier, d) for tier in NOISE_TIERS for d in (3, 5, 7)]


def local_for(tier, d, p=None):
    g = detector_graph(tier, d, TIER_P[tier] if p is None else p)
    return g, Cube3DLocal(g)


def dets_for_edges(g, edges):
    dets = np.zeros((len(edges), g.n_det), dtype=bool)
    for i, e in enumerate(edges):
        dets[i, g.edge_u[e]] = True
        if g.edge_v[e] != g.boundary:
            dets[i, g.edge_v[e]] = True
    return dets


def explained(g, res, n_shots):
    """Syndrome produced by the recorded flips, per shot."""
    syn = np.zeros((n_shots, g.n_det + 1), dtype=np.uint8)
    np.bitwise_xor.at(syn, (res.flip_shot, g.edge_u[res.flip_edge]), 1)
    np.bitwise_xor.at(syn, (res.flip_shot, g.edge_v[res.flip_edge]), 1)
    return syn[:, : g.n_det].astype(bool)


@pytest.mark.parametrize("tier,d", CASES)
def test_every_single_component_is_accepted_and_correct(tier, d, report):
    g, loc = local_for(tier, d)
    edges = np.arange(g.n_edges)
    res = loc.run(dets_for_edges(g, edges), record_flips=True)
    assert res.accepted.all(), f"{(~res.accepted).sum()} single components rejected"
    np.testing.assert_array_equal(res.predictions, g.edge_obs[edges])
    np.testing.assert_array_equal(explained(g, res, len(edges)), dets_for_edges(g, edges))
    report.add("cube3d_local_colors", tier, f"d={d}", loc.n_colors, ylabel="conflict-graph colors")


@pytest.mark.parametrize("tier,d", CASES)
def test_full_dem_instructions_never_accepted_wrongly(tier, d, report):
    """A whole DEM error instruction (all its components) as one shot. The fast path
    may reject it, but whatever it accepts must predict the instruction's observables."""
    g, loc = local_for(tier, d)
    dem = tier_circuit(tier, d, TIER_P[tier]).detector_error_model(decompose_errors=True)
    rows, truth = [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        det_row = np.zeros(g.n_det, dtype=bool)
        obs_row = np.zeros(g.n_obs, dtype=bool)
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                det_row[t.val] ^= True
            elif t.is_logical_observable_id():
                obs_row[t.val] ^= True
        rows.append(det_row)
        truth.append(obs_row)
    dets, truth = np.array(rows), np.array(truth)
    res = loc.run(dets)
    wrong = (res.predictions != truth).any(axis=1) & res.accepted
    report.add("cube3d_local_single_fault_accept", tier, f"d={d}", 100 * res.accept_fraction,
               ylabel="% of DEM instructions accepted by fast path")
    assert wrong.sum() == 0


@pytest.mark.parametrize("d", [3, 5, 7])
def test_measurement_error_pair(d):
    g, loc = local_for("circuit", d)
    e = int(np.flatnonzero(g.edge_type == T_TIME)[len(np.flatnonzero(g.edge_type == T_TIME)) // 2])
    u, v = int(g.edge_u[e]), int(g.edge_v[e])
    assert (g.coords[u, :2] == g.coords[v, :2]).all() and abs(g.coords[u, 2] - g.coords[v, 2]) == 1
    res = loc.run(dets_for_edges(g, [e]), record_flips=True)
    assert res.accepted[0] and res.cleared["r1"][0] == 2
    assert res.flip_edge.tolist() == [e]
    np.testing.assert_array_equal(res.predictions[0], g.edge_obs[e])


@pytest.mark.parametrize("d", [3, 5, 7])
def test_hook_error(d):
    """Stim's hook errors: one fault -> two checks 2 sites apart in a line, consecutive rounds."""
    g, loc = local_for("circuit", d)
    off = np.abs(g.coords[g.edge_u, :2] - g.coords[np.minimum(g.edge_v, g.n_det - 1), :2])
    hooks = np.flatnonzero((g.edge_type == T_DIAGONAL) & (((off[:, 0] == 4) & (off[:, 1] == 0)) |
                                                           ((off[:, 0] == 0) & (off[:, 1] == 4))))
    assert hooks.size > 0
    res = loc.run(dets_for_edges(g, hooks), record_flips=True)
    assert res.accepted.all() and (res.cleared["r1"] == 2).all()
    np.testing.assert_array_equal(res.predictions, g.edge_obs[hooks])


def test_r2_and_r3_fixtures():
    g, loc = local_for("circuit", 5)
    n = g.n_det
    # R3: one boundary edge alone
    be = int(g.boundary_edge[np.flatnonzero(g.boundary_edge >= 0)[0]])
    res = loc.run(dets_for_edges(g, [be]))
    assert res.accepted[0] and res.cleared["r3"][0] == 1
    # R2: two edges through an unlit center c to non-adjacent a, b
    tables = loc.tables
    for c in range(n):
        nb = g.nbr[c][g.nbr[c] >= 0]
        pair = next(((a, b) for a in nb for b in nb if a < b and b not in g.nbr[a]), None)
        if pair is None:
            continue
        a, b = pair
        dets = np.zeros((1, n), dtype=bool)
        dets[0, [a, b]] = True
        res = loc.run(dets, record_flips=True)
        assert res.accepted[0] and res.cleared["r2"][0] == 2
        np.testing.assert_array_equal(explained(g, res, 1)[0], dets[0])
        break
    else:
        pytest.fail("no R2 configuration found")


def test_gate_rejects():
    g, loc = local_for("circuit", 5)
    # more than K lit nodes, all pairwise far apart, would each be decodable -> still rejected by K
    res = loc.run(np.ones((1, g.n_det), dtype=bool), K=6)
    assert not res.accepted[0]
    # a lit triangle / chain of 3 mutually close nodes is not resolvable by the rules
    u = int(np.argmax(g.degree))
    a, b = (int(x) for x in g.nbr[u][:2])
    dets = np.zeros((1, g.n_det), dtype=bool)
    dets[0, [u, a, b]] = True
    res = loc.run(dets)
    assert not res.accepted[0] and not res.predictions[0].any()
    # zero detection events: accepted, no flip
    res = loc.run(np.zeros((3, g.n_det), dtype=bool))
    assert res.accepted.all() and not res.predictions.any()


@pytest.mark.parametrize("tier", NOISE_TIERS)
@pytest.mark.parametrize("d", [3, 5, 7])
def test_accepted_shots_are_syndrome_consistent(tier, d):
    circuit = tier_circuit(tier, d, TIER_P[tier])
    g, loc = local_for(tier, d)
    dets, _ = circuit.compile_detector_sampler(seed=d).sample(3000, separate_observables=True)
    res = loc.run(dets, record_flips=True)
    syn = explained(g, res, len(dets))
    acc = res.accepted
    np.testing.assert_array_equal(syn[acc], dets[acc])
    total = sum(res.cleared[r] for r in ("r1", "r2", "r3"))
    np.testing.assert_array_equal(total[acc], res.lit_count[acc])
    assert (res.lit_count[acc] <= 6).all()
