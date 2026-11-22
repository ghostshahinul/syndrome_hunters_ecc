from itertools import product
from math import prod

import numpy as np
import stim

from src.decoders.faults import read_model
from src.decoders.soft_paths import SoftPathModel


def _joint_mass(rates, outcomes):
    joint = np.zeros((2, 2))
    for state in product((0, 1), repeat=len(rates)):
        probability = prod(p if fired else 1 - p for p, fired in zip(rates, state))
        joint[outcomes(*state)] += probability
    return joint


def test_local_conditional_odds_by_enumerating_elementary_faults():
    dem = stim.DetectorErrorModel(
        "error(0.1) D0 L0 ^ D1\nerror(0.2) D0 L0\nerror(0.3) D1"
    )
    soft = SoftPathModel(read_model(dem), 2, width=1)
    # A common fault toggles both the edge and the auxiliary check, while
    # the other two faults can flip each independently.
    joint = _joint_mass((.1, .2, .3), lambda common, primary, auxiliary:
                        (common ^ auxiliary, common ^ primary))
    np.testing.assert_allclose(soft.evidence.odds[0], joint[:, 1] / joint[:, 0])


def test_one_defect_paths_straddle_the_exact_logical_margin():
    dem = stim.DetectorErrorModel(
        "error(0.01) D0\nerror(0.1) D0 D1 L0\nerror(0.2) D1"
    )
    soft = SoftPathModel(read_model(dem), 2)
    # D0 alone admits precisely two explanations: the left boundary fault,
    # or the internal logical edge together with the right boundary fault.
    joint = _joint_mass((.01, .1, .2), lambda left, link, right:
                        (int((left ^ link, right ^ link) == (1, 0)), link))
    even, odd = joint[1]
    margin = abs(odd - even) / (odd + even)
    syndrome = np.array([[True, False]])
    prediction, deferred = soft.predict(syndrome, margin - 1e-6)
    assert not deferred[0] and prediction[0] == (odd > even)
    _, deferred = soft.predict(syndrome, margin + 1e-6)
    assert deferred[0]
