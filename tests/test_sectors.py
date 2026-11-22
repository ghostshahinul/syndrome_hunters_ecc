"""Check that sector regrouping preserves the complete physical fault law."""
import numpy as np
import stim

from src.decoders.sectors import sector_model


def _law(size, faults):
    mass = np.zeros(size)
    mass[0] = 1
    words = np.arange(size)
    for rate, mask in faults:
        mass = (1 - rate) * mass + rate * mass[words ^ mask]
    return mass


def _distribution(dem):
    """Ignore all graph separators and expand each independent Stim fault."""
    events = []
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        mask = 0
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                mask ^= 1 << target.val
            elif target.is_logical_observable_id():
                mask ^= 1 << (dem.num_detectors + target.val)
        events.append((instruction.args_copy()[0], mask))
    return _law(1 << (dem.num_detectors + dem.num_observables), events)


def test_regroup_split_boundary_checks_without_changing_joint_distribution():
    dem = stim.DetectorErrorModel("""
        error(.08) D0 D1
        error(.11) D2 D3
        error(.19) D0 L0 ^ D2 ^ D1 L0 ^ D3 L1
        error(.21) D0 ^ L0
    """)
    result = sector_model(dem)
    before = _law(64, [(.08, 3), (.11, 12), (.19, 15 | 32), (.21, 1 | 16)])
    np.testing.assert_allclose(_distribution(result), before, rtol=1e-14, atol=1e-16)
    groups = next(ins.target_groups() for ins in result if ins.type == "error" and ins.args_copy() == [.19])
    assert len(groups) == 2
    assert sorted(sum(t.is_relative_detector_id() for t in group) for group in groups) == [2, 2]


def test_center_biased_and_deterministic_faults_preserves_affine_law_and_dimensions():
    dem = stim.DetectorErrorModel("""
        error(.12) D0 D1
        error(.5) D1 L0
        error(.8) D0 L0
        error(1) D2 L1
    """)
    centered = sector_model(dem, center=True)
    assert (centered.num_detectors, centered.num_observables) == (3, 2)
    assert all(ins.args_copy()[0] <= .5 for ins in centered if ins.type == "error")
    # Attaching both reference shifts flips D0, D2, L0 and L1.
    shift = 1 | 4 | 8 | 16
    original = _law(32, [(.12, 3), (.5, 2 | 8), (.8, 1 | 8), (1., 4 | 16)])
    np.testing.assert_allclose(_distribution(centered)[np.arange(32) ^ shift], original,
                               rtol=1e-14, atol=1e-16)
