"""Every registered decoder obeys the Decoder interface in src/decoders/base.py."""

import numpy as np
import pytest

from src.data import sample
from src.decoders import DECODERS, Decoder


@pytest.mark.parametrize("name", sorted(DECODERS))
def test_registered_correctly(name):
    cls = DECODERS[name]
    assert issubclass(cls, Decoder)
    assert cls.name == name


def test_output_shape_and_dtype(decoder, small_circuit):
    s = sample(small_circuit, shots=200, seed=1)
    out = decoder.decode_batch(s.detection_events)
    assert out.shape == (200, decoder.num_observables)
    assert out.dtype == bool


def test_single_shot_matches_batch(decoder, small_circuit):
    s = sample(small_circuit, shots=50, seed=2)
    batch = decoder.decode_batch(s.detection_events)
    for i in range(s.shots):
        np.testing.assert_array_equal(decoder.decode(s.detection_events[i]), batch[i])


def test_deterministic_and_does_not_mutate_input(decoder, small_circuit):
    dets = sample(small_circuit, shots=200, seed=3).detection_events
    original = dets.copy()
    a = decoder.decode_batch(dets)
    b = decoder.decode_batch(dets)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(dets, original)


def test_empty_batch(decoder):
    out = decoder.decode_batch(np.zeros((0, decoder.num_detectors), dtype=bool))
    assert out.shape == (0, decoder.num_observables)


def test_rejects_wrong_width(decoder):
    with pytest.raises(ValueError):
        decoder.decode_batch(np.zeros((1, decoder.num_detectors + 1), dtype=bool))
