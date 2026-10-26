"""Tests for offline trace analysis: AUROC, rule replay, early features."""

import pytest

from energy_search.analysis import auroc, entropy_rule_step, velocity_rule_step, early_features


def test_auroc_perfect_random_and_ties():
    assert auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5
    with pytest.raises(ValueError):
        auroc([0.1, 0.2], [1, 1])


def test_entropy_rule_needs_persistence():
    h = [0.1, 2.0, 2.0, 0.1, 2.0, 2.0, 2.0, 0.1]
    assert entropy_rule_step(h, tau=1.0, k=3) == 6  # third consecutive high-entropy token
    assert entropy_rule_step(h, tau=1.0, k=4) is None


def test_velocity_rule_uses_past_window_only():
    v = [1.0] * 20 + [5.0] * 5
    step = velocity_rule_step(v, tau=2.5, k=2, window=16)
    assert step == 21  # z is measured against the past window, so a sustained jump keeps firing
    assert velocity_rule_step([1.0] * 30, tau=2.5, k=2) is None


def test_early_features_window():
    trace = {"v_rel": [1.0, 2.0, 3.0, 100.0], "entropy": [0.0, 1.0, 2.0, 100.0], "cos": [0.0] * 4}
    f = early_features(trace, k=3)
    assert f["v_mean"] == 2.0 and f["h_max"] == 2.0 and f["h_mean"] == 1.0
