import pytest

from f1_predict.live.service import live_position_weight


def test_live_position_weight_leader_is_one():
    assert live_position_weight(1) == pytest.approx(1.0)


def test_live_position_weight_decreases_with_position():
    assert live_position_weight(2) < live_position_weight(1)
    assert live_position_weight(10) < live_position_weight(2)


def test_live_position_weight_handles_missing_position():
    assert live_position_weight(None) == pytest.approx(1.0)
