import pytest

from f1_predict.live.service import RaceIntelligenceService, live_position_weight


def test_live_position_weight_leader_is_one():
    assert live_position_weight(1) == pytest.approx(1.0)


def test_live_position_weight_decreases_with_position():
    assert live_position_weight(2) < live_position_weight(1)
    assert live_position_weight(10) < live_position_weight(2)


def test_live_position_weight_handles_missing_position():
    assert live_position_weight(None) == pytest.approx(1.0)


def test_select_latest_by_driver_is_point_in_time():
    service = object.__new__(RaceIntelligenceService)
    target = service._select_latest_by_driver(
        [
            {"driver_number": 1, "date": "2026-09-26T13:00:00+00:00", "position": 2},
            {"driver_number": 1, "date": "2026-09-26T13:10:00+00:00", "position": 1},
            {"driver_number": 2, "date": "2026-09-26T13:05:00+00:00", "position": 3},
        ],
        "date",
        __import__("datetime").datetime.fromisoformat("2026-09-26T13:07:00+00:00"),
    )
    assert target[1]["position"] == 2
    assert target[2]["position"] == 3


def test_replay_fallback_is_not_marked_as_replayable():
    import pandas as pd

    service = object.__new__(RaceIntelligenceService)
    service.model_season = 2025
    service.features = pd.DataFrame(
        [{"season": 2025, "round": 1, "driver_id": "ver", "grid": 1}]
    )
    service.prior_predictions = {"ver": 1.0}
    service._driver_identity = lambda: {
        "ver": {"code": "VER", "name": "Max Verstappen"}
    }
    result = service._replay_fallback("provider unavailable")
    assert result.mode == "FALLBACK"
    assert result.replay_available is False
