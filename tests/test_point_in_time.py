import pandas as pd
import pytest

from f1_predict.features.point_in_time import (
    _add_constructor_history,
    _parse_lap_time,
    validate_point_in_time_features,
)


def test_parse_lap_time():
    assert _parse_lap_time("1:18.500") == pytest.approx(78.5)
    assert _parse_lap_time("0:59.250") == pytest.approx(59.25)
    assert _parse_lap_time(None) is None


def test_constructor_history_uses_prior_races_only():
    frame = pd.DataFrame(
        [
            {
                "season": 2024, "round": 1, "driver_id": "a", "constructor_id": "x",
                "prediction_time": pd.Timestamp("2024-03-01T12:00:00Z"),
                "points": 8.0, "winner": 1,
            },
            {
                "season": 2024, "round": 1, "driver_id": "b", "constructor_id": "x",
                "prediction_time": pd.Timestamp("2024-03-01T12:00:00Z"),
                "points": 0.0, "winner": 0,
            },
            {
                "season": 2024, "round": 2, "driver_id": "a", "constructor_id": "x",
                "prediction_time": pd.Timestamp("2024-03-08T12:00:00Z"),
                "points": 2.0, "winner": 0,
            },
            {
                "season": 2024, "round": 2, "driver_id": "b", "constructor_id": "x",
                "prediction_time": pd.Timestamp("2024-03-08T12:00:00Z"),
                "points": 1.0, "winner": 0,
            },
        ]
    )

    result = _add_constructor_history(frame)

    first_race = result[result["round"] == 1]
    assert first_race["constructor_points_last_3"].isna().all()
    assert first_race["constructor_points_last_5"].isna().all()
    assert first_race["constructor_prior_win_rate"].isna().all()
    assert (first_race["constructor_prior_starts"] == 0).all()

    second_race = result[result["round"] == 2]
    assert (second_race["constructor_points_last_3"] == 8.0 / 1).all()
    assert (second_race["constructor_points_last_5"] == 8.0 / 1).all()
    assert (second_race["constructor_prior_win_rate"] == 1.0).all()
    assert (second_race["constructor_prior_starts"] == 1).all()


def test_validate_point_in_time_features_accepts_valid_race():
    frame = pd.DataFrame(
        [
            {
                "season": 2025,
                "round": 1,
                "driver_id": "a",
                "constructor_id": "x",
                "circuit_id": "c",
                "prediction_time": pd.Timestamp("2025-03-01T12:00:00Z"),
                "winner": 1,
                "qualifying_position": 1,
                "grid": 1,
                "driver_prior_starts": 0,
                "constructor_prior_starts": 0,
                "driver_circuit_prior_starts": 0,
            },
            {
                "season": 2025,
                "round": 1,
                "driver_id": "b",
                "constructor_id": "y",
                "circuit_id": "c",
                "prediction_time": pd.Timestamp("2025-03-01T12:00:00Z"),
                "winner": 0,
                "qualifying_position": 2,
                "grid": 2,
                "driver_prior_starts": 3,
                "constructor_prior_starts": 4,
                "driver_circuit_prior_starts": 1,
            },
        ]
    )
    validate_point_in_time_features(frame)


def test_validate_rejects_multiple_winners():
    frame = pd.DataFrame(
        [
            {
                "season": 2025,
                "round": 1,
                "driver_id": "a",
                "constructor_id": "x",
                "circuit_id": "c",
                "prediction_time": pd.Timestamp("2025-03-01T12:00:00Z"),
                "winner": 1,
                "qualifying_position": 1,
                "grid": 1,
                "driver_prior_starts": 0,
                "constructor_prior_starts": 0,
                "driver_circuit_prior_starts": 0,
            },
            {
                "season": 2025,
                "round": 1,
                "driver_id": "b",
                "constructor_id": "y",
                "circuit_id": "c",
                "prediction_time": pd.Timestamp("2025-03-01T12:00:00Z"),
                "winner": 1,
                "qualifying_position": 2,
                "grid": 2,
                "driver_prior_starts": 0,
                "constructor_prior_starts": 0,
                "driver_circuit_prior_starts": 0,
            },
        ]
    )
    with pytest.raises(ValueError, match="exactly one winner"):
        validate_point_in_time_features(frame)


def test_validate_rejects_duplicate_driver_race():
    frame = pd.DataFrame(
        [
            {
                "season": 2025,
                "round": 1,
                "driver_id": "a",
                "constructor_id": "x",
                "circuit_id": "c",
                "prediction_time": pd.Timestamp("2025-03-01T12:00:00Z"),
                "winner": 1,
                "qualifying_position": 1,
                "grid": 1,
                "driver_prior_starts": 0,
                "constructor_prior_starts": 0,
                "driver_circuit_prior_starts": 0,
            },
            {
                "season": 2025,
                "round": 1,
                "driver_id": "a",
                "constructor_id": "x",
                "circuit_id": "c",
                "prediction_time": pd.Timestamp("2025-03-01T12:00:00Z"),
                "winner": 0,
                "qualifying_position": 2,
                "grid": 2,
                "driver_prior_starts": 0,
                "constructor_prior_starts": 0,
                "driver_circuit_prior_starts": 0,
            },
        ]
    )
    with pytest.raises(ValueError, match="Duplicate driver-race"):
        validate_point_in_time_features(frame)
