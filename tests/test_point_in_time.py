import pandas as pd
import pytest

from f1_predict.features.point_in_time import (
    _add_constructor_history,
    _add_historical_features,
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


def test_mutating_current_race_targets_cannot_change_current_features():
    base = pd.DataFrame(
        [
            {
                "season": 2024, "round": 1, "driver_id": "a", "constructor_id": "x",
                "circuit_id": "c1", "prediction_time": pd.Timestamp("2024-03-01T12:00:00Z"),
                "position": 1, "finish_position_numeric": 1.0, "points": 25.0, "winner": 1,
            },
            {
                "season": 2024, "round": 1, "driver_id": "b", "constructor_id": "x",
                "circuit_id": "c1", "prediction_time": pd.Timestamp("2024-03-01T12:00:00Z"),
                "position": 2, "finish_position_numeric": 2.0, "points": 18.0, "winner": 0,
            },
            {
                "season": 2024, "round": 2, "driver_id": "a", "constructor_id": "x",
                "circuit_id": "c2", "prediction_time": pd.Timestamp("2024-03-08T12:00:00Z"),
                "position": 2, "finish_position_numeric": 2.0, "points": 18.0, "winner": 0,
            },
            {
                "season": 2024, "round": 2, "driver_id": "b", "constructor_id": "x",
                "circuit_id": "c2", "prediction_time": pd.Timestamp("2024-03-08T12:00:00Z"),
                "position": 1, "finish_position_numeric": 1.0, "points": 25.0, "winner": 1,
            },
            {
                "season": 2024, "round": 3, "driver_id": "a", "constructor_id": "x",
                "circuit_id": "c3", "prediction_time": pd.Timestamp("2024-03-15T12:00:00Z"),
                "position": 3, "finish_position_numeric": 3.0, "points": 15.0, "winner": 0,
            },
            {
                "season": 2024, "round": 3, "driver_id": "b", "constructor_id": "x",
                "circuit_id": "c3", "prediction_time": pd.Timestamp("2024-03-15T12:00:00Z"),
                "position": 4, "finish_position_numeric": 4.0, "points": 12.0, "winner": 0,
            },
        ]
    )

    mutated = base.copy()
    race_two = mutated["round"] == 2
    mutated.loc[race_two & (mutated["driver_id"] == "a"), ["position", "finish_position_numeric", "points", "winner"]] = [20, 20.0, 0.0, 0]
    mutated.loc[race_two & (mutated["driver_id"] == "b"), ["position", "finish_position_numeric", "points", "winner"]] = [1, 1.0, 25.0, 1]

    original_features = _add_historical_features(base)
    mutated_features = _add_historical_features(mutated)

    historical_columns = [
        "driver_finish_position_last_3",
        "driver_finish_position_last_5",
        "driver_points_last_3",
        "driver_points_last_5",
        "driver_prior_win_rate",
        "driver_prior_starts",
        "constructor_points_last_3",
        "constructor_points_last_5",
        "constructor_prior_win_rate",
        "constructor_prior_starts",
        "driver_circuit_finish_last_3",
        "driver_circuit_finish_last_5",
        "driver_circuit_prior_starts",
    ]

    original_race_two = original_features.loc[original_features["round"] == 2, historical_columns].reset_index(drop=True)
    mutated_race_two = mutated_features.loc[mutated_features["round"] == 2, historical_columns].reset_index(drop=True)

    pd.testing.assert_frame_equal(original_race_two, mutated_race_two)

    original_race_three = original_features.loc[original_features["round"] == 3, historical_columns].reset_index(drop=True)
    mutated_race_three = mutated_features.loc[mutated_features["round"] == 3, historical_columns].reset_index(drop=True)

    assert not original_race_three.equals(mutated_race_three)


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
