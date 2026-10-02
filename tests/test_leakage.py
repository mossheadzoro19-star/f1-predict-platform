import pandas as pd
import pytest

from f1_predict.features.leakage import (
    check_first_start_features,
    check_one_winner_per_race,
    check_prediction_timestamps,
    check_prior_counts_monotonic,
    check_unique_driver_race_keys,
)


def _valid_frame():
    return pd.DataFrame(
        [
            {
                "season": 2024, "round": 1, "driver_id": "a",
                "constructor_id": "x", "circuit_id": "c",
                "prediction_time": pd.Timestamp("2024-03-01T12:00:00Z"),
                "winner": 1, "driver_prior_starts": 0,
                "constructor_prior_starts": 0, "driver_circuit_prior_starts": 0,
                "driver_finish_position_last_3": None,
                "driver_finish_position_last_5": None,
            },
            {
                "season": 2024, "round": 1, "driver_id": "b",
                "constructor_id": "y", "circuit_id": "c",
                "prediction_time": pd.Timestamp("2024-03-01T12:00:00Z"),
                "winner": 0, "driver_prior_starts": 0,
                "constructor_prior_starts": 0, "driver_circuit_prior_starts": 0,
                "driver_finish_position_last_3": 4.0,
                "driver_finish_position_last_5": 4.0,
            },
            {
                "season": 2024, "round": 2, "driver_id": "a",
                "constructor_id": "x", "circuit_id": "d",
                "prediction_time": pd.Timestamp("2024-03-08T12:00:00Z"),
                "winner": 0, "driver_prior_starts": 1,
                "constructor_prior_starts": 1, "driver_circuit_prior_starts": 0,
                "driver_finish_position_last_3": 1.0,
                "driver_finish_position_last_5": 1.0,
            },
            {
                "season": 2024, "round": 2, "driver_id": "b",
                "constructor_id": "y", "circuit_id": "d",
                "prediction_time": pd.Timestamp("2024-03-08T12:00:00Z"),
                "winner": 1, "driver_prior_starts": 1,
                "constructor_prior_starts": 1, "driver_circuit_prior_starts": 0,
                "driver_finish_position_last_3": 4.0,
                "driver_finish_position_last_5": 4.0,
            },
        ]
    )


def test_unique_driver_race_keys_pass():
    check_unique_driver_race_keys(_valid_frame())


def test_duplicate_driver_race_is_rejected():
    frame = pd.concat([_valid_frame(), _valid_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="Duplicate driver-race"):
        check_unique_driver_race_keys(frame)


def test_one_winner_per_race_passes():
    check_one_winner_per_race(_valid_frame())


def test_multiple_winners_are_rejected():
    frame = _valid_frame()
    frame.loc[1, "winner"] = 1
    with pytest.raises(AssertionError, match="exactly one winner"):
        check_one_winner_per_race(frame)


def test_first_driver_start_has_zero_prior_starts():
    check_first_start_features(_valid_frame())


def test_nonzero_first_driver_start_is_rejected():
    frame = _valid_frame()
    frame.loc[0, "driver_prior_starts"] = 1
    with pytest.raises(AssertionError, match="First driver starts"):
        check_first_start_features(frame)


def test_prior_counts_are_monotonic():
    check_prior_counts_monotonic(_valid_frame())


def test_decreasing_prior_count_is_rejected():
    frame = _valid_frame()
    frame.loc[2, "driver_prior_starts"] = -1
    with pytest.raises(AssertionError, match="decreases"):
        check_prior_counts_monotonic(frame)


def test_prediction_timestamps_are_present_and_sorted():
    check_prediction_timestamps(_valid_frame())


def test_unsorted_prediction_timestamps_are_rejected():
    frame = _valid_frame().iloc[::-1].reset_index(drop=True)
    with pytest.raises(AssertionError, match="chronological"):
        check_prediction_timestamps(frame)
