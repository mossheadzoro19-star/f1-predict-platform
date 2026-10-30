import pandas as pd
import pytest

from f1_predict.features.pre_race import build_pre_race_features


def _inputs():
    races = pd.DataFrame(
        [
            {
                "season": 2020, "round": 1, "race_name": "Race A",
                "circuit_id": "track-a", "date": "2020-03-01",
            },
            {
                "season": 2020, "round": 2, "race_name": "Race B",
                "circuit_id": "track-a", "date": "2020-03-08",
            },
        ]
    )
    results = pd.DataFrame(
        [
            {
                "season": 2020, "round": 1, "driver_id": "driver-a",
                "constructor_id": "team-a", "grid": 1, "position": 1, "points": 25, "status": "Finished",
            },
            {
                "season": 2020, "round": 1, "driver_id": "driver-b",
                "constructor_id": "team-b", "grid": 2, "position": 2, "points": 18, "status": "Finished",
            },
            {
                "season": 2020, "round": 2, "driver_id": "driver-a",
                "constructor_id": "team-a", "grid": 2, "position": 2, "points": 18, "status": "Finished",
            },
            {
                "season": 2020, "round": 2, "driver_id": "driver-b",
                "constructor_id": "team-b", "grid": 1, "position": 1, "points": 25, "status": "Finished",
            },
        ]
    )
    qualifying = pd.DataFrame(
        [
            {
                "season": 2020, "round": 1, "driver_id": "driver-a",
                "position": 1, "q1": "1:20.000", "q2": "1:19.500", "q3": "1:19.000",
            },
            {
                "season": 2020, "round": 1, "driver_id": "driver-b",
                "position": 2, "q1": "1:21.000", "q2": "1:20.500", "q3": "1:20.000",
            },
            {
                "season": 2020, "round": 2, "driver_id": "driver-a",
                "position": 2, "q1": "1:20.100", "q2": "1:19.600", "q3": "1:19.100",
            },
            {
                "season": 2020, "round": 2, "driver_id": "driver-b",
                "position": 1, "q1": "1:20.900", "q2": "1:20.400", "q3": "1:19.900",
            },
        ]
    )
    return races, results, qualifying


def test_features_use_only_prior_race_history():
    races, results, qualifying = _inputs()

    features = build_pre_race_features(races, results, qualifying)
    first_race = features[features["round"] == 1]
    second_race = features[features["round"] == 2]

    assert first_race["driver_prior_starts"].eq(0).all()
    assert first_race["constructor_prior_starts"].eq(0).all()

    driver_a = second_race[second_race["driver_id"] == "driver-a"].iloc[0]
    assert driver_a["driver_prior_starts"] == 1
    assert driver_a["driver_prior_wins"] == 1
    assert driver_a["driver_prior_win_rate"] == 1.0
    assert driver_a["driver_finish_last_3"] == 1.0
    assert driver_a["driver_circuit_prior_starts"] == 1
    assert driver_a["constructor_prior_starts"] == 1
    assert driver_a["constructor_prior_wins"] == 1
    assert driver_a["constructor_prior_points"] == 25.0


def test_winner_label_is_one_per_race_and_qualifying_time_is_seconds():
    races, results, qualifying = _inputs()

    features = build_pre_race_features(races, results, qualifying)

    assert features.groupby(["season", "round"])["winner"].sum().eq(1).all()
    driver_a = features[
        (features["round"] == 1) & (features["driver_id"] == "driver-a")
    ].iloc[0]
    assert driver_a["q3_seconds"] == 79.0
    assert driver_a["winner"] == 1


def test_duplicate_race_driver_results_are_rejected():
    races, results, qualifying = _inputs()
    results = pd.concat([results, results.iloc[[0]]], ignore_index=True)

    try:
        build_pre_race_features(races, results, qualifying)
    except ValueError as exc:
        assert "duplicate race/driver" in str(exc)
    else:
        raise AssertionError("Expected duplicate race-driver rows to be rejected")



def test_enhanced_features_are_point_in_time_and_teammate_relative():
    races, results, qualifying = _inputs()
    results.loc[
        (results["round"] == 1) & (results["driver_id"] == "driver-b"),
        "status",
    ] = "Retired"

    features = build_pre_race_features(races, results, qualifying)
    first = features[features["round"] == 1]
    second = features[features["round"] == 2]

    # No current-race outcomes may appear in the same race's prior-history features.
    assert first["driver_prior_dnf_rate"].eq(0.0).all()
    assert first["constructor_prior_dnf_rate"].eq(0.0).all()
    assert first["constructor_points_last_3"].eq(0.0).all()

    driver_a_first = first[first["driver_id"] == "driver-a"].iloc[0]
    driver_b_first = first[first["driver_id"] == "driver-b"].iloc[0]
    assert driver_a_first["field_size"] == 2
    assert driver_a_first["qualifying_gap_to_pole_seconds"] == 0.0
    assert pd.isna(driver_a_first["teammate_q3_gap_seconds"])
    assert driver_b_first["qualifying_gap_to_pole_seconds"] == 1.0

    driver_a_second = second[second["driver_id"] == "driver-a"].iloc[0]
    driver_b_second = second[second["driver_id"] == "driver-b"].iloc[0]
    assert driver_a_second["driver_prior_dnf_rate"] == 0.0
    assert driver_b_second["driver_prior_dnf_rate"] == 1.0
    assert driver_a_second["constructor_points_last_3"] == 25.0
    assert driver_a_second["constructor_prior_dnf_rate"] == 0.0
    assert driver_b_second["constructor_prior_dnf_rate"] == 1.0
    assert driver_a_second["teammate_q3_gap_seconds"] == pytest.approx(-0.8)
