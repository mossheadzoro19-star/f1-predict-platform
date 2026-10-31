import pandas as pd
import pytest

from f1_predict.features.pre_race import (
    _constructor_lineage_id,
    build_pre_race_features,
)


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
    assert pd.isna(driver_a_second["teammate_q3_gap_seconds"])

def test_teammate_q3_gap_is_computed_for_actual_teammates():
    races, results, qualifying = _inputs()
    # Make both drivers members of the same constructor for round 2.
    results.loc[
        (results["round"] == 2) & (results["driver_id"] == "driver-b"),
        "constructor_id",
    ] = "team-a"

    features = build_pre_race_features(races, results, qualifying)
    second = features[features["round"] == 2]
    driver_a = second[second["driver_id"] == "driver-a"].iloc[0]
    driver_b = second[second["driver_id"] == "driver-b"].iloc[0]

    assert driver_a["teammate_q3_gap_seconds"] == pytest.approx(-0.8)
    assert driver_b["teammate_q3_gap_seconds"] == pytest.approx(0.8)



def test_constructor_relative_points_share_is_point_in_time():
    races, results, qualifying = _inputs()

    features = build_pre_race_features(races, results, qualifying)
    first = features[features["round"] == 1]
    second = features[features["round"] == 2]

    # First race has no prior constructor performance.
    assert first["constructor_points_share_last_3"].isna().all()
    assert first["constructor_points_share_last_5"].isna().all()
    assert first["constructor_points_share_trend_3v3"].isna().all()

    # Round 2 can only see each constructor's Round 1 points share.
    team_a = second[second["constructor_id"] == "team-a"].iloc[0]
    team_b = second[second["constructor_id"] == "team-b"].iloc[0]
    assert team_a["constructor_points_share_last_3"] == pytest.approx(25 / 43)
    assert team_b["constructor_points_share_last_3"] == pytest.approx(18 / 43)
    assert team_a["constructor_points_share_last_5"] == pytest.approx(25 / 43)
    assert pd.isna(team_a["constructor_points_share_trend_3v3"])


def test_constructor_lineage_features_inherit_only_prior_alias_history():
    races, results, qualifying = _inputs()
    races.loc[1, "season"] = 2021
    races.loc[1, "round"] = 1
    races.loc[1, "date"] = "2021-03-08"

    results.loc[0, "constructor_id"] = "toro_rosso"
    results.loc[1, "constructor_id"] = "ferrari"
    results.loc[2, "season"] = 2021
    results.loc[2, "round"] = 1
    results.loc[2, "constructor_id"] = "alphatauri"
    results.loc[3, "season"] = 2021
    results.loc[3, "round"] = 1
    results.loc[3, "constructor_id"] = "ferrari"

    qualifying.loc[2, "season"] = 2021
    qualifying.loc[2, "round"] = 1
    qualifying.loc[3, "season"] = 2021
    qualifying.loc[3, "round"] = 1

    features = build_pre_race_features(races, results, qualifying)
    current = features[
        (features["season"] == 2021) & (features["driver_id"] == "driver-a")
    ].iloc[0]

    # Exact source-ID history remains isolated, while experimental lineage
    # history carries the prior Toro Rosso performance into AlphaTauri.
    assert current["constructor_id"] == "alphatauri"
    assert current["constructor_lineage_id"] == "rb_sister_team_lineage"
    assert current["constructor_prior_starts"] == 0
    assert current["constructor_prior_points"] == 0.0
    assert current["lineage_prior_starts"] == 1
    assert current["lineage_prior_wins"] == 1
    assert current["lineage_prior_points"] == 25.0
    assert current["lineage_points_last_3"] == 25.0
    assert current["lineage_points_share_last_3"] == pytest.approx(25 / 43)


def test_red_bull_racing_is_not_merged_with_rb_sister_team():
    assert _constructor_lineage_id("rb") == "rb_sister_team_lineage"
    assert _constructor_lineage_id("red_bull") == "red_bull"
