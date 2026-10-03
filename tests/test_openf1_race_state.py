import pandas as pd

from f1_predict.ingestion.openf1_race_state import (
    _coerce_gap_column,
    _add_point_in_time_dynamics,
    _latest_by_driver,
)


def test_latest_by_driver_respects_snapshot_time():
    snapshots = pd.DataFrame(
        {
            "driver_number": [1, 1],
            "snapshot_time": pd.to_datetime(
                ["2025-01-01T00:00:05Z", "2025-01-01T00:00:15Z"]
            ),
        }
    )
    rows = [
        {"driver_number": 1, "date": "2025-01-01T00:00:01Z", "position": 5},
        {"driver_number": 1, "date": "2025-01-01T00:00:10Z", "position": 2},
        {"driver_number": 1, "date": "2025-01-01T00:00:20Z", "position": 1},
    ]
    result = _latest_by_driver(rows, "date", snapshots)
    assert result["position"].tolist() == [5, 2]


def test_latest_by_driver_sorts_asof_key_across_drivers():
    snapshots = pd.DataFrame(
        {
            "driver_number": [1, 2, 1, 2],
            "snapshot_time": pd.to_datetime(
                [
                    "2025-01-01T00:00:15Z",
                    "2025-01-01T00:00:05Z",
                    "2025-01-01T00:00:05Z",
                    "2025-01-01T00:00:15Z",
                ]
            ),
        }
    )
    rows = [
        {"driver_number": 2, "date": "2025-01-01T00:00:02Z", "position": 8},
        {"driver_number": 1, "date": "2025-01-01T00:00:01Z", "position": 7},
        {"driver_number": 2, "date": "2025-01-01T00:00:10Z", "position": 4},
        {"driver_number": 1, "date": "2025-01-01T00:00:10Z", "position": 3},
    ]
    result = _latest_by_driver(rows, "date", snapshots)
    result = result.sort_values(["driver_number", "snapshot_time"])
    assert result["position"].tolist() == [7, 3, 8, 4]


def test_coerce_gap_column_handles_seconds_and_laps():
    frame = pd.DataFrame({"interval_gap_to_leader": ["1.234", "+1 LAP", "+2 LAPS"]})
    result = _coerce_gap_column(
        frame,
        "interval_gap_to_leader",
        "interval_laps_behind_leader",
    )
    assert result["interval_gap_to_leader"].tolist() == [1.234, pd.NA, pd.NA]
    assert result["interval_laps_behind_leader"].tolist() == [pd.NA, 1.0, 2.0]


def test_point_in_time_dynamics_use_only_prior_observations():
    joined = pd.DataFrame(
        {
            "driver_number": [1, 1, 1, 1],
            "lap_number": [1, 2, 3, 4],
            "snapshot_time": pd.to_datetime(
                [
                    "2025-01-01T00:00:01Z",
                    "2025-01-01T00:01:31Z",
                    "2025-01-01T00:03:01Z",
                    "2025-01-01T00:04:31Z",
                ]
            ),
            "prediction_time": pd.to_datetime(
                [
                    "2025-01-01T00:00:01Z",
                    "2025-01-01T00:01:31Z",
                    "2025-01-01T00:03:01Z",
                    "2025-01-01T00:04:31Z",
                ]
            ),
            "position_position": [1, 2, 1, 1],
            "interval_gap_to_leader": [0.0, 1.0, 0.5, 0.2],
            "interval_interval": [0.0, 1.0, 0.5, 0.2],
        }
    )
    lap_frame = pd.DataFrame(
        {
            "driver_number": [1, 1, 1, 1],
            "lap_number": [1, 2, 3, 4],
            "lap_duration": [90.0, 91.0, 92.0, 93.0],
        }
    )
    result = _add_point_in_time_dynamics(joined, lap_frame, [])
    result = result.sort_values("lap_number").reset_index(drop=True)

    assert pd.isna(result.loc[0, "previous_lap_duration"])
    assert result.loc[1, "previous_lap_duration"] == 90.0
    assert result.loc[1, "position_change_1_lap"] == 1.0
    assert result.loc[2, "gap_change_1_lap"] == -0.5
    assert result.loc[3, "previous_3_lap_mean"] == 91.0


def test_pit_stop_state_excludes_current_lap():
    joined = pd.DataFrame(
        {
            "driver_number": [1, 1, 1],
            "lap_number": [10, 11, 12],
            "snapshot_time": pd.to_datetime(
                [
                    "2025-01-01T00:10:00Z",
                    "2025-01-01T00:11:00Z",
                    "2025-01-01T00:12:00Z",
                ]
            ),
            "prediction_time": pd.to_datetime(
                [
                    "2025-01-01T00:10:00Z",
                    "2025-01-01T00:11:00Z",
                    "2025-01-01T00:12:00Z",
                ]
            ),
            "position_position": [1, 2, 3],
            "interval_gap_to_leader": [0.0, 1.0, 2.0],
            "interval_interval": [0.0, 1.0, 1.0],
        }
    )
    lap_frame = pd.DataFrame(
        {
            "driver_number": [1, 1, 1],
            "lap_number": [10, 11, 12],
            "lap_duration": [90.0, 91.0, 92.0],
        }
    )
    result = _add_point_in_time_dynamics(
        joined,
        lap_frame,
        [{"driver_number": 1, "lap_number": 11}],
    ).sort_values("lap_number").reset_index(drop=True)

    assert result["pit_stops_completed"].tolist() == [0.0, 0.0, 1.0]
    assert result["laps_since_pit"].tolist() == [10.0, 11.0, 1.0]
