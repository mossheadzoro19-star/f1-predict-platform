import pandas as pd

from f1_predict.ingestion.openf1_race_state import (
    _coerce_gap_column,
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
