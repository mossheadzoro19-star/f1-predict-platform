import pandas as pd

from f1_predict.ingestion.openf1_race_state import _latest_by_driver


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
