from __future__ import annotations

import pandas as pd

from f1_predict.ingestion.static_race_state import build_race_state_from_tables


def test_static_archive_builder_creates_point_in_time_race_rows() -> None:
    races = pd.DataFrame(
        [
            {
                "raceId": 1,
                "year": 2023,
                "round": 1,
                "name": "Example Grand Prix",
                "date": "2023-03-05",
                "time": "15:00:00Z",
            }
        ]
    )
    results = pd.DataFrame(
        [
            {"raceId": 1, "driverId": "driver_a", "positionOrder": 1},
            {"raceId": 1, "driverId": "driver_b", "positionOrder": 2},
        ]
    )
    laps = pd.DataFrame(
        [
            {"raceId": 1, "driverId": "driver_a", "lap": 1, "position": 1, "milliseconds": 90000},
            {"raceId": 1, "driverId": "driver_b", "lap": 1, "position": 2, "milliseconds": 91000},
            {"raceId": 1, "driverId": "driver_a", "lap": 2, "position": 1, "milliseconds": 92000},
            {"raceId": 1, "driverId": "driver_b", "lap": 2, "position": 2, "milliseconds": 93000},
        ]
    )
    pits = pd.DataFrame(columns=["raceId", "driverId", "lap", "stop"])

    frame = build_race_state_from_tables(laps, races, results, pits, 2023, 2023)

    assert len(frame) == 4
    assert frame["session_key"].nunique() == 1
    assert frame["winner"].sum() == 2
    lap_one = frame.loc[frame["lap_number"] == 1].set_index("driver_number")
    assert lap_one.loc["driver_b", "position_position"] == 2
    assert lap_one.loc["driver_b", "interval_gap_to_leader"] == 1.0
    lap_two = frame.loc[frame["lap_number"] == 2].set_index("driver_number")
    assert lap_two.loc["driver_a", "previous_lap_duration"] == 90.0
    assert pd.to_datetime(frame["prediction_time"], utc=True).notna().all()
