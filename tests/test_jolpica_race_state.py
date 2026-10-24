from __future__ import annotations

import pandas as pd

from f1_predict.ingestion.jolpica_race_state import _build_race, _parse_seconds


def _payload(laps: list[dict]) -> dict:
    return {"MRData": {"RaceTable": {"Races": [{"Laps": laps}]}}}


def test_parse_jolpica_lap_time_to_seconds() -> None:
    assert _parse_seconds("1:30.500") == 90.5
    assert _parse_seconds("92.25") == 92.25


def test_build_race_state_uses_lap_times_and_final_result_only_for_target() -> None:
    schedule = {
        "date": "2023-03-05",
        "time": "15:00:00Z",
        "raceName": "Bahrain Grand Prix",
    }
    results = {
        "Results": [
            {"position": "1", "Driver": {"driverId": "driver_a"}},
            {"position": "2", "Driver": {"driverId": "driver_b"}},
        ]
    }
    laps = _payload(
        [
            {
                "number": "1",
                "Timings": [
                    {"driverId": "driver_a", "time": "1:30.000"},
                    {"driverId": "driver_b", "time": "1:31.000"},
                ],
            },
            {
                "number": "2",
                "Timings": [
                    {"driverId": "driver_a", "time": "1:32.000"},
                    {"driverId": "driver_b", "time": "1:33.000"},
                ],
            },
        ]
    )
    pits = _payload([])

    frame = _build_race(
        2023,
        1,
        schedule,
        results,
        laps,
        pits,
    )

    assert len(frame) == 4
    assert frame["session_key"].nunique() == 1
    assert frame["winner"].sum() == 2
    assert set(frame["driver_number"]) == {"driver_a", "driver_b"}

    lap_one = frame.loc[frame["lap_number"] == 1].set_index("driver_number")
    assert lap_one.loc["driver_a", "position_position"] == 1.0
    assert lap_one.loc["driver_b", "position_position"] == 2.0
    assert lap_one.loc["driver_b", "interval_gap_to_leader"] == 1.0

    lap_two = frame.loc[frame["lap_number"] == 2].set_index("driver_number")
    assert lap_two.loc["driver_a", "previous_lap_duration"] == 90.0
    assert lap_two.loc["driver_b", "previous_lap_duration"] == 91.0
    assert pd.to_datetime(frame["prediction_time"], utc=True).notna().all()


def test_build_race_state_returns_empty_without_lap_records() -> None:
    frame = _build_race(
        2023,
        1,
        {"date": "2023-03-05", "raceName": "Empty GP"},
        {"Results": [{"position": "1", "Driver": {"driverId": "driver_a"}}]},
        _payload([]),
        _payload([]),
    )

    assert frame.empty

def test_multilap_features_do_not_cross_race_boundaries() -> None:
    results = {
        "Results": [
            {"position": "1", "Driver": {"driverId": "driver_a"}},
            {"position": "2", "Driver": {"driverId": "driver_b"}},
        ]
    }
    laps = _payload(
        [
            {
                "number": "1",
                "Timings": [
                    {"driverId": "driver_a", "time": "1:30.000"},
                    {"driverId": "driver_b", "time": "1:31.000"},
                ],
            },
            {
                "number": "2",
                "Timings": [
                    {"driverId": "driver_a", "time": "1:32.000"},
                    {"driverId": "driver_b", "time": "1:33.000"},
                ],
            },
        ]
    )

    first_race = _build_race(
        2023,
        1,
        {"date": "2023-03-05", "time": "15:00:00Z", "raceName": "Race One"},
        results,
        laps,
        _payload([]),
    )
    second_race = _build_race(
        2023,
        2,
        {"date": "2023-03-12", "time": "15:00:00Z", "raceName": "Race Two"},
        results,
        laps,
        _payload([]),
    )

    combined = pd.concat([first_race, second_race], ignore_index=True)

    # Each race has only two laps, so a three-lap shift must be missing
    # everywhere. A non-null value would mean a shift crossed sessions.
    assert combined["position_change_3_laps"].isna().all()
    assert combined["gap_change_3_laps"].isna().all()
    assert combined["interval_change_3_laps"].isna().all()

