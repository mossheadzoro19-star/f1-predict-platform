from __future__ import annotations

"""Build historical race-state observations from Jolpica lap timing."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from f1_predict.data_sources.jolpica import JolpicaClient


def _parse_seconds(value: object) -> float:
    if value is None:
        return np.nan
    text = str(value)
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return float(minutes) * 60.0 + float(seconds)
    try:
        return float(text)
    except (TypeError, ValueError):
        return np.nan


def _race_date(schedule_race: dict[str, object]) -> pd.Timestamp:
    date = schedule_race.get("date")
    time = schedule_race.get("time") or "00:00:00Z"
    return pd.to_datetime(f"{date}T{time}", utc=True, errors="coerce")


def _extract_race(payload: dict[str, object]) -> dict[str, object]:
    races = (
        payload.get("MRData", {})
        .get("RaceTable", {})
        .get("Races", [])
    )
    if not races:
        return {}
    return races[0]


def _build_race(
    year: int,
    round_number: int,
    schedule_race: dict[str, object],
    results_race: dict[str, object],
    laps_race: dict[str, object],
    pits_race: dict[str, object],
) -> pd.DataFrame:
    laps = _extract_race(laps_race).get("Laps", [])
    if not laps:
        return pd.DataFrame()

    winner = next(
        (
            str(row.get("Driver", {}).get("driverId"))
            for row in results_race.get("Results", [])
            if str(row.get("position")) == "1"
        ),
        None,
    )
    if winner is None:
        return pd.DataFrame()

    pit_rows = _extract_race(pits_race).get("PitStops", [])
    pit_laps: dict[str, list[int]] = {}
    for pit in pit_rows:
        driver = str(pit.get("driverId", ""))
        try:
            pit_laps.setdefault(driver, []).append(int(pit["lap"]))
        except (KeyError, ValueError):
            continue

    rows: list[dict[str, object]] = []
    race_start = _race_date(schedule_race)
    cumulative: dict[str, float] = {}
    previous_position: dict[str, float] = {}
    previous_gap: dict[str, float] = {}
    history: dict[str, list[float]] = {}

    for lap in laps:
        lap_number = int(lap["number"])
        timings = lap.get("Timings", [])
        state: list[dict[str, object]] = []

        for timing in timings:
            driver = str(timing.get("driverId", ""))
            lap_time = _parse_seconds(timing.get("time"))
            if not driver or not np.isfinite(lap_time):
                continue
            cumulative[driver] = cumulative.get(driver, 0.0) + lap_time
            state.append(
                {
                    "driver_id": driver,
                    "lap_time": lap_time,
                    "cumulative_time": cumulative[driver],
                }
            )
            history.setdefault(driver, []).append(lap_time)

        if not state:
            continue

        # Rank by completed-lap count first, then cumulative elapsed time.
        # This reconstructs race order without using final classification.
        state.sort(key=lambda row: row["cumulative_time"])
        position_by_driver = {
            row["driver_id"]: float(index + 1)
            for index, row in enumerate(state)
        }
        leader_time = min(row["cumulative_time"] for row in state)

        for index, row in enumerate(state):
            driver = str(row["driver_id"])
            cumulative_time = float(row["cumulative_time"])
            position = position_by_driver[driver]
            gap = cumulative_time - leader_time
            prior_laps = history[driver][:-1]

            timestamp = race_start + pd.to_timedelta(cumulative_time, unit="s")\n            interval = (\n                cumulative_time - float(state[index - 1]["cumulative_time"])\n                if index > 0 else 0.0\n            )
            pit_count = sum(
                pit_lap < lap_number for pit_lap in pit_laps.get(driver, [])
            )
            last_pit = max(
                (pit_lap for pit_lap in pit_laps.get(driver, []) if pit_lap < lap_number),
                default=0,
            )

            rows.append(
                {
                    "season": year,
                    "meeting_key": year * 100 + round_number,
                    "session_key": year * 100 + round_number,
                    "race_name": schedule_race.get("raceName", ""),
                    "driver_number": driver,
                    "lap_number": lap_number,
                    "prediction_time": timestamp,
                    "snapshot_time": timestamp,
                    "winner": int(driver == winner),
                    "position_position": position,
                    "interval_gap_to_leader": gap,
                    "interval_laps_behind_leader": np.nan,
                    "interval_interval": (
                        gap - previous_gap.get(driver, gap)
                        if driver in previous_gap
                        else np.nan
                    ),
                    "interval_laps_behind_car_ahead": np.nan,
                    "stint_compound": None,
                    "stint_tyre_age_at_start": np.nan,
                    "lap_duration": lap_time,
                    "previous_lap_duration": prior_laps[-1] if prior_laps else np.nan,
                    "previous_3_lap_mean": (
                        float(np.mean(prior_laps[-3:])) if prior_laps else np.nan
                    ),
                    "pit_stops_completed": float(pit_count),
                    "laps_since_pit": float(lap_number - last_pit),
                    "position_change_1_lap": (
                        previous_position[driver] - position
                        if driver in previous_position
                        else np.nan
                    ),
                    "position_change_3_laps": np.nan,
                    "position_change_5_laps": np.nan,
                    "gap_change_1_lap": (
                        gap - previous_gap[driver]
                        if driver in previous_gap
                        else np.nan
                    ),
                    "gap_change_3_laps": np.nan,
                    "gap_change_5_laps": np.nan,
                    "interval_change_1_lap": np.nan,
                    "interval_change_3_laps": np.nan,
                    "interval_change_5_laps": np.nan,
                    "weather_air_temperature": np.nan,
                    "weather_track_temperature": np.nan,
                    "weather_humidity": np.nan,
                    "weather_rainfall": np.nan,
                    "weather_wind_speed": np.nan,
                }
            )
            previous_position[driver] = position
            previous_gap[driver] = gap

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    # Derive multi-lap dynamics from earlier observations only.
    frame = frame.sort_values(["driver_number", "lap_number", "prediction_time"]).copy()
    grouped = frame.groupby("driver_number", sort=False)
    for window in (3, 5):
        frame[f"position_change_{window}_laps"] = (
            frame["position_position"]
            - grouped["position_position"].shift(window)
        )
        frame[f"gap_change_{window}_laps"] = (
            frame["interval_gap_to_leader"]
            - grouped["interval_gap_to_leader"].shift(window)
        )
        frame[f"interval_change_{window}_laps"] = (
            frame["interval_interval"]
            - grouped["interval_interval"].shift(window)
        )
    return frame


def build_race_state_dataset(
    start_year: int,
    end_year: int,
    output_path: Path,
) -> pd.DataFrame:
    client = JolpicaClient(request_delay_seconds=0.5)
    frames: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        schedule_payload = client.get_season_schedule(year)
        results_payload = client.get_season_results(year)
        # Jolpica's season-level results/schedule contain all races, while
        # lap/pit data are fetched once per race rather than once per driver.
        schedule_rows = (
            schedule_payload.get("MRData", {})
            .get("RaceTable", {})
            .get("Races", [])
        )
        result_rows = (
            results_payload.get("MRData", {})
            .get("RaceTable", {})
            .get("Races", [])
        )
        result_by_round = {str(r.get("round")): r for r in result_rows}

        for schedule_race in schedule_rows:
            round_number = int(schedule_race["round"])
            try:
                result_race = result_by_round[str(round_number)]
                laps = client.get_race_laps(year, round_number)
                pits = client.get_race_pitstops(year, round_number)
                frame = _build_race(
                    year, round_number, schedule_race, result_race, laps, pits
                )
                if not frame.empty:
                    frames.append(frame)
                print(
                    f"OK {year} R{round_number:02d} "
                    f"{schedule_race.get('raceName', '')}: {len(frame)} rows"
                )
            except Exception as exc:
                print(
                    f"SKIP {year} R{round_number:02d}: "
                    f"{type(exc).__name__}: {exc}"
                )

    if not frames:
        raise RuntimeError("No Jolpica race-state data was collected.")

    dataset = pd.concat(frames, ignore_index=True).sort_values(
        ["prediction_time", "season", "session_key", "driver_number"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/features/jolpica_race_state_enriched.parquet"),
    )
    args = parser.parse_args()
    frame = build_race_state_dataset(args.start_year, args.end_year, args.output)
    print(f"Rows: {len(frame)}")
    print(f"Races: {frame['session_key'].nunique()}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
