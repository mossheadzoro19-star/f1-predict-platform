from __future__ import annotations

"""Build leakage-safe historical race-state data from FastF1."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _seconds(values: pd.Series) -> pd.Series:
    """Convert FastF1 timedeltas or numeric values to seconds."""
    if pd.api.types.is_timedelta64_dtype(values):
        return values.dt.total_seconds()
    return pd.to_numeric(values, errors="coerce")


def _session_key(year: int, round_number: int) -> int:
    """Create a stable internal session key independent of provider IDs."""
    return year * 100 + round_number


def _merge_weather(laps: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    columns = {
        "AirTemp": "weather_air_temperature",
        "TrackTemp": "weather_track_temperature",
        "Humidity": "weather_humidity",
        "Rainfall": "weather_rainfall",
        "WindSpeed": "weather_wind_speed",
    }
    if weather is None or weather.empty or "Time" not in weather:
        for name in columns.values():
            laps[name] = np.nan
        return laps

    right = weather.rename(columns=columns).copy()
    right["session_seconds"] = _seconds(right["Time"])
    keep = ["session_seconds", *[name for name in columns.values() if name in right]]
    right = right[keep].dropna(subset=["session_seconds"]).sort_values("session_seconds")

    left = laps.copy()
    left["_row_order"] = np.arange(len(left))
    merged = pd.merge_asof(
        left.sort_values("session_seconds"),
        right,
        on="session_seconds",
        direction="backward",
    )
    merged = merged.sort_values("_row_order").drop(columns="_row_order")
    for name in columns.values():
        if name not in merged:
            merged[name] = np.nan
    return merged


def _build_session_frame(session: object, year: int, round_number: int) -> pd.DataFrame:
    laps = session.laps.copy()
    required = {"DriverNumber", "LapNumber", "Time", "Position"}
    missing = required - set(laps.columns)
    if laps.empty or missing:
        raise ValueError(f"FastF1 laps missing required columns: {sorted(missing)}")

    laps["driver_number"] = pd.to_numeric(laps["DriverNumber"], errors="coerce")
    laps["lap_number"] = pd.to_numeric(laps["LapNumber"], errors="coerce")
    laps["position_position"] = pd.to_numeric(laps["Position"], errors="coerce")
    laps["session_seconds"] = _seconds(laps["Time"])
    laps["lap_duration"] = _seconds(laps.get("LapTime", pd.Series(index=laps.index)))
    laps = laps.dropna(subset=["driver_number", "lap_number", "session_seconds"]).copy()
    laps["driver_number"] = laps["driver_number"].astype(int)
    laps["lap_number"] = laps["lap_number"].astype(int)

    # Lap-end session time gives a point-in-time clock. Same-lap time deltas
    # provide gap-to-leader and interval-to-car-ahead without future data.
    leader_time = laps.groupby("lap_number")["session_seconds"].transform("min")
    laps["interval_gap_to_leader"] = laps["session_seconds"] - leader_time
    laps = laps.sort_values(["lap_number", "session_seconds", "driver_number"])
    laps["interval_interval"] = laps.groupby("lap_number")["session_seconds"].diff()
    laps["interval_laps_behind_leader"] = np.nan
    laps["interval_laps_behind_car_ahead"] = np.nan

    laps["stint_compound"] = laps.get("Compound", pd.Series(index=laps.index, dtype=object))
    laps["stint_tyre_age_at_start"] = pd.to_numeric(
        laps.get("TyreLife", pd.Series(index=laps.index)), errors="coerce"
    )

    pit_rows: list[dict[str, object]] = []
    if "PitInTime" in laps:
        pit_rows = laps.loc[
            laps["PitInTime"].notna(), ["driver_number", "lap_number"]
        ].to_dict("records")

    if "LapStartDate" in laps:
        prediction_time = pd.to_datetime(laps["LapStartDate"], utc=True, errors="coerce")
        prediction_time += pd.to_timedelta(laps["lap_duration"], unit="s")
    else:
        session_date = getattr(session, "date", None)
        if session_date is None:
            prediction_time = pd.to_datetime(
                laps["session_seconds"], unit="s", origin="unix", utc=True
            )
        else:
            prediction_time = pd.Timestamp(session_date) + pd.to_timedelta(
                laps["session_seconds"], unit="s"
            )

    laps["prediction_time"] = prediction_time
    laps["snapshot_time"] = prediction_time
    laps = _merge_weather(laps, getattr(session, "weather_data", None))

    results = session.results.copy()
    results["driver_number"] = pd.to_numeric(results["DriverNumber"], errors="coerce")
    winners = set(
        results.loc[
            pd.to_numeric(results["Position"], errors="coerce") == 1,
            "driver_number",
        ].dropna().astype(int)
    )
    if len(winners) != 1:
        return pd.DataFrame()

    # Reuse the tested point-in-time transformation rather than duplicating
    # feature logic in the FastF1 provider adapter.
    from f1_predict.ingestion.openf1_race_state import _add_point_in_time_dynamics

    frame = _add_point_in_time_dynamics(laps, laps, pit_rows)
    event = getattr(session, "event", None)
    race_name = getattr(event, "EventName", None) or getattr(event, "Location", "")
    key = _session_key(year, round_number)
    frame["season"] = year
    frame["meeting_key"] = key
    frame["session_key"] = key
    frame["race_name"] = race_name
    frame["winner"] = frame["driver_number"].isin(winners).astype("int8")

    columns = [
        "season", "meeting_key", "session_key", "race_name", "driver_number",
        "lap_number", "prediction_time", "winner", "position_position",
        "interval_gap_to_leader", "interval_laps_behind_leader",
        "interval_interval", "interval_laps_behind_car_ahead",
        "stint_compound", "stint_tyre_age_at_start",
        "position_change_1_lap", "position_change_3_laps", "position_change_5_laps",
        "gap_change_1_lap", "gap_change_3_laps", "gap_change_5_laps",
        "interval_change_1_lap", "interval_change_3_laps", "interval_change_5_laps",
        "previous_lap_duration", "previous_3_lap_mean",
        "pit_stops_completed", "laps_since_pit",
        "weather_air_temperature", "weather_track_temperature",
        "weather_humidity", "weather_rainfall", "weather_wind_speed",
    ]
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    return frame[columns].sort_values(
        ["prediction_time", "lap_number", "driver_number"]
    )


def build_race_state_dataset(
    start_year: int,
    end_year: int,
    output_path: Path,
) -> pd.DataFrame:
    """Load each historical race once and write normalized race-state Parquet."""
    import fastf1

    cache = output_path.parent / "fastf1_cache"
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))

    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, event in schedule.iterrows():
            round_number = int(event["RoundNumber"])
            if round_number <= 0:
                continue
            try:
                session = fastf1.get_session(year, round_number, "R")
                session.load(telemetry=False, weather=True, messages=False)
                frame = _build_session_frame(session, year, round_number)
                if not frame.empty:
                    frames.append(frame)
                print(
                    f"OK {year} R{round_number:02d} {event['EventName']}: "
                    f"{len(frame)} rows"
                )
            except Exception as exc:
                print(
                    f"SKIP {year} R{round_number:02d} {event['EventName']}: "
                    f"{type(exc).__name__}: {exc}"
                )

    if not frames:
        raise RuntimeError("No FastF1 race-state data was collected.")

    dataset = pd.concat(frames, ignore_index=True).sort_values(
        ["prediction_time", "season", "session_key", "driver_number"]
    ).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build historical race-state data with FastF1."
    )
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/features/fastf1_race_state_enriched.parquet"
        ),
    )
    args = parser.parse_args()
    frame = build_race_state_dataset(args.start_year, args.end_year, args.output)
    print(f"Rows: {len(frame)}")
    print(f"Races: {frame['session_key'].nunique()}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
