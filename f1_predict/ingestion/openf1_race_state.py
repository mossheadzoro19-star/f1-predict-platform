from __future__ import annotations

"""Build a point-in-time race-state training dataset from OpenF1.

Only observations available at the snapshot timestamp are joined into features.
The final session result is used only as the race winner label.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd

from f1_predict.data_sources.openf1 import OpenF1Client


def _ts(value: Any) -> pd.Timestamp | None:
    if not value:
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _latest_by_driver(
    rows: list[dict[str, Any]],
    time_key: str,
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    if not rows or snapshots.empty:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "driver_number" not in frame or time_key not in frame:
        return pd.DataFrame()
    frame["observation_time"] = frame[time_key].map(_ts)
    frame = frame.dropna(subset=["driver_number", "observation_time"]).copy()
    frame["driver_number"] = frame["driver_number"].astype(int)

    # pandas merge_asof requires the merge key itself to be globally sorted.
    # Sorting by driver first causes "left keys must be sorted" on multi-driver data.
    frame = frame.sort_values(["observation_time", "driver_number"])
    left = snapshots[["driver_number", "snapshot_time"]].copy()
    left["driver_number"] = left["driver_number"].astype(int)
    left = left.sort_values(["snapshot_time", "driver_number"])

    return pd.merge_asof(
        left,
        frame,
        left_on="snapshot_time",
        right_on="observation_time",
        by="driver_number",
        direction="backward",
    )


def _latest_weather(
    rows: list[dict[str, Any]],
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    if not rows or snapshots.empty:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "date" not in frame:
        return pd.DataFrame()
    frame["observation_time"] = frame["date"].map(_ts)
    frame = frame.dropna(subset=["observation_time"]).sort_values("observation_time")
    left = snapshots[["snapshot_time"]].drop_duplicates().sort_values("snapshot_time")
    return pd.merge_asof(
        left,
        frame,
        left_on="snapshot_time",
        right_on="observation_time",
        direction="backward",
    )


def _latest_stint(
    rows: list[dict[str, Any]],
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    if not rows or snapshots.empty:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if not {"driver_number", "lap_start"}.issubset(frame.columns):
        return pd.DataFrame()
    frame["driver_number"] = pd.to_numeric(frame["driver_number"], errors="coerce")
    frame["lap_start"] = pd.to_numeric(frame["lap_start"], errors="coerce")
    frame = frame.dropna(subset=["driver_number", "lap_start"]).copy()
    frame["driver_number"] = frame["driver_number"].astype(int)
    frame["lap_start"] = frame["lap_start"].astype(int)

    # The asof key is lap_number, so it must be globally sorted before
    # grouping by driver.
    frame = frame.sort_values(["lap_start", "driver_number"])
    left = snapshots[["driver_number", "lap_number", "snapshot_time"]].copy()
    left["driver_number"] = left["driver_number"].astype(int)
    left["lap_number"] = left["lap_number"].astype(int)
    left = left.sort_values(["lap_number", "driver_number"])

    return pd.merge_asof(
        left,
        frame,
        left_on="lap_number",
        right_on="lap_start",
        by="driver_number",
        direction="backward",
    )


def _build_session_rows(
    client: OpenF1Client,
    session: dict[str, Any],
    request_pause: float,
) -> pd.DataFrame:
    session_key = int(session["session_key"])
    laps = client.get_laps(session_key)
    results = client.get_session_result(session_key)
    winners = {
        int(row["driver_number"])
        for row in results
        if row.get("position") == 1 and not row.get("dnf", False)
    }
    if not winners:
        return pd.DataFrame()

    lap_frame = pd.DataFrame(laps)
    if lap_frame.empty or "date_start" not in lap_frame:
        return pd.DataFrame()
    lap_frame["snapshot_time"] = lap_frame["date_start"].map(_ts)
    lap_frame["lap_number"] = pd.to_numeric(
        lap_frame.get("lap_number"), errors="coerce"
    )
    lap_frame["driver_number"] = pd.to_numeric(
        lap_frame.get("driver_number"), errors="coerce"
    )
    lap_frame = lap_frame.dropna(
        subset=["snapshot_time", "lap_number", "driver_number"]
    ).copy()
    lap_frame["driver_number"] = lap_frame["driver_number"].astype(int)
    lap_frame["lap_number"] = lap_frame["lap_number"].astype(int)
    snapshots = lap_frame[
        ["driver_number", "lap_number", "snapshot_time"]
    ].drop_duplicates()

    joined = snapshots.copy()

    position_rows = client.get_positions(session_key)
    time.sleep(request_pause)
    position = _latest_by_driver(position_rows, "date", snapshots)
    if not position.empty:
        keep = [
            column for column in position.columns
            if column not in {"snapshot_time", "driver_number", "observation_time"}
        ]
        position = position[["driver_number", "snapshot_time", *keep]].rename(
            columns={column: f"position_{column}" for column in keep}
        )
        joined = joined.merge(
            position, on=["driver_number", "snapshot_time"], how="left"
        )

    interval_rows = client.get_intervals(session_key)
    time.sleep(request_pause)
    interval = _latest_by_driver(interval_rows, "date", snapshots)
    if not interval.empty:
        keep = [
            column for column in interval.columns
            if column not in {"snapshot_time", "driver_number", "observation_time"}
        ]
        interval = interval[["driver_number", "snapshot_time", *keep]].rename(
            columns={column: f"interval_{column}" for column in keep}
        )
        joined = joined.merge(
            interval, on=["driver_number", "snapshot_time"], how="left"
        )

    stint_rows = client.get_stints(session_key)
    time.sleep(request_pause)
    stint = _latest_stint(stint_rows, snapshots)
    if not stint.empty:
        keep = [
            column for column in stint.columns
            if column not in {"snapshot_time", "driver_number"}
        ]
        stint = stint[["driver_number", "snapshot_time", *keep]].rename(
            columns={column: f"stint_{column}" for column in keep}
        )
        joined = joined.merge(
            stint, on=["driver_number", "snapshot_time"], how="left"
        )

    weather_rows = client.get_weather(session_key)
    time.sleep(request_pause)
    weather = _latest_weather(weather_rows, snapshots)
    if not weather.empty:
        weather = weather.rename(
            columns={
                column: f"weather_{column}"
                for column in weather.columns
                if column not in {"snapshot_time", "observation_time"}
            }
        )
        joined = joined.merge(weather, on="snapshot_time", how="left")

    joined["season"] = int(session["year"])
    joined["session_key"] = session_key
    joined["meeting_key"] = session.get("meeting_key")
    joined["race_name"] = session.get("meeting_name") or session.get("location")
    joined["winner"] = joined["driver_number"].isin(winners).astype("int8")
    joined["prediction_time"] = joined["snapshot_time"]

    columns = [
        "season", "meeting_key", "session_key", "race_name",
        "driver_number", "lap_number", "prediction_time", "winner",
        "position_position", "interval_gap_to_leader", "interval_interval",
        "stint_compound", "stint_tyre_age_at_start",
        "weather_air_temperature", "weather_track_temperature",
        "weather_humidity", "weather_rainfall", "weather_wind_speed",
    ]
    for column in columns:
        if column not in joined:
            joined[column] = pd.NA
    return joined[columns].sort_values(
        ["prediction_time", "lap_number", "driver_number"]
    )


def build_race_state_dataset(
    start_year: int,
    end_year: int,
    output_path: Path,
    request_pause: float = 1.0,
) -> pd.DataFrame:
    """Download real historical OpenF1 race observations and write Parquet."""
    client = OpenF1Client()
    all_rows: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        sessions = [
            item for item in client.get_sessions(year)
            if item.get("session_name") == "Race"
            and not item.get("is_cancelled", False)
        ]
        for session in sessions:
            try:
                frame = _build_session_rows(client, session, request_pause)
            except Exception as exc:
                print(
                    f"SKIP {year} {session.get('location')}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            if not frame.empty:
                all_rows.append(frame)
            time.sleep(request_pause)

    if not all_rows:
        raise RuntimeError("No OpenF1 race-state data was collected.")

    dataset = pd.concat(all_rows, ignore_index=True)
    dataset = dataset.sort_values(
        ["prediction_time", "season", "session_key", "driver_number"]
    ).reset_index(drop=True)
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
        default=Path("data/processed/features/openf1_race_state.parquet"),
    )
    parser.add_argument("--request-pause", type=float, default=1.0)
    args = parser.parse_args()
    frame = build_race_state_dataset(
        args.start_year, args.end_year, args.output, args.request_pause
    )
    print(f"Rows: {len(frame)}")
    print(f"Races: {frame['session_key'].nunique()}")
    print(f"Drivers: {frame['driver_number'].nunique()}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
