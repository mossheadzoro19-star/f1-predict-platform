from __future__ import annotations

"""Build a point-in-time race-state training dataset from OpenF1.

Only observations available at the snapshot timestamp are joined into features.
The final session result is used only as the race winner label.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_predict.data_sources.openf1 import OpenF1Client


def _ts(value: Any) -> pd.Timestamp | None:
    if not value:
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _coerce_gap_column(
    frame: pd.DataFrame,
    column: str,
    lap_column: str,
) -> pd.DataFrame:
    """Split OpenF1 gap values into numeric seconds and lap-based gaps.

    OpenF1 can return a numeric gap in seconds or strings such as +1 LAP.
    A single numeric column cannot safely represent both units, so lap gaps
    are kept in a separate feature and the seconds column becomes NaN.
    """
    if column not in frame:
        frame[column] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
        frame[lap_column] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
        return frame

    raw = frame[column].astype("string").str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    lap_gap = pd.to_numeric(
        raw.str.extract(r"([+-]?\d+(?:\.\d+)?)\s*LAPS?")[0],
        errors="coerce",
    )

    frame[column] = numeric.astype("Float64")
    frame[lap_column] = lap_gap.astype("Float64")
    return frame


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


def _add_point_in_time_dynamics(
    joined: pd.DataFrame,
    lap_frame: pd.DataFrame,
    pit_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Add only signals that were knowable before each lap snapshot."""
    frame = joined.sort_values(
        ["driver_number", "lap_number", "snapshot_time"]
    ).copy()
    grouped = frame.groupby("driver_number", sort=False)

    position = pd.to_numeric(frame["position_position"], errors="coerce")
    gap = pd.to_numeric(frame["interval_gap_to_leader"], errors="coerce")
    ahead_gap = pd.to_numeric(frame["interval_interval"], errors="coerce")

    frame["position_change_1_lap"] = position - grouped["position_position"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").shift(1)
    )
    frame["gap_change_1_lap"] = gap - grouped["interval_gap_to_leader"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").shift(1)
    )
    frame["interval_change_1_lap"] = ahead_gap - grouped["interval_interval"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").shift(1)
    )

    for window in (3, 5):
        frame[f"position_change_{window}_laps"] = position - grouped[
            "position_position"
        ].transform(lambda s, w=window: pd.to_numeric(s, errors="coerce").shift(w))
        frame[f"gap_change_{window}_laps"] = gap - grouped[
            "interval_gap_to_leader"
        ].transform(lambda s, w=window: pd.to_numeric(s, errors="coerce").shift(w))
        frame[f"interval_change_{window}_laps"] = ahead_gap - grouped[
            "interval_interval"
        ].transform(lambda s, w=window: pd.to_numeric(s, errors="coerce").shift(w))

    lap_history = lap_frame[["driver_number", "lap_number", "lap_duration"]].copy()
    for column in ("driver_number", "lap_number", "lap_duration"):
        lap_history[column] = pd.to_numeric(lap_history[column], errors="coerce")
    lap_history = lap_history.dropna(subset=["driver_number", "lap_number"]).copy()
    lap_history["driver_number"] = lap_history["driver_number"].astype(int)
    lap_history["lap_number"] = lap_history["lap_number"].astype(int)
    lap_history = lap_history.sort_values(["driver_number", "lap_number"])
    lap_history["previous_lap_duration"] = lap_history.groupby(
        "driver_number"
    )["lap_duration"].shift(1)
    lap_history["previous_3_lap_mean"] = (
        lap_history.groupby("driver_number")["lap_duration"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    frame = frame.merge(
        lap_history[
            ["driver_number", "lap_number", "previous_lap_duration", "previous_3_lap_mean"]
        ],
        on=["driver_number", "lap_number"],
        how="left",
    )

    frame["pit_stops_completed"] = 0.0
    frame["laps_since_pit"] = frame["lap_number"].astype(float)
    pit_frame = pd.DataFrame(pit_rows)
    if not pit_frame.empty and {"driver_number", "lap_number"}.issubset(pit_frame.columns):
        pit_frame["driver_number"] = pd.to_numeric(
            pit_frame["driver_number"], errors="coerce"
        )
        pit_frame["lap_number"] = pd.to_numeric(
            pit_frame["lap_number"], errors="coerce"
        )
        pit_frame = pit_frame.dropna(subset=["driver_number", "lap_number"]).copy()
        pit_frame["driver_number"] = pit_frame["driver_number"].astype(int)
        pit_frame["lap_number"] = pit_frame["lap_number"].astype(int)
        for driver, indexes in frame.groupby("driver_number").groups.items():
            stops = np.sort(
                pit_frame.loc[
                    pit_frame["driver_number"] == int(driver), "lap_number"
                ].unique()
            )
            if len(stops) == 0:
                continue
            laps = frame.loc[indexes, "lap_number"].to_numpy()
            completed = np.searchsorted(stops, laps, side="left")
            frame.loc[indexes, "pit_stops_completed"] = completed
            has_stop = completed > 0
            if has_stop.any():
                last_stop = stops[completed[has_stop] - 1]
                frame.loc[indexes[has_stop], "laps_since_pit"] = (
                    laps[has_stop] - last_stop
                )

    return frame.sort_values(["prediction_time", "lap_number", "driver_number"])


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
        interval = _coerce_gap_column(
            interval,
            "interval_gap_to_leader",
            "interval_laps_behind_leader",
        )
        interval = _coerce_gap_column(
            interval,
            "interval_interval",
            "interval_laps_behind_car_ahead",
        )
        joined = joined.merge(
            interval, on=["driver_number", "snapshot_time"], how="left"
        )

    pit_rows = client.get_pit_stops(session_key)
    time.sleep(request_pause)

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

    joined = _add_point_in_time_dynamics(joined, lap_frame, pit_rows)

    joined["season"] = int(session["year"])
    joined["session_key"] = session_key
    joined["meeting_key"] = session.get("meeting_key")
    joined["race_name"] = session.get("meeting_name") or session.get("location")
    joined["winner"] = joined["driver_number"].isin(winners).astype("int8")
    joined["prediction_time"] = joined["snapshot_time"]

    columns = [
        "season", "meeting_key", "session_key", "race_name",
        "driver_number", "lap_number", "prediction_time", "winner",
        "position_position", "interval_gap_to_leader",
        "interval_laps_behind_leader", "interval_interval",
        "interval_laps_behind_car_ahead", "stint_compound",
        "stint_tyre_age_at_start",
        "position_change_1_lap", "position_change_3_laps", "position_change_5_laps",
        "gap_change_1_lap", "gap_change_3_laps", "gap_change_5_laps",
        "interval_change_1_lap", "interval_change_3_laps", "interval_change_5_laps",
        "previous_lap_duration", "previous_3_lap_mean",
        "pit_stops_completed", "laps_since_pit",
        "weather_air_temperature",
        "weather_track_temperature", "weather_humidity", "weather_rainfall",
        "weather_wind_speed",
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
    # Historical OpenF1 data (2023+) is public; never send live credentials
    # during offline training-data ingestion.
    client = OpenF1Client(authenticated=False)
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

    float_columns = [
        "interval_gap_to_leader",
        "interval_laps_behind_leader",
        "interval_interval",
        "interval_laps_behind_car_ahead",
        "stint_tyre_age_at_start",
        "weather_air_temperature",
        "weather_track_temperature",
        "weather_humidity",
        "weather_rainfall",
        "weather_wind_speed",
    ]
    for column in float_columns:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce").astype(
            "Float64"
        )

    dataset["position_position"] = pd.to_numeric(
        dataset["position_position"], errors="coerce"
    ).astype("Int64")
    dataset["lap_number"] = pd.to_numeric(
        dataset["lap_number"], errors="coerce"
    ).astype("Int64")
    dataset["driver_number"] = pd.to_numeric(
        dataset["driver_number"], errors="coerce"
    ).astype("Int64")
    dataset["winner"] = (
        pd.to_numeric(dataset["winner"], errors="coerce")
        .fillna(0)
        .astype("int8")
    )
    dataset["prediction_time"] = pd.to_datetime(
        dataset["prediction_time"], utc=True
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
