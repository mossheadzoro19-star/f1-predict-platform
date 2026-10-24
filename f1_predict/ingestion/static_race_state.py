from __future__ import annotations

"""Build race-state snapshots from a static, no-key F1 results archive.

This is an alternative to Jolpica's rate-limited API. The archive provides
historical lap times, lap-end positions, race metadata, results and pit stops.
Weather and tyre compounds are intentionally left missing when unavailable.
"""

import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import httpx

ARCHIVE_URL = (
    "https://github.com/TracingInsights/RaceData/releases/latest/download/data.zip"
)
DEFAULT_CACHE = Path("data/raw/external/tracinginsights_racedata.zip")
REQUIRED_FILES = {"races.csv", "results.csv", "lap_times.csv"}


def _read_archive(path: Path, url: str = ARCHIVE_URL) -> dict[str, pd.DataFrame]:
    """Load required CSV tables from a cached/downloaded ZIP archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        print(f"Downloading static F1 archive from {url}", flush=True)
        with httpx.Client(timeout=180.0, follow_redirects=True) as client:
            response = client.get(url, headers={"User-Agent": "F1PredictPlatform/0.1"})
            response.raise_for_status()
        if not response.content.startswith(b"PK"):
            raise ValueError("The static F1 data URL did not return a ZIP archive.")
        path.write_bytes(response.content)

    with ZipFile(path) as archive:
        members = {
            Path(name).name.lower(): name
            for name in archive.namelist()
            if Path(name).name.lower().endswith(".csv")
        }
        missing = REQUIRED_FILES - set(members)
        if missing:
            raise ValueError(
                f"Static F1 archive is missing required CSV files: {sorted(missing)}"
            )

        tables: dict[str, pd.DataFrame] = {}
        for name in REQUIRED_FILES | {"pit_stops.csv"}:
            member = members.get(name)
            if member is None:
                if name == "pit_stops.csv":
                    tables[name] = pd.DataFrame(
                        columns=["raceId", "driverId", "lap", "stop"]
                    )
                    continue
                raise ValueError(f"Missing {name} in static F1 archive.")
            with archive.open(member) as stream:
                tables[name] = pd.read_csv(stream, low_memory=False)
    return tables


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_race_state_from_tables(
    laps: pd.DataFrame,
    races: pd.DataFrame,
    results: pd.DataFrame,
    pit_stops: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Transform normalized static tables into the platform's live-feature schema."""
    required = {"raceId", "driverId", "lap", "position"}
    if not required.issubset(laps.columns):
        raise ValueError(f"lap_times table must contain columns: {sorted(required)}")
    if not {"raceId", "year", "round", "name", "date"}.issubset(races.columns):
        raise ValueError("races table is missing required race metadata columns.")
    if not {"raceId", "driverId"}.issubset(results.columns):
        raise ValueError("results table must contain raceId and driverId.")

    selected_races = races.loc[
        _numeric(races["year"]).between(start_year, end_year)
    ].copy()
    if selected_races.empty:
        raise ValueError(f"No races found for seasons {start_year}-{end_year}.")

    lap_frame = laps.copy()
    lap_frame["lap"] = _numeric(lap_frame["lap"])
    if "milliseconds" in lap_frame.columns:
        lap_frame["lap_duration"] = _numeric(lap_frame["milliseconds"]) / 1000.0
    elif "time" in lap_frame.columns:
        def parse_time(value: object) -> float:
            text = str(value)
            try:
                if ":" in text:
                    minutes, seconds = text.split(":", 1)
                    return float(minutes) * 60.0 + float(seconds)
                return float(text)
            except (TypeError, ValueError):
                return np.nan
        lap_frame["lap_duration"] = lap_frame["time"].map(parse_time)
    else:
        raise ValueError("lap_times table needs either milliseconds or time.")

    lap_frame["position"] = _numeric(lap_frame["position"])
    result_frame = results.copy()
    result_frame["positionOrder"] = (
        _numeric(result_frame["positionOrder"])
        if "positionOrder" in result_frame.columns
        else _numeric(result_frame.get("position", pd.Series(index=result_frame.index, dtype=float)))
    )
    pit_frame = pit_stops.copy()
    if not pit_frame.empty:
        pit_frame["lap"] = _numeric(pit_frame["lap"])

    race_rows: list[pd.DataFrame] = []
    for race in selected_races.to_dict("records"):
        race_id = race["raceId"]
        year = int(race["year"])
        round_number = int(race["round"])
        race_laps = lap_frame.loc[
            (lap_frame["raceId"] == race_id)
            & lap_frame["lap"].notna()
            & lap_frame["lap_duration"].gt(0)
        ].copy()
        if race_laps.empty:
            continue

        race_results = result_frame.loc[result_frame["raceId"] == race_id]
        winners = race_results.loc[race_results["positionOrder"] == 1, "driverId"]
        if winners.empty:
            continue
        winner_id = str(winners.iloc[0])

        race_laps["driverId"] = race_laps["driverId"].astype(str)
        race_laps["lap"] = race_laps["lap"].astype(int)
        race_laps = race_laps.sort_values(["driverId", "lap"])
        race_laps["cumulative_time"] = race_laps.groupby("driverId")[
            "lap_duration"
        ].cumsum()
        lap_leader_time = race_laps.groupby("lap")["cumulative_time"].transform("min")
        race_laps["gap"] = (race_laps["cumulative_time"] - lap_leader_time).clip(lower=0)

        race_date = pd.to_datetime(race.get("date"), utc=True, errors="coerce")
        race_clock = race.get("time")
        if pd.notna(race_date) and pd.notna(race_clock) and str(race_clock).strip():
            clock = str(race_clock).strip()
            race_date = pd.to_datetime(
                f"{race.get('date')}T{clock}", utc=True, errors="coerce"
            )
        if pd.isna(race_date):
            # Date remains a race-day anchor; no invented time-of-day is claimed.
            race_date = pd.to_datetime(race.get("date"), utc=True, errors="coerce")
        race_laps["prediction_time"] = race_date + pd.to_timedelta(
            race_laps["cumulative_time"], unit="s"
        )

        race_laps["position_position"] = race_laps["position"]
        if race_laps["position_position"].isna().all():
            race_laps["position_position"] = race_laps.groupby("lap")[
                "cumulative_time"
            ].rank(method="first")
        race_laps["winner"] = (race_laps["driverId"] == winner_id).astype(int)
        race_laps["pit_stops_completed"] = 0.0
        race_laps["laps_since_pit"] = race_laps["lap"].astype(float)

        if not pit_frame.empty:
            race_pits = pit_frame.loc[pit_frame["raceId"] == race_id]
            pit_by_driver = {
                str(driver): sorted(
                    int(lap)
                    for lap in driver_pits["lap"].dropna().tolist()
                    if float(lap).is_integer()
                )
                for driver, driver_pits in race_pits.groupby("driverId")
            }
            for index, row in race_laps.iterrows():
                previous_pits = [
                    lap for lap in pit_by_driver.get(str(row["driverId"]), [])
                    if lap < int(row["lap"])
                ]
                race_laps.at[index, "pit_stops_completed"] = float(len(previous_pits))
                last_pit = max(previous_pits, default=0)
                race_laps.at[index, "laps_since_pit"] = float(int(row["lap"]) - last_pit)

        race_laps["season"] = year
        race_laps["meeting_key"] = year * 100 + round_number
        race_laps["session_key"] = year * 100 + round_number
        race_laps["race_name"] = str(race.get("name", ""))
        race_laps["driver_number"] = race_laps["driverId"].astype(str)
        race_laps["lap_number"] = race_laps["lap"].astype(int)
        race_laps["snapshot_time"] = race_laps["prediction_time"]
        race_laps["interval_gap_to_leader"] = race_laps["gap"]
        race_laps["interval_laps_behind_leader"] = np.nan
        race_laps["interval_laps_behind_car_ahead"] = np.nan
        # Interval to the car ahead follows lap-end position, not driver ID.
        # The leader has no car-ahead interval, so its value remains missing.
        race_laps["interval_interval"] = np.nan
        for _, lap_group in race_laps.groupby("lap", sort=False):
            ordered = lap_group.sort_values(
                ["position_position", "gap"], na_position="last"
            )
            race_laps.loc[ordered.index, "interval_interval"] = ordered["gap"].diff()
        race_laps["stint_compound"] = None
        race_laps["stint_tyre_age_at_start"] = np.nan
        race_laps["previous_lap_duration"] = race_laps.groupby("driverId")[
            "lap_duration"
        ].shift(1)
        race_laps["previous_3_lap_mean"] = race_laps.groupby("driverId")[
            "lap_duration"
        ].transform(lambda values: values.shift(1).rolling(3, min_periods=1).mean())
        race_laps["position_change_1_lap"] = race_laps.groupby("driverId")[
            "position_position"
        ].diff()
        race_laps["gap_change_1_lap"] = race_laps.groupby("driverId")["gap"].diff()
        race_laps["interval_change_1_lap"] = race_laps.groupby("driverId")[
            "interval_interval"
        ].diff()
        for window in (3, 5):
            race_laps[f"position_change_{window}_laps"] = race_laps.groupby("driverId")[
                "position_position"
            ].diff(window)
            race_laps[f"gap_change_{window}_laps"] = race_laps.groupby("driverId")[
                "gap"
            ].diff(window)
            race_laps[f"interval_change_{window}_laps"] = race_laps.groupby("driverId")[
                "interval_interval"
            ].diff(window)
        for column in (
            "weather_air_temperature",
            "weather_track_temperature",
            "weather_humidity",
            "weather_rainfall",
            "weather_wind_speed",
        ):
            race_laps[column] = np.nan
        race_rows.append(race_laps)

    if not race_rows:
        raise RuntimeError(
            f"No lap-state rows could be built for seasons {start_year}-{end_year}."
        )
    output = pd.concat(race_rows, ignore_index=True)
    output = output.drop_duplicates(
        subset=["session_key", "driver_number", "lap_number"], keep="last"
    )
    return output.sort_values(
        ["prediction_time", "season", "session_key", "driver_number"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build historical race-state features without calling Jolpica's API."
    )
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/features/jolpica_race_state_enriched.parquet"),
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()

    tables = _read_archive(args.archive)
    frame = build_race_state_from_tables(
        tables["lap_times.csv"],
        tables["races.csv"],
        tables["results.csv"],
        tables.get("pit_stops.csv", pd.DataFrame()),
        args.start_year,
        args.end_year,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"Rows: {len(frame)}")
    print(f"Races: {frame['session_key'].nunique()}")
    print(f"Seasons: {sorted(frame['season'].unique().tolist())}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
