from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _read_season(year: int) -> dict[str, pd.DataFrame]:
    base = ROOT / "data" / "processed" / "jolpica" / str(year)
    required = ["races", "drivers", "constructors", "qualifying", "race_results"]
    missing = [name for name in required if not (base / f"{name}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"Missing normalized files for {year}: {', '.join(missing)}")
    return {name: pd.read_parquet(base / f"{name}.parquet") for name in required}


def _parse_lap_time(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = parts
            return float(minutes) * 60.0 + float(seconds)
        if len(parts) == 1:
            return float(parts[0])
    except ValueError:
        return None
    return None


def _add_prior_rolling(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    output_prefix: str,
    windows: tuple[int, ...] = (3, 5),
) -> pd.DataFrame:
    frame = frame.copy()
    grouped = frame.groupby(group_col, sort=False)[value_col]

    for window in windows:
        frame[f"{output_prefix}_last_{window}"] = (
            grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    return frame


def _add_prior_rate(
    frame: pd.DataFrame,
    group_col: str,
    event_col: str,
    output_col: str,
) -> pd.DataFrame:
    frame = frame.copy()
    event = frame[event_col].astype(float)
    frame[output_col] = (
        event.groupby(frame[group_col], sort=False)
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    return frame


def _add_prior_count(
    frame: pd.DataFrame,
    group_col: str,
    output_col: str,
) -> pd.DataFrame:
    frame = frame.copy()
    frame[output_col] = frame.groupby(group_col, sort=False).cumcount()
    return frame


def _add_constructor_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Add constructor history at race granularity to prevent same-race leakage."""
    frame = frame.copy()

    constructor_races = (
        frame.groupby(["season", "round", "constructor_id"], as_index=False)
        .agg(
            constructor_race_points=("points", "sum"),
            constructor_race_winner=("winner", "max"),
            prediction_time=("prediction_time", "min"),
        )
        .sort_values(["prediction_time", "season", "round", "constructor_id"])
        .reset_index(drop=True)
    )

    grouped_points = constructor_races.groupby("constructor_id", sort=False)[
        "constructor_race_points"
    ]
    for window in (3, 5):
        constructor_races[f"constructor_points_last_{window}"] = (
            grouped_points.transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
        )

    event = constructor_races["constructor_race_winner"].astype(float)
    constructor_races["constructor_prior_win_rate"] = (
        event.groupby(constructor_races["constructor_id"], sort=False)
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )

    constructor_races["constructor_prior_starts"] = (
        constructor_races.groupby("constructor_id", sort=False).cumcount()
    )

    history = constructor_races[
        [
            "season",
            "round",
            "constructor_id",
            "constructor_points_last_3",
            "constructor_points_last_5",
            "constructor_prior_win_rate",
            "constructor_prior_starts",
        ]
    ]

    frame = frame.drop(
        columns=[
            "constructor_points_last_3",
            "constructor_points_last_5",
            "constructor_prior_win_rate",
            "constructor_prior_starts",
        ],
        errors="ignore",
    )

    return frame.merge(
        history,
        on=["season", "round", "constructor_id"],
        how="left",
        validate="many_to_one",
    )


def _prepare_season(year: int) -> pd.DataFrame:
    """Prepare one season without calculating historical features."""
    data = _read_season(year)
    races = data["races"].copy()
    results = data["race_results"].copy()
    qualifying = data["qualifying"].copy()

    races["prediction_time"] = pd.to_datetime(
        races["date"].astype(str) + " " + races["time"].fillna("00:00:00Z").astype(str),
        utc=True,
        errors="coerce",
    )

    q = qualifying[
        ["season", "round", "driver_id", "constructor_id", "position", "q1", "q2", "q3"]
    ].copy()
    for col in ("q1", "q2", "q3"):
        q[f"{col}_seconds"] = q[col].map(_parse_lap_time)

    q = q.rename(columns={"position": "qualifying_position"})

    r = results[
        [
            "season", "round", "driver_id", "constructor_id", "position",
            "points", "grid", "laps", "status"
        ]
    ].copy()

    frame = r.merge(
        q[
            [
                "season", "round", "driver_id", "qualifying_position",
                "q1_seconds", "q2_seconds", "q3_seconds"
            ]
        ],
        on=["season", "round", "driver_id"],
        how="left",
        validate="one_to_one",
    )
    frame = frame.merge(
        races[["season", "round", "circuit_id", "prediction_time"]],
        on=["season", "round"],
        how="left",
        validate="many_to_one",
    )

    frame["winner"] = (frame["position"] == 1).astype("int8")
    frame["finish_position_numeric"] = pd.to_numeric(frame["position"], errors="coerce")
    frame["field_size"] = frame.groupby(["season", "round"])["driver_id"].transform("size")
    return frame


def _add_historical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate all historical features across the complete global timeline."""
    frame = frame.copy()
    frame = frame.sort_values(
        ["prediction_time", "season", "round", "driver_id"]
    ).reset_index(drop=True)

    # Driver history is naturally one row per driver per race, so shift(1)
    # always moves to a genuinely earlier race for that driver.
    frame = _add_prior_rolling(
        frame, "driver_id", "finish_position_numeric", "driver_finish_position"
    )
    frame = _add_prior_rolling(frame, "driver_id", "points", "driver_points")
    frame = _add_prior_rate(frame, "driver_id", "winner", "driver_prior_win_rate")
    frame = _add_prior_count(frame, "driver_id", "driver_prior_starts")

    # Constructor history must be calculated at constructor-race granularity.
    # Calculating directly on driver rows would let the second driver inherit
    # the first driver's current-race result, which is target leakage.
    frame = _add_constructor_history(frame)

    frame = _add_prior_rolling(
        frame,
        ["driver_id", "circuit_id"],
        "finish_position_numeric",
        "driver_circuit_finish",
    )
    frame = _add_prior_count(
        frame, ["driver_id", "circuit_id"], "driver_circuit_prior_starts"
    )

    return frame


def build_point_in_time_features(
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")

    frames = [_prepare_season(year) for year in range(start_year, end_year + 1)]
    result = pd.concat(frames, ignore_index=True)

    # Historical features must be computed after concatenating all seasons.
    # Otherwise every season would incorrectly reset prior driver/constructor history.
    return _add_historical_features(result)


def validate_point_in_time_features(frame: pd.DataFrame) -> None:
    required = {
        "season", "round", "driver_id", "constructor_id", "circuit_id",
        "prediction_time", "winner", "qualifying_position", "grid",
        "driver_prior_starts", "constructor_prior_starts",
        "driver_circuit_prior_starts",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    if frame.empty:
        raise ValueError("Feature dataset is empty")

    if frame[["season", "round", "driver_id"]].duplicated().any():
        raise ValueError("Duplicate driver-race feature keys detected")

    if not frame["winner"].isin([0, 1]).all():
        raise ValueError("Winner target must be binary")

    winners_per_race = frame.groupby(["season", "round"])["winner"].sum()
    if not (winners_per_race == 1).all():
        bad = winners_per_race[winners_per_race != 1].to_dict()
        raise ValueError(f"Expected exactly one winner per race: {bad}")

    prior_cols = [
        "driver_prior_starts",
        "constructor_prior_starts",
        "driver_circuit_prior_starts",
    ]
    if (frame[prior_cols] < 0).any().any():
        raise ValueError("Prior-count features cannot be negative")


def write_features(frame: pd.DataFrame, name: str = "race_driver_features") -> Path:
    out_dir = ROOT / "data" / "processed" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe point-in-time F1 driver-race features."
    )
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    args = parser.parse_args()

    frame = build_point_in_time_features(args.start_year, args.end_year)
    validate_point_in_time_features(frame)
    path = write_features(frame)

    print("Point-in-time feature build complete:")
    print(f"  Rows: {len(frame)}")
    print(f"  Races: {frame[['season', 'round']].drop_duplicates().shape[0]}")
    print(f"  Columns: {len(frame.columns)}")
    print(f"  Output: {path}")


if __name__ == "__main__":
    main()
