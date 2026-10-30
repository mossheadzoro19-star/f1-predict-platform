from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_START_YEAR = 2018
DEFAULT_END_YEAR = 2024


def _seconds(value: object) -> float | None:
    """Convert a qualifying time such as '1:23.456' to seconds."""
    if value is None or pd.isna(value) or str(value).strip() in {"", "None"}:
        return None
    try:
        parts = str(value).strip().split(":")
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
    except (TypeError, ValueError):
        return None
    return None


def _mean(values: list[float], count: int) -> float | None:
    recent = values[-count:]
    return sum(recent) / len(recent) if recent else None


def build_pre_race_features(
    races: pd.DataFrame,
    results: pd.DataFrame,
    qualifying: pd.DataFrame,
) -> pd.DataFrame:
    """Build one driver-row per race using only information available pre-race.

    Result position is used only to create the winner label and update history
    after all feature rows for that race have been created.
    """
    required_races = {"season", "round", "race_name", "circuit_id", "date"}
    required_results = {
        "season", "round", "driver_id", "constructor_id", "grid",
        "position", "points",
    }
    required_qualifying = {"season", "round", "driver_id", "position", "q1", "q2", "q3"}
    for name, frame, required in (
        ("races", races, required_races),
        ("results", results, required_results),
        ("qualifying", qualifying, required_qualifying),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

    race_keys = ["season", "round"]
    if races.duplicated(race_keys).any():
        raise ValueError("races contains duplicate season/round keys")
    if results.duplicated(race_keys + ["driver_id"]).any():
        raise ValueError("results contains duplicate race/driver keys")
    if qualifying.duplicated(race_keys + ["driver_id"]).any():
        raise ValueError("qualifying contains duplicate race/driver keys")

    race_rows = races.copy()
    race_rows["_date"] = pd.to_datetime(race_rows["date"], errors="coerce")
    race_rows = race_rows.sort_values(["_date", "season", "round"], kind="stable")

    driver_history: dict[str, dict[str, object]] = defaultdict(
        lambda: {"finishes": [], "points": [], "starts": 0, "wins": 0}
    )
    constructor_history: dict[str, dict[str, object]] = defaultdict(
        lambda: {"points": [], "starts": 0, "wins": 0}
    )
    circuit_history: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {"finishes": [], "starts": 0}
    )

    results_by_race = {
        key: group.copy()
        for key, group in results.groupby(race_keys, sort=False)
    }
    qualifying_by_race = {
        key: group.set_index("driver_id", drop=False)
        for key, group in qualifying.groupby(race_keys, sort=False)
    }

    output: list[dict[str, object]] = []
    for race in race_rows.to_dict("records"):
        key = (int(race["season"]), int(race["round"]))
        race_results = results_by_race.get(key)
        if race_results is None or race_results.empty:
            continue
        quali = qualifying_by_race.get(key)
        circuit_id = str(race["circuit_id"])

        # Build every row before updating any history from this race.
        race_output: list[dict[str, object]] = []
        for result in race_results.to_dict("records"):
            driver_id = str(result["driver_id"])
            constructor_id = str(result["constructor_id"])
            dh = driver_history[driver_id]
            ch = constructor_history[constructor_id]
            cih = circuit_history[(driver_id, circuit_id)]

            qrow = None
            if quali is not None and driver_id in quali.index:
                qrow = quali.loc[driver_id]
                if isinstance(qrow, pd.DataFrame):
                    raise ValueError(f"Duplicate qualifying row for {key}, {driver_id}")

            position = result.get("position")
            numeric_position = float(position) if pd.notna(position) else None
            q_position = None if qrow is None else qrow.get("position")
            row = {
                "season": key[0],
                "round": key[1],
                "race_name": race.get("race_name"),
                "circuit_id": circuit_id,
                "race_date": race.get("date"),
                "driver_id": driver_id,
                "constructor_id": constructor_id,
                "grid": result.get("grid"),
                "qualifying_position": q_position,
                "q1_seconds": None if qrow is None else _seconds(qrow.get("q1")),
                "q2_seconds": None if qrow is None else _seconds(qrow.get("q2")),
                "q3_seconds": None if qrow is None else _seconds(qrow.get("q3")),
                "driver_prior_starts": int(dh["starts"]),
                "driver_prior_wins": int(dh["wins"]),
                "driver_prior_win_rate": (
                    float(dh["wins"]) / int(dh["starts"]) if dh["starts"] else 0.0
                ),
                "driver_finish_last_3": _mean(dh["finishes"], 3),
                "driver_finish_last_5": _mean(dh["finishes"], 5),
                "driver_points_last_3": sum(dh["points"][-3:]),
                "driver_points_last_5": sum(dh["points"][-5:]),
                "constructor_prior_starts": int(ch["starts"]),
                "constructor_prior_wins": int(ch["wins"]),
                "constructor_prior_points": sum(ch["points"]),
                "constructor_prior_win_rate": (
                    float(ch["wins"]) / int(ch["starts"]) if ch["starts"] else 0.0
                ),
                "driver_circuit_prior_starts": int(cih["starts"]),
                "driver_circuit_finish_last_3": _mean(cih["finishes"], 3),
                "winner": int(numeric_position == 1.0) if numeric_position is not None else 0,
            }
            race_output.append(row)
        output.extend(race_output)

        # Update state only after all entrants' pre-race features are frozen.
        constructor_race_points: dict[str, float] = defaultdict(float)
        constructor_race_wins: dict[str, int] = defaultdict(int)
        for result in race_results.to_dict("records"):
            driver_id = str(result["driver_id"])
            constructor_id = str(result["constructor_id"])
            position = result.get("position")
            points = result.get("points")
            numeric_position = float(position) if pd.notna(position) else None
            numeric_points = float(points) if pd.notna(points) else 0.0

            dh = driver_history[driver_id]
            dh["starts"] += 1
            dh["points"].append(numeric_points)
            if numeric_position is not None:
                dh["finishes"].append(numeric_position)
            if numeric_position == 1.0:
                dh["wins"] += 1

            cih = circuit_history[(driver_id, circuit_id)]
            cih["starts"] += 1
            if numeric_position is not None:
                cih["finishes"].append(numeric_position)

            constructor_race_points[constructor_id] += numeric_points
            if numeric_position == 1.0:
                constructor_race_wins[constructor_id] += 1

        for constructor_id, race_points in constructor_race_points.items():
            ch = constructor_history[constructor_id]
            ch["starts"] += 1
            ch["points"].append(race_points)
            ch["wins"] += constructor_race_wins[constructor_id]

    frame = pd.DataFrame(output)
    if frame.empty:
        raise ValueError("No race-driver rows were built")
    if frame.duplicated(["season", "round", "driver_id"]).any():
        raise AssertionError("Output contains duplicate race-driver rows")
    return frame.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)


def load_seasons(start_year: int, end_year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    races, results, qualifying = [], [], []
    for year in range(start_year, end_year + 1):
        folder = ROOT / "data" / "processed" / "jolpica" / str(year)
        for name, target in (
            ("races.parquet", races),
            ("race_results.parquet", results),
            ("qualifying.parquet", qualifying),
        ):
            path = folder / name
            if not path.exists():
                raise FileNotFoundError(f"Missing normalized input: {path}")
            target.append(pd.read_parquet(path))
    return (
        pd.concat(races, ignore_index=True),
        pd.concat(results, ignore_index=True),
        pd.concat(qualifying, ignore_index=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build point-in-time pre-race driver features.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        help="Defaults to 2024 so the 2025 final holdout stays untouched.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "features" / "pre_race_driver_features.parquet",
    )
    args = parser.parse_args()

    races, results, qualifying = load_seasons(args.start_year, args.end_year)
    features = build_pre_race_features(races, results, qualifying)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    print(f"Built {len(features):,} driver-race rows")
    print(f"Seasons: {features['season'].min()}–{features['season'].max()}")
    print(f"Races: {features[['season', 'round']].drop_duplicates().shape[0]:,}")
    print(f"Winners: {int(features['winner'].sum()):,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
