from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_START_YEAR = 2018
DEFAULT_END_YEAR = 2024

# Reviewed organizational lineages for experimental historical features.
# Unmapped constructor IDs retain their own identity. Red Bull Racing remains
# separate from the AlphaTauri/RB sister-team lineage.
CONSTRUCTOR_LINEAGE_IDS = {
    "force_india": "aston_martin_lineage",
    "racing_point": "aston_martin_lineage",
    "aston_martin": "aston_martin_lineage",
    "renault": "alpine_lineage",
    "alpine": "alpine_lineage",
    "toro_rosso": "rb_sister_team_lineage",
    "alphatauri": "rb_sister_team_lineage",
    "rb": "rb_sister_team_lineage",
    "alfa": "sauber_lineage",
    "sauber": "sauber_lineage",
}


def _constructor_lineage_id(constructor_id: str) -> str:
    return CONSTRUCTOR_LINEAGE_IDS.get(constructor_id, constructor_id)


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


def _dnf(status: object) -> int:
    """Return 1 when Jolpica status does not indicate a classified finish."""
    if status is None or pd.isna(status):
        return 0
    text = str(status).strip().lower()
    return int(text not in {"finished"} and not text.startswith("+"))


def build_pre_race_features(
    races: pd.DataFrame,
    results: pd.DataFrame,
    qualifying: pd.DataFrame,
) -> pd.DataFrame:
    """Build strictly point-in-time pre-race driver features.

    Current-race finishing position, points and status are labels/history updates
    only. All entrants' features are frozen before any current-race history update.
    """
    required_races = {"season", "round", "race_name", "circuit_id", "date"}
    required_results = {
        "season", "round", "driver_id", "constructor_id", "grid",
        "position", "points", "status",
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
    if race_rows["_date"].isna().any():
        raise ValueError("Race calendar contains missing or invalid dates")
    race_rows = race_rows.sort_values(["_date", "season", "round"], kind="stable")

    driver_history: dict[str, dict[str, object]] = defaultdict(
        lambda: {"finishes": [], "points": [], "dnfs": [], "starts": 0, "wins": 0}
    )
    constructor_history: dict[str, dict[str, object]] = defaultdict(
        lambda: {"points": [], "points_share": [], "dnfs": [], "starts": 0, "entries": 0, "wins": 0}
    )
    lineage_history: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "points": [], "points_share": [], "starts": 0,
            "entries": 0, "wins": 0,
        }
    )
    circuit_history: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {"finishes": [], "starts": 0}
    )

    results_by_race = {key: group.copy() for key, group in results.groupby(race_keys, sort=False)}
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
        field_size = len(race_results)

        quali_rows: dict[str, dict[str, object]] = {}
        for qrecord in ([] if quali is None else quali.to_dict("records")):
            quali_rows[str(qrecord["driver_id"])] = qrecord
        q3_times = {
            driver: _seconds(record.get("q3"))
            for driver, record in quali_rows.items()
        }
        valid_q3 = [value for value in q3_times.values() if value is not None]
        best_q3 = min(valid_q3) if valid_q3 else None

        # Freeze every driver's feature row before updating current-race history.
        race_output: list[dict[str, object]] = []
        for result in race_results.to_dict("records"):
            driver_id = str(result["driver_id"])
            constructor_id = str(result["constructor_id"])
            dh = driver_history[driver_id]
            ch = constructor_history[constructor_id]
            lineage_id = _constructor_lineage_id(constructor_id)
            lh = lineage_history[lineage_id]
            cih = circuit_history[(driver_id, circuit_id)]
            qrow = quali_rows.get(driver_id)
            q3 = None if qrow is None else _seconds(qrow.get("q3"))

            teammate_q3 = [
                q3_times.get(str(other["driver_id"]))
                for other in race_results.to_dict("records")
                if str(other["constructor_id"]) == constructor_id
                and str(other["driver_id"]) != driver_id
                and q3_times.get(str(other["driver_id"])) is not None
            ]
            teammate_best = min(teammate_q3) if teammate_q3 else None
            finish_history = dh["finishes"]
            points_history = dh["points"]
            constructor_points = ch["points"]
            numeric_position = (
                float(result["position"])
                if pd.notna(result.get("position")) else None
            )
            row = {
                "season": key[0],
                "round": key[1],
                "race_name": race.get("race_name"),
                "circuit_id": circuit_id,
                "race_date": race.get("date"),
                "driver_id": driver_id,
                "constructor_id": constructor_id,
                "constructor_lineage_id": lineage_id,
                "field_size": field_size,
                "grid": result.get("grid"),
                "qualifying_position": None if qrow is None else qrow.get("position"),
                "q1_seconds": None if qrow is None else _seconds(qrow.get("q1")),
                "q2_seconds": None if qrow is None else _seconds(qrow.get("q2")),
                "q3_seconds": q3,
                "qualifying_gap_to_pole_seconds": (
                    None if q3 is None or best_q3 is None else q3 - best_q3
                ),
                "teammate_q3_gap_seconds": (
                    None if q3 is None or teammate_best is None else q3 - teammate_best
                ),
                "driver_prior_starts": int(dh["starts"]),
                "driver_prior_wins": int(dh["wins"]),
                "driver_prior_win_rate": (
                    float(dh["wins"]) / int(dh["starts"]) if dh["starts"] else 0.0
                ),
                "driver_finish_last_3": _mean(finish_history, 3),
                "driver_finish_last_5": _mean(finish_history, 5),
                "driver_points_last_3": sum(points_history[-3:]),
                "driver_points_last_5": sum(points_history[-5:]),
                "driver_weighted_form_5": (
                    float(sum(value * weight for value, weight in zip(
                        finish_history[-5:], range(1, min(len(finish_history), 5) + 1)
                    )) / sum(range(1, min(len(finish_history), 5) + 1)))
                    if finish_history else None
                ),
                "driver_prior_dnf_rate": (
                    float(sum(dh["dnfs"])) / int(dh["starts"]) if dh["starts"] else 0.0
                ),
                "constructor_prior_starts": int(ch["starts"]),
                "constructor_prior_wins": int(ch["wins"]),
                "constructor_prior_points": sum(constructor_points),
                "constructor_points_last_3": sum(constructor_points[-3:]),
                "constructor_points_last_5": sum(constructor_points[-5:]),
                "constructor_points_share_last_3": _mean(ch["points_share"], 3),
                "constructor_points_share_last_5": _mean(ch["points_share"], 5),
                "constructor_points_share_trend_3v3": (
                    _mean(ch["points_share"], 3)
                    - _mean(ch["points_share"][-6:-3], 3)
                    if len(ch["points_share"]) >= 6
                    else None
                ),
                "constructor_prior_win_rate": (
                    float(ch["wins"]) / int(ch["starts"]) if ch["starts"] else 0.0
                ),
                "constructor_prior_dnf_rate": (
                    float(sum(ch["dnfs"])) / int(ch["entries"])
                    if ch["entries"] else 0.0
                ),
                # Experimental lineage features; baseline source-ID features above
                # remain unchanged and continue to use exact Jolpica IDs.
                "lineage_prior_starts": int(lh["starts"]),
                "lineage_prior_wins": int(lh["wins"]),
                "lineage_prior_points": sum(lh["points"]),
                "lineage_prior_win_rate": (
                    float(lh["wins"]) / int(lh["starts"]) if lh["starts"] else 0.0
                ),
                "lineage_points_last_3": sum(lh["points"][-3:]),
                "lineage_points_last_5": sum(lh["points"][-5:]),
                "lineage_points_share_last_3": _mean(lh["points_share"], 3),
                "lineage_points_share_last_5": _mean(lh["points_share"], 5),
                "driver_circuit_prior_starts": int(cih["starts"]),
                "driver_circuit_finish_last_3": _mean(cih["finishes"], 3),
                "driver_circuit_finish_last_5": _mean(cih["finishes"], 5),
                "winner": int(numeric_position == 1.0) if numeric_position is not None else 0,
            }
            race_output.append(row)
        output.extend(race_output)

        constructor_race_points: dict[str, float] = defaultdict(float)
        constructor_race_wins: dict[str, int] = defaultdict(int)
        constructor_race_dnfs: dict[str, int] = defaultdict(int)
        for result in race_results.to_dict("records"):
            driver_id = str(result["driver_id"])
            constructor_id = str(result["constructor_id"])
            position = result.get("position")
            points = result.get("points")
            numeric_position = float(position) if pd.notna(position) else None
            numeric_points = float(points) if pd.notna(points) else 0.0
            dnf = _dnf(result.get("status"))

            dh = driver_history[driver_id]
            dh["starts"] += 1
            dh["points"].append(numeric_points)
            dh["dnfs"].append(dnf)
            if numeric_position is not None:
                dh["finishes"].append(numeric_position)
            if numeric_position == 1.0:
                dh["wins"] += 1

            cih = circuit_history[(driver_id, circuit_id)]
            cih["starts"] += 1
            if numeric_position is not None:
                cih["finishes"].append(numeric_position)

            constructor_race_points[constructor_id] += numeric_points
            constructor_race_dnfs[constructor_id] += dnf
            if numeric_position == 1.0:
                constructor_race_wins[constructor_id] += 1

        constructor_race_entries: dict[str, int] = defaultdict(int)
        for result in race_results.to_dict("records"):
            constructor_race_entries[str(result["constructor_id"])] += 1

        total_constructor_points = sum(constructor_race_points.values())
        lineage_race_points: dict[str, float] = defaultdict(float)
        lineage_race_wins: dict[str, int] = defaultdict(int)
        lineage_race_entries: dict[str, int] = defaultdict(int)
        for constructor_id, race_points in constructor_race_points.items():
            lineage_id = _constructor_lineage_id(constructor_id)
            lineage_race_points[lineage_id] += race_points
            lineage_race_wins[lineage_id] += constructor_race_wins[constructor_id]
            lineage_race_entries[lineage_id] += constructor_race_entries[constructor_id]

        for constructor_id, race_points in constructor_race_points.items():
            ch = constructor_history[constructor_id]
            ch["points_share"].append(
                race_points / total_constructor_points
                if total_constructor_points > 0
                else float("nan")
            )
            ch["starts"] += 1
            ch["entries"] += constructor_race_entries[constructor_id]
            ch["points"].append(race_points)
            ch["dnfs"].append(constructor_race_dnfs[constructor_id])
            ch["wins"] += constructor_race_wins[constructor_id]

        for lineage_id, race_points in lineage_race_points.items():
            lh = lineage_history[lineage_id]
            lh["points"].append(race_points)
            lh["points_share"].append(
                race_points / total_constructor_points
                if total_constructor_points > 0
                else float("nan")
            )
            lh["starts"] += 1
            lh["entries"] += lineage_race_entries[lineage_id]
            lh["wins"] += lineage_race_wins[lineage_id]

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
