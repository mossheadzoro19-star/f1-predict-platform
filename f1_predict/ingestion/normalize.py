from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _load_json(year: int, name: str) -> dict[str, Any]:
    path = ROOT / "data" / "raw" / "jolpica" / str(year) / name
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _table(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    rows: list[dict[str, Any]] = []
    for race in races:
        for item in race.get(key, []):
            row = dict(item)
            row["_race"] = race
            rows.append(row)
    return rows


def _race_fields(race: dict[str, Any]) -> dict[str, Any]:
    circuit = race.get("Circuit") or {}
    location = circuit.get("Location") or {}
    return {
        "season": int(race["season"]),
        "round": int(race["round"]),
        "race_name": race.get("raceName"),
        "circuit_id": circuit.get("circuitId"),
        "circuit_name": circuit.get("circuitName"),
        "country": location.get("country"),
        "locality": location.get("locality"),
        "latitude": location.get("lat"),
        "longitude": location.get("long"),
        "date": race.get("date"),
        "time": race.get("time"),
    }


def normalize_races(payload: dict[str, Any]) -> pd.DataFrame:
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    return pd.DataFrame([_race_fields(race) for race in races])


def _driver_fields(driver: dict[str, Any]) -> dict[str, Any]:
    return {
        "driver_id": driver.get("driverId"),
        "given_name": driver.get("givenName"),
        "family_name": driver.get("familyName"),
        "code": driver.get("code"),
        "permanent_number": driver.get("permanentNumber"),
        "nationality": driver.get("nationality"),
    }


def _constructor_fields(constructor: dict[str, Any]) -> dict[str, Any]:
    return {
        "constructor_id": constructor.get("constructorId"),
        "name": constructor.get("name"),
        "nationality": constructor.get("nationality"),
    }


def normalize_entities(results_payload: dict[str, Any], qualifying_payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    drivers: dict[str, dict[str, Any]] = {}
    constructors: dict[str, dict[str, Any]] = {}

    for payload, key in ((results_payload, "Results"), (qualifying_payload, "QualifyingResults")):
        for row in _table(payload, key):
            driver = row.get("Driver") or {}
            constructor = row.get("Constructor") or {}
            driver_id = driver.get("driverId")
            constructor_id = constructor.get("constructorId")
            if driver_id:
                drivers[driver_id] = _driver_fields(driver)
            if constructor_id:
                constructors[constructor_id] = _constructor_fields(constructor)

    return (
        pd.DataFrame(sorted(drivers.values(), key=lambda x: x["driver_id"])),
        pd.DataFrame(sorted(constructors.values(), key=lambda x: x["constructor_id"])),
    )


def _base_result_fields(row: dict[str, Any]) -> dict[str, Any]:
    race = row["_race"]
    driver = row.get("Driver") or {}
    constructor = row.get("Constructor") or {}
    return {
        "season": int(race["season"]),
        "round": int(race["round"]),
        "driver_id": driver.get("driverId"),
        "constructor_id": constructor.get("constructorId"),
    }


def _nullable_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_results(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in _table(payload, "Results"):
        result = _base_result_fields(row)
        fastest = row.get("FastestLap") or {}
        speed = fastest.get("AverageSpeed") or {}
        time = row.get("Time") or {}
        fastest_time = fastest.get("Time") or {}
        result.update(
            {
                "number": _nullable_int(row.get("number")),
                "position": _nullable_int(row.get("position")),
                "position_text": row.get("positionText"),
                "points": float(row["points"]) if row.get("points") not in (None, "") else None,
                "grid": _nullable_int(row.get("grid")),
                "laps": _nullable_int(row.get("laps")),
                "status": row.get("status"),
                "race_time": time.get("time"),
                "race_time_millis": _nullable_int(time.get("millis")),
                "fastest_lap_rank": _nullable_int(fastest.get("rank")),
                "fastest_lap": _nullable_int(fastest.get("lap")),
                "fastest_lap_time": fastest_time.get("time"),
                "fastest_lap_time_millis": _nullable_int(fastest_time.get("millis")),
                "fastest_lap_speed": float(speed["speed"]) if speed.get("speed") not in (None, "") else None,
                "fastest_lap_speed_units": speed.get("units"),
            }
        )
        rows.append(result)
    return pd.DataFrame(rows)


def normalize_qualifying(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in _table(payload, "QualifyingResults"):
        base = _base_result_fields(row)
        base.update(
            {
                "number": _nullable_int(row.get("number")),
                "position": _nullable_int(row.get("position")),
                "q1": row.get("Q1"),
                "q2": row.get("Q2"),
                "q3": row.get("Q3"),
            }
        )
        rows.append(base)
    return pd.DataFrame(rows)


def _write(df: pd.DataFrame, year: int, name: str) -> Path:
    out_dir = ROOT / "data" / "processed" / "jolpica" / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def normalize_season(year: int) -> list[Path]:
    races_payload = _load_json(year, "races.json")
    results_payload = _load_json(year, "results.json")
    qualifying_payload = _load_json(year, "qualifying.json")

    races = normalize_races(races_payload)
    results = normalize_results(results_payload)
    qualifying = normalize_qualifying(qualifying_payload)
    drivers, constructors = normalize_entities(results_payload, qualifying_payload)

    return [
        _write(races, year, "races"),
        _write(drivers, year, "drivers"),
        _write(constructors, year, "constructors"),
        _write(qualifying, year, "qualifying"),
        _write(results, year, "race_results"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Jolpica raw season data into Parquet tables.")
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    print("Normalization complete:")
    for path in normalize_season(args.year):
        print(f"  {path}")


if __name__ == "__main__":
    main()
