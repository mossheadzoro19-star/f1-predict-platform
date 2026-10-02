from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_COLUMNS = {
    "races": {
        "season", "round", "race_name", "circuit_id", "circuit_name",
        "country", "locality", "latitude", "longitude", "date", "time",
    },
    "drivers": {
        "driver_id", "given_name", "family_name", "code",
        "permanent_number", "nationality",
    },
    "constructors": {"constructor_id", "name", "nationality"},
    "qualifying": {
        "season", "round", "driver_id", "constructor_id",
        "number", "position", "q1", "q2", "q3",
    },
    "race_results": {
        "season", "round", "driver_id", "constructor_id",
        "number", "position", "position_text", "points", "grid", "laps",
        "status", "race_time", "race_time_millis", "fastest_lap_rank",
        "fastest_lap", "fastest_lap_time", "fastest_lap_time_millis",
        "fastest_lap_speed", "fastest_lap_speed_units",
    },
}


def _load(year: int, name: str) -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "jolpica" / str(year) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing normalized file: {path}")
    return pd.read_parquet(path)


def _assert_columns(name: str, df: pd.DataFrame) -> None:
    missing = EXPECTED_COLUMNS[name] - set(df.columns)
    if missing:
        raise AssertionError(f"{name}: missing columns: {sorted(missing)}")


def _assert_no_duplicates(name: str, df: pd.DataFrame, keys: list[str]) -> None:
    duplicates = int(df.duplicated(keys).sum())
    if duplicates:
        raise AssertionError(f"{name}: {duplicates} duplicate rows for key {keys}")


def audit_normalized_season(year: int) -> None:
    races = _load(year, "races")
    drivers = _load(year, "drivers")
    constructors = _load(year, "constructors")
    qualifying = _load(year, "qualifying")
    results = _load(year, "race_results")

    tables = {
        "races": races,
        "drivers": drivers,
        "constructors": constructors,
        "qualifying": qualifying,
        "race_results": results,
    }

    for name, df in tables.items():
        _assert_columns(name, df)
        if df.empty:
            raise AssertionError(f"{name}: table is empty")

    _assert_no_duplicates("races", races, ["season", "round"])
    _assert_no_duplicates("qualifying", qualifying, ["season", "round", "driver_id"])
    _assert_no_duplicates("race_results", results, ["season", "round", "driver_id"])

    race_rounds = set(races["round"].astype(int))
    result_rounds = set(results["round"].astype(int))
    qualifying_rounds = set(qualifying["round"].astype(int))

    if result_rounds != race_rounds:
        raise AssertionError(
            f"race_results round mismatch: missing={sorted(race_rounds - result_rounds)}, "
            f"extra={sorted(result_rounds - race_rounds)}"
        )
    if qualifying_rounds != race_rounds:
        raise AssertionError(
            f"qualifying round mismatch: missing={sorted(race_rounds - qualifying_rounds)}, "
            f"extra={sorted(qualifying_rounds - race_rounds)}"
        )

    driver_ids = set(drivers["driver_id"].dropna().astype(str))
    constructor_ids = set(constructors["constructor_id"].dropna().astype(str))

    result_driver_ids = set(results["driver_id"].dropna().astype(str))
    qualifying_driver_ids = set(qualifying["driver_id"].dropna().astype(str))
    result_constructor_ids = set(results["constructor_id"].dropna().astype(str))
    qualifying_constructor_ids = set(qualifying["constructor_id"].dropna().astype(str))

    unknown_result_drivers = result_driver_ids - driver_ids
    unknown_qualifying_drivers = qualifying_driver_ids - driver_ids
    unknown_result_constructors = result_constructor_ids - constructor_ids
    unknown_qualifying_constructors = qualifying_constructor_ids - constructor_ids

    if unknown_result_drivers or unknown_qualifying_drivers:
        raise AssertionError(
            "Unknown driver references: "
            f"results={sorted(unknown_result_drivers)}, "
            f"qualifying={sorted(unknown_qualifying_drivers)}"
        )
    if unknown_result_constructors or unknown_qualifying_constructors:
        raise AssertionError(
            "Unknown constructor references: "
            f"results={sorted(unknown_result_constructors)}, "
            f"qualifying={sorted(unknown_qualifying_constructors)}"
        )

    expected_rounds = set(range(1, len(races) + 1))
    if race_rounds != expected_rounds:
        raise AssertionError(
            f"Unexpected race rounds: expected={sorted(expected_rounds)}, "
            f"actual={sorted(race_rounds)}"
        )

    print(f"Normalized Jolpica audit: {year}")
    print("=" * 60)
    print(f"Races:                {len(races)}")
    print(f"Race result records:  {len(results)}")
    print(f"Qualifying records:   {len(qualifying)}")
    print(f"Drivers:              {len(drivers)}")
    print(f"Constructors:         {len(constructors)}")
    print(f"Race rounds:          {min(race_rounds)} -> {max(race_rounds)}")
    print(f"Result rounds:        {min(result_rounds)} -> {max(result_rounds)}")
    print(f"Qualifying rounds:    {min(qualifying_rounds)} -> {max(qualifying_rounds)}")
    print()
    print("Integrity checks")
    print("-" * 60)
    print("Required columns:     PASS")
    print("Non-empty tables:     PASS")
    print("Unique race keys:     PASS")
    print("Unique result keys:   PASS")
    print("Unique qualifying:    PASS")
    print("Round coverage:       PASS")
    print("Driver references:    PASS")
    print("Constructor refs:     PASS")
    print()
    print("Normalized data audit: PASS")


def audit_range(start_year: int, end_year: int) -> None:
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")

    passed: list[int] = []
    failed: list[tuple[int, str]] = []

    for year in range(start_year, end_year + 1):
        try:
            audit_normalized_season(year)
            passed.append(year)
            print(f"\nNORMALIZED AUDIT {year}: PASS")
        except Exception as exc:
            failed.append((year, str(exc)))
            print(f"\nNORMALIZED AUDIT {year}: FAIL — {exc}")

    print("\n" + "=" * 60)
    print("Historical normalized-data audit")
    print("=" * 60)
    print(f"Seasons checked: {len(passed) + len(failed)}")
    print(f"Seasons passed:  {len(passed)}")
    print(f"Seasons failed:  {len(failed)}")
    if failed:
        print("\nFailures")
        print("-" * 60)
        for year, error in failed:
            print(f"{year}: {error}")
        raise RuntimeError("Historical normalized-data audit failed.")
    print("\nNORMALIZED DATASET AUDIT: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit normalized Jolpica Parquet tables.")
    parser.add_argument("--year", type=int)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    args = parser.parse_args()

    range_mode = args.start_year is not None or args.end_year is not None
    if range_mode and (args.start_year is None or args.end_year is None):
        parser.error("--start-year and --end-year must be provided together")
    if not range_mode and args.year is None:
        parser.error("--year is required unless --start-year and --end-year are provided")

    if range_mode:
        audit_range(args.start_year, args.end_year)
    else:
        audit_normalized_season(args.year)


if __name__ == "__main__":
    main()
