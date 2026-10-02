from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _races(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return payload["MRData"]["RaceTable"]["Races"]


def _nested_records(races: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for race in races:
        values = race.get(key, [])
        if isinstance(values, list):
            records.extend(item for item in values if isinstance(item, dict))
    return records


def _round_counts(races: list[dict[str, Any]], key: str) -> dict[str, int]:
    return {
        str(race.get("round")): len(race.get(key, []))
        for race in races
    }


def inspect_season(year: int) -> None:
    directory = ROOT / "data" / "raw" / "jolpica" / str(year)
    races_payload = _load(directory / "races.json")
    results_payload = _load(directory / "results.json")
    qualifying_payload = _load(directory / "qualifying.json")
    metadata = _load(directory / "metadata.json")

    races = _races(races_payload)
    result_races = _races(results_payload)
    qualifying_races = _races(qualifying_payload)
    results = _nested_records(result_races, "Results")
    qualifying = _nested_records(qualifying_races, "QualifyingResults")

    race_rounds = [str(race.get("round")) for race in races]
    result_rounds = [str(race.get("round")) for race in result_races]
    qualifying_rounds = [str(race.get("round")) for race in qualifying_races]

    driver_ids = sorted({
        str(result["Driver"]["driverId"])
        for result in results
        if isinstance(result.get("Driver"), dict) and result["Driver"].get("driverId")
    })
    constructor_ids = sorted({
        str(result["Constructor"]["constructorId"])
        for result in results
        if isinstance(result.get("Constructor"), dict) and result["Constructor"].get("constructorId")
    })
    status_counts = Counter(str(result.get("status", "UNKNOWN")) for result in results)

    result_counts = _round_counts(result_races, "Results")
    qualifying_counts = _round_counts(qualifying_races, "QualifyingResults")

    result_keys = {
        (str(race.get("round")), str(item.get("number")))
        for race in result_races
        for item in race.get("Results", [])
    }
    qualifying_keys = {
        (str(race.get("round")), str(item.get("number")))
        for race in qualifying_races
        for item in race.get("QualifyingResults", [])
    }

    print(f"Jolpica season audit: {year}")
    print("=" * 60)
    print(f"Races:                {len(races)}")
    print(f"Race result records:  {len(results)}")
    print(f"Qualifying records:   {len(qualifying)}")
    print(f"Unique drivers:       {len(driver_ids)}")
    print(f"Unique constructors:  {len(constructor_ids)}")
    print(f"Race rounds:          {race_rounds[0]} -> {race_rounds[-1]}")
    print(f"Result rounds:        {result_rounds[0]} -> {result_rounds[-1]}")
    print(f"Qualifying rounds:    {qualifying_rounds[0]} -> {qualifying_rounds[-1]}")
    print(f"Result rows min/max:  {min(result_counts.values())}/{max(result_counts.values())}")
    print(f"Qualifying min/max:   {min(qualifying_counts.values())}/{max(qualifying_counts.values())}")
    print(f"Result/qualifying IDs: {len(result_keys & qualifying_keys)} shared")
    print(f"Metadata source:      {metadata.get('source')}")
    print(f"Retrieved at (UTC):   {metadata.get('retrieved_at_utc')}")
    print()

    missing_results = sorted(set(race_rounds) - set(result_rounds), key=int)
    missing_qualifying = sorted(set(race_rounds) - set(qualifying_rounds), key=int)
    extra_results = sorted(set(result_rounds) - set(race_rounds), key=int)
    extra_qualifying = sorted(set(qualifying_rounds) - set(race_rounds), key=int)

    print("Round coverage")
    print("-" * 60)
    print(f"Missing result rounds:       {missing_results or 'None'}")
    print(f"Missing qualifying rounds:   {missing_qualifying or 'None'}")
    print(f"Extra result rounds:          {extra_results or 'None'}")
    print(f"Extra qualifying rounds:      {extra_qualifying or 'None'}")
    print()

    print("Per-round record counts")
    print("-" * 60)
    for round_number in race_rounds:
        print(
            f"Round {round_number:>2}: "
            f"results={result_counts.get(round_number, 0):>2} "
            f"qualifying={qualifying_counts.get(round_number, 0):>2}"
        )
    print()

    print("Result status distribution")
    print("-" * 60)
    for status, count in status_counts.most_common():
        print(f"{status:30} {count}")

    if not races or not results or not qualifying:
        raise RuntimeError("Season audit failed: one or more datasets are empty.")
    if missing_results or missing_qualifying:
        raise RuntimeError("Season audit failed: one or more race rounds are missing.")
    if len(result_keys) != len(results) or len(qualifying_keys) != len(qualifying):
        raise RuntimeError("Season audit failed: duplicate round/driver-number keys detected.")
    if result_keys != qualifying_keys:
        print("\nWARNING: result and qualifying driver-number coverage differs.")
        print(f"Only in results: {sorted(result_keys - qualifying_keys)}")
        print(f"Only in qualifying: {sorted(qualifying_keys - result_keys)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an ingested Jolpica season.")
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    inspect_season(args.year)


if __name__ == "__main__":
    main()
