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


def _nested_records(
    races: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for race in races:
        values = race.get(key, [])
        if isinstance(values, list):
            records.extend(item for item in values if isinstance(item, dict))
    return records


def inspect_season(year: int) -> None:
    directory = ROOT / "data" / "raw" / "jolpica" / str(year)
    races_payload = _load(directory / "races.json")
    results_payload = _load(directory / "results.json")
    qualifying_payload = _load(directory / "qualifying.json")
    metadata = _load(directory / "metadata.json")

    races = _races(races_payload)
    results = _nested_records(_races(results_payload), "Results")
    qualifying = _nested_records(_races(qualifying_payload), "QualifyingResults")

    race_rounds = [str(race.get("round")) for race in races]
    result_rounds = [
        str(race.get("round"))
        for race in _races(results_payload)
    ]
    qualifying_rounds = [
        str(race.get("round"))
        for race in _races(qualifying_payload)
    ]

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

    status_counts = Counter(
        str(result.get("status", "UNKNOWN"))
        for result in results
    )

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

    print("Result status distribution")
    print("-" * 60)
    for status, count in status_counts.most_common():
        print(f"{status:30} {count}")

    if not races or not results or not qualifying:
        raise RuntimeError("Season audit failed: one or more datasets are empty.")

    if missing_results or missing_qualifying:
        raise RuntimeError("Season audit failed: one or more race rounds are missing.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an ingested Jolpica season.")
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    inspect_season(args.year)


if __name__ == "__main__":
    main()
