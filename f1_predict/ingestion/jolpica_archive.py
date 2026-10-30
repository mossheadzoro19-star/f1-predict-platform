from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from f1_predict.data_sources.jolpica import JolpicaClient


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "data" / "jolpica"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def get_races(payload: dict[str, Any]) -> list[dict[str, Any]]:
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    return races if isinstance(races, list) else []


def count_records(endpoint: str, payload: dict[str, Any]) -> int:
    races = get_races(payload)

    if endpoint == "laps":
        return sum(
            len(lap.get("Timings", []))
            for race in races
            for lap in race.get("Laps", [])
            if isinstance(lap, dict)
            and isinstance(lap.get("Timings", []), list)
        )

    if endpoint == "pitstops":
        return sum(
            len(race.get("PitStops", []))
            for race in races
            if isinstance(race.get("PitStops", []), list)
        )

    raise ValueError(f"Unsupported endpoint: {endpoint}")


def validate_cached_payload(
    path: Path,
    endpoint: str,
    year: int,
    round_number: int,
) -> dict[str, Any] | None:
    if endpoint not in {"laps", "pitstops"}:
        raise ValueError(f"Unsupported endpoint: {endpoint}")

    payload = read_json(path)
    if payload is None:
        return None

    races = get_races(payload)
    if len(races) != 1:
        return None

    race = races[0]
    if str(race.get("season")) != str(year):
        return None
    if str(race.get("round")) != str(round_number):
        return None

    collection = "Laps" if endpoint == "laps" else "PitStops"
    if not isinstance(race.get(collection), list):
        return None

    return payload


class JolpicaArchive:
    """Download and cache raw Jolpica lap and pit-stop responses."""

    def __init__(
        self,
        root: Path = DEFAULT_ROOT,
        client: JolpicaClient | None = None,
        pause_seconds: float = 1.0,
    ) -> None:
        if pause_seconds < 0:
            raise ValueError("pause_seconds must be >= 0")

        self.root = Path(root)
        self.client = client or JolpicaClient(
            request_delay_seconds=1.0,
            max_retries=6,
        )
        self.pause_seconds = pause_seconds
        self.manifest_path = (
            self.root / "manifests" / "ingestion_manifest.json"
        )

        self.manifest = read_json(self.manifest_path)
        if self.manifest is None:
            self.manifest = {
                "schema_version": 1,
                "source": "jolpica",
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
                "races": {},
            }

        if not isinstance(self.manifest.get("races"), dict):
            raise ValueError("Manifest 'races' field must be an object")

    def race_directory(self, year: int, round_number: int) -> Path:
        return (
            self.root
            / "raw"
            / str(year)
            / f"round_{round_number:02d}"
        )

    def save_manifest(self) -> None:
        self.manifest["updated_at_utc"] = utc_now()
        atomic_write_json(self.manifest_path, self.manifest)

    def get_race_entry(
        self,
        year: int,
        round_number: int,
    ) -> dict[str, Any]:
        key = f"{year}-{round_number:02d}"
        return self.manifest["races"].setdefault(
            key,
            {
                "season": year,
                "round": round_number,
                "endpoints": {},
            },
        )

    def fetch_endpoint(
        self,
        year: int,
        round_number: int,
        endpoint: str,
    ) -> dict[str, Any]:
        if endpoint not in {"laps", "pitstops"}:
            raise ValueError(f"Unsupported endpoint: {endpoint}")

        output_path = (
            self.race_directory(year, round_number) / f"{endpoint}.json"
        )
        entry = self.get_race_entry(year, round_number)
        endpoint_state = entry["endpoints"]

        cached = validate_cached_payload(
            output_path, endpoint, year, round_number
        )
        if cached is not None:
            count = count_records(endpoint, cached)
            endpoint_state[endpoint] = {
                "status": "complete",
                "records": count,
                "file": str(output_path.relative_to(self.root)),
                "verified_at_utc": utc_now(),
                "cache_hit": True,
            }
            self.save_manifest()
            print(
                f"CACHE {year} R{round_number:02d} {endpoint}: "
                f"{count} records",
                flush=True,
            )
            return cached

        endpoint_state[endpoint] = {
            "status": "in_progress",
            "started_at_utc": utc_now(),
        }
        self.save_manifest()

        try:
            if endpoint == "laps":
                payload = self.client.get_race_laps(year, round_number)
            else:
                payload = self.client.get_race_pitstops(year, round_number)

            races = get_races(payload)
            if len(races) != 1:
                raise ValueError(
                    f"Expected exactly one race, received {len(races)}"
                )

            race = races[0]
            if str(race.get("season")) != str(year):
                raise ValueError("Response season does not match request")
            if str(race.get("round")) != str(round_number):
                raise ValueError("Response round does not match request")

            collection = "Laps" if endpoint == "laps" else "PitStops"
            if not isinstance(race.get(collection), list):
                raise ValueError(
                    f"Response does not contain a valid {collection} list"
                )

            count = count_records(endpoint, payload)
            atomic_write_json(output_path, payload)

            endpoint_state[endpoint] = {
                "status": "complete",
                "records": count,
                "file": str(output_path.relative_to(self.root)),
                "verified_at_utc": utc_now(),
                "cache_hit": False,
            }
            self.save_manifest()

            print(
                f"OK {year} R{round_number:02d} {endpoint}: "
                f"{count} records",
                flush=True,
            )
            return payload

        except Exception as exc:
            endpoint_state[endpoint] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "failed_at_utc": utc_now(),
            }
            self.save_manifest()
            raise

    def ingest_race(self, year: int, round_number: int) -> None:
        if year < 1950:
            raise ValueError("Invalid F1 season")
        if round_number < 1:
            raise ValueError("Round number must be positive")

        for index, endpoint in enumerate(("laps", "pitstops")):
            self.fetch_endpoint(year, round_number, endpoint)
            if index == 0 and self.pause_seconds:
                time.sleep(self.pause_seconds)

    def write_audit(self, start_year: int, end_year: int) -> Path:
        if start_year > end_year:
            raise ValueError("start_year must be <= end_year")

        rows = []
        for key, race in sorted(self.manifest["races"].items()):
            year = int(race["season"])
            if not start_year <= year <= end_year:
                continue

            endpoints = race.get("endpoints", {})
            laps = endpoints.get("laps", {})
            pits = endpoints.get("pitstops", {})
            rows.append(
                {
                    "race": key,
                    "season": year,
                    "round": race["round"],
                    "laps_status": laps.get("status", "missing"),
                    "laps_records": laps.get("records", 0),
                    "pitstops_status": pits.get("status", "missing"),
                    "pitstops_records": pits.get("records", 0),
                }
            )

        audit = {
            "source": "jolpica",
            "start_year": start_year,
            "end_year": end_year,
            "generated_at_utc": utc_now(),
            "race_count_in_manifest": len(rows),
            "complete_races": sum(
                row["laps_status"] == "complete"
                and row["pitstops_status"] == "complete"
                for row in rows
            ),
            "incomplete_races": sum(
                row["laps_status"] != "complete"
                or row["pitstops_status"] != "complete"
                for row in rows
            ),
            "total_lap_timing_records": sum(
                row["laps_records"] for row in rows
            ),
            "total_pitstop_records": sum(
                row["pitstops_records"] for row in rows
            ),
            "races": rows,
        }

        output_path = (
            self.root
            / "audits"
            / f"ingestion_{start_year}_{end_year}.json"
        )
        atomic_write_json(output_path, audit)
        return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download raw Jolpica lap-time and pit-stop data."
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    args = parser.parse_args()

    archive = JolpicaArchive(
        root=args.root,
        pause_seconds=args.pause_seconds,
    )
    archive.ingest_race(args.year, args.round)
    audit_path = archive.write_audit(args.year, args.year)
    print(f"\nAudit written to: {audit_path}")


if __name__ == "__main__":
    main()
