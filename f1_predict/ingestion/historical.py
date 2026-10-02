from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from f1_predict.data_sources.jolpica import JolpicaClient
from f1_predict.data_sources.openf1 import OpenF1Client

ROOT = Path(__file__).resolve().parents[2]


def _write_json(source: str, year: int, name: str, payload: Any) -> Path:
    out_dir = ROOT / "data" / "raw" / source / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_metadata(source: str, year: int, endpoints: list[str]) -> Path:
    metadata = {
        "source": source,
        "season": year,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoints": endpoints,
        "raw_data_policy": "Raw provider responses are retained locally and excluded from git.",
    }
    return _write_json(source, year, "metadata.json", metadata)


def ingest_jolpica(year: int) -> list[Path]:
    """Download the core season-level historical dataset from Jolpica."""
    client = JolpicaClient()
    files = [
        ("races.json", client.get_season_schedule(year)),
        ("results.json", client.get_season_results(year)),
        ("qualifying.json", client.get_season_qualifying(year)),
    ]
    paths = [_write_json("jolpica", year, name, payload) for name, payload in files]
    paths.append(
        _write_metadata(
            "jolpica",
            year,
            [
                f"/ergast/f1/{year}/races/",
                f"/ergast/f1/{year}/results/",
                f"/ergast/f1/{year}/qualifying/",
            ],
        )
    )
    return paths


def cleanup_legacy_jolpica_files(year: int) -> None:
    """Remove files from the retired pre-paginated ingestion format."""
    legacy_path = ROOT / "data" / "raw" / "jolpica" / str(year) / "season_results.json"
    legacy_path.unlink(missing_ok=True)


def ingest_openf1(year: int) -> list[Path]:
    payload = OpenF1Client().get_sessions(year)
    path = _write_json("openf1", year, "sessions.json", payload)
    metadata = _write_metadata("openf1", year, ["/v1/sessions"])
    return [path, metadata]


def ingest_jolpica_range(start_year: int, end_year: int, pause_seconds: float = 2.0) -> dict[int, list[Path]]:
    """Ingest multiple Jolpica seasons with a pause between seasons."""
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    if pause_seconds < 0:
        raise ValueError("pause_seconds must be >= 0")

    completed: dict[int, list[Path]] = {}
    for year in range(start_year, end_year + 1):
        print(f"\n=== Jolpica {year} ===")
        completed[year] = ingest_jolpica(year)
        cleanup_legacy_jolpica_files(year)
        if year < end_year and pause_seconds:
            time.sleep(pause_seconds)
    return completed

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest historical F1 data.")
    parser.add_argument("--year", type=int)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    parser.add_argument(
        "--source",
        choices=("jolpica", "openf1", "both"),
        default="jolpica",
    )
    args = parser.parse_args()

    range_mode = args.start_year is not None or args.end_year is not None
    if range_mode and (args.start_year is None or args.end_year is None):
        parser.error("--start-year and --end-year must be provided together")
    if not range_mode and args.year is None:
        parser.error("--year is required unless --start-year and --end-year are provided")
    if range_mode and args.source != "jolpica":
        parser.error("multi-season mode currently supports --source jolpica only")

    if range_mode:
        results = ingest_jolpica_range(args.start_year, args.end_year, args.pause_seconds)
        print("\nMulti-season ingestion complete:")
        for year, paths in results.items():
            print(f"  {year}: {len(paths)} files")
        return

    paths: list[Path] = []
    if args.source in ("jolpica", "both"):
        paths.extend(ingest_jolpica(args.year))
        cleanup_legacy_jolpica_files(args.year)
    if args.source in ("openf1", "both"):
        paths.extend(ingest_openf1(args.year))

    print("Ingestion complete:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
