from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest historical F1 data.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--source",
        choices=("jolpica", "openf1", "both"),
        default="both",
    )
    args = parser.parse_args()

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
