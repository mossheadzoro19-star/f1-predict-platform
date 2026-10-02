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


def ingest_jolpica(year: int) -> Path:
    payload = JolpicaClient().get_season_results(year)
    return _write_json("jolpica", year, "season_results.json", payload)


def ingest_openf1(year: int) -> Path:
    payload = OpenF1Client().get_sessions(year)
    return _write_json("openf1", year, "sessions.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest historical F1 data.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--source",
        choices=("jolpica", "openf1", "both"),
        default="both",
    )
    args = parser.parse_args()

    retrieved_at = datetime.now(timezone.utc).isoformat()

    if args.source in ("jolpica", "both"):
        path = ingest_jolpica(args.year)
        print(f"Jolpica data written to {path}")

    if args.source in ("openf1", "both"):
        path = ingest_openf1(args.year)
        print(f"OpenF1 data written to {path}")

    print(f"retrieved_at_utc={retrieved_at}")


if __name__ == "__main__":
    main()
