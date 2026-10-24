from __future__ import annotations

"""Safely recompute historical multi-lap dynamics from an existing race-state Parquet."""

import argparse
from pathlib import Path

import pandas as pd


GROUP_COLUMNS = ["session_key", "driver_number"]
KEY_COLUMNS = ["session_key", "driver_number", "lap_number"]
DYNAMIC_COLUMNS = {
    "position_change": "position_position",
    "gap_change": "interval_gap_to_leader",
    "interval_change": "interval_interval",
}
WINDOWS = (3, 5)


def recompute_multi_lap_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute 3-/5-lap deltas without crossing race or driver boundaries."""
    required = set(KEY_COLUMNS) | set(DYNAMIC_COLUMNS.values())
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing source columns needed for recomputation: {missing}")

    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"Duplicate historical observations for keys: {KEY_COLUMNS}")

    result = frame.copy()
    result["driver_number"] = result["driver_number"].astype(str)
    result = result.sort_values(
        ["session_key", "driver_number", "lap_number"],
        kind="mergesort",
    ).copy()
    grouped = result.groupby(GROUP_COLUMNS, sort=False, dropna=False)

    for window in WINDOWS:
        for prefix, source in DYNAMIC_COLUMNS.items():
            result[f"{prefix}_{window}_laps"] = (
                result[source] - grouped[source].shift(window)
            )

    return result


def repair_file(input_path: Path, output_path: Path, overwrite: bool = False) -> pd.DataFrame:
    """Repair to a separate output file; never overwrite the source by accident."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input Parquet file not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Choose a new path or pass --overwrite."
        )

    frame = pd.read_parquet(input_path)
    repaired = recompute_multi_lap_features(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_parquet(output_path, index=False)
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute 3- and 5-lap race-state features from an existing Parquet file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/features/jolpica_race_state_enriched.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/features/jolpica_race_state_repaired.parquet"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing the output file (the input file is never overwritten).",
    )
    args = parser.parse_args()

    repaired = repair_file(args.input, args.output, overwrite=args.overwrite)
    print(f"Rows: {len(repaired)}")
    print(f"Races: {repaired['session_key'].nunique()}")
    print("Recomputed feature missingness:")
    for column in (
        "position_change_3_laps",
        "position_change_5_laps",
        "gap_change_3_laps",
        "gap_change_5_laps",
        "interval_change_3_laps",
        "interval_change_5_laps",
    ):
        print(f"  {column}: {repaired[column].isna().mean():.1%}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
