from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "data" / "processed" / "features" / "race_driver_features.parquet"


def load_features(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load the point-in-time driver-race feature dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")
    return pd.read_parquet(path)


def summarize_dataset(frame: pd.DataFrame) -> dict:
    """Return high-level dataset dimensions and temporal coverage."""
    races = frame[["season", "round"]].drop_duplicates()
    return {
        "rows": int(len(frame)),
        "races": int(len(races)),
        "seasons": sorted(frame["season"].dropna().astype(int).unique().tolist()),
        "drivers": int(frame["driver_id"].nunique()),
        "constructors": int(frame["constructor_id"].nunique()),
        "circuits": int(frame["circuit_id"].nunique()),
        "start_season": int(frame["season"].min()),
        "end_season": int(frame["season"].max()),
    }


def summarize_target(frame: pd.DataFrame) -> dict:
    """Summarize the winner target at driver-row level and race level."""
    winners = frame["winner"].value_counts().sort_index()
    race_count = frame[["season", "round"]].drop_duplicates().shape[0]
    return {
        "winner_rows": int(winners.get(1, 0)),
        "non_winner_rows": int(winners.get(0, 0)),
        "winner_row_rate": float(frame["winner"].mean()),
        "races": int(race_count),
        "one_winner_per_race": bool(
            frame.groupby(["season", "round"])["winner"].sum().eq(1).all()
        ),
    }


def summarize_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    """Return columns ordered by missing-value count."""
    result = pd.DataFrame(
        {
            "column": frame.columns,
            "missing_count": frame.isna().sum().astype(int).values,
        }
    )
    result["missing_rate"] = result["missing_count"] / len(frame)
    return result.sort_values(
        ["missing_count", "column"], ascending=[False, True]
    ).reset_index(drop=True)


def summarize_seasons(frame: pd.DataFrame) -> pd.DataFrame:
    """Return season-level row, race, driver and winner counts."""
    rows = []
    for season, group in frame.groupby("season", sort=True):
        rows.append(
            {
                "season": int(season),
                "rows": int(len(group)),
                "races": int(group["round"].nunique()),
                "drivers": int(group["driver_id"].nunique()),
                "constructors": int(group["constructor_id"].nunique()),
                "winners": int(group["winner"].sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_qualifying_and_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure winner rates by qualifying and starting-grid position."""
    result = frame[["qualifying_position", "grid", "winner"]].copy()
    rows: list[dict] = []

    for feature in ("qualifying_position", "grid"):
        grouped = result.dropna(subset=[feature]).groupby(feature)["winner"]
        summary = grouped.agg(["count", "sum", "mean"]).reset_index()
        for row in summary.itertuples(index=False):
            rows.append(
                {
                    "feature": feature,
                    "position": int(row[0]),
                    "starts": int(row[1]),
                    "wins": int(row[2]),
                    "win_rate": float(row[3]),
                }
            )

    return pd.DataFrame(rows)


def summarize_numeric_target_association(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson association between numeric features and the winner target.

    This is descriptive EDA only; it is not used as a model-selection criterion.
    """
    numeric = frame.select_dtypes(include="number")
    target = numeric["winner"]
    rows = []

    for column in numeric.columns:
        if column == "winner":
            continue
        pair = numeric[[column, "winner"]].dropna()
        if len(pair) < 2 or pair[column].nunique() < 2:
            continue
        rows.append(
            {
                "feature": column,
                "correlation_with_winner": float(pair[column].corr(pair["winner"])),
                "non_null_rows": int(len(pair)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "correlation_with_winner", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)


def build_eda_report(frame: pd.DataFrame) -> dict:
    """Build a JSON-serializable descriptive EDA report."""
    missingness = summarize_missingness(frame)
    seasons = summarize_seasons(frame)
    qualifying_grid = summarize_qualifying_and_grid(frame)
    associations = summarize_numeric_target_association(frame)

    return {
        "dataset": summarize_dataset(frame),
        "target": summarize_target(frame),
        "missingness": missingness.to_dict(orient="records"),
        "seasons": seasons.to_dict(orient="records"),
        "qualifying_and_grid": qualifying_grid.to_dict(orient="records"),
        "numeric_target_association": associations.to_dict(orient="records"),
    }


def write_report(report: dict, path: Path) -> Path:
    """Write the EDA report as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run descriptive EDA on F1 features.")
    parser.add_argument("--input", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "features" / "eda_report.json",
    )
    args = parser.parse_args()

    frame = load_features(args.input)
    report = build_eda_report(frame)
    write_report(report, args.output)

    print("=" * 60)
    print("F1 Predict — Exploratory Data Analysis")
    print("=" * 60)
    print(json.dumps(report["dataset"], indent=2))
    print()
    print("TARGET")
    print(json.dumps(report["target"], indent=2))
    print()
    print("TOP MISSINGNESS")
    print(pd.DataFrame(report["missingness"]).head(10).to_string(index=False))
    print()
    print("SEASON COVERAGE")
    print(pd.DataFrame(report["seasons"]).to_string(index=False))
    print()
    print("TOP NUMERIC ASSOCIATIONS WITH WINNER")
    print(pd.DataFrame(report["numeric_target_association"]).head(15).to_string(index=False))
    print()
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
