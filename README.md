# F1 Predict — Race Intelligence Platform

Production-style F1 race intelligence platform for research into dynamically changing race-winning probabilities.

## Current project policy

- **Historical structured data:** Jolpica-F1.
- **Historical race-state prototype:** Jolpica lap timing and pit-stop records, with reconstructed/approximate race-state features.
- **FastF1:** optional only. The upstream F1 timing archive began requiring authentication in October 2026; do not make it a required source or purchase a subscription for this project.
- **Live timing:** still to be selected and validated behind a replaceable adapter. Free access does not automatically grant commercial redistribution rights.
- **OpenF1:** retained as historical project context, but not required for historical ingestion or live access.

## Historical ingestion

The Jolpica season ingestion downloads three core datasets:

- race schedule and circuit metadata
- race driver results
- qualifying results

The adapter paginates the API responses and writes raw JSON files.

Run:

```bash
python -m f1_predict.ingestion.historical --year 2025 --source jolpica
```

Raw files are written under:

```text
data/raw/jolpica/2025/
├── races.json
├── results.json
├── qualifying.json
└── metadata.json
```

Raw provider data is intentionally ignored by git. The repository stores ingestion code and provenance rules, not a large copy of provider responses.

## Build historical race-state features

For the current no-paid-API path, use Jolpica race laps and pit stops:

```bash
python -m f1_predict.ingestion.jolpica_race_state \
  --start-year 2023 \
  --end-year 2025 \
  --output data/processed/features/jolpica_race_state_enriched.parquet
```

This dataset reconstructs approximate race-state features from published lap timing and results. It does **not** provide reliable live timing, tyre compound, or weather observations; unavailable features must remain missing, not be fabricated. Treat the reconstructed data as a separate experimental dataset and audit it before model comparison.

## Quick start in Codespaces

```bash
python --version
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
python -m f1_predict.ingestion.historical --year 2025 --source jolpica
```

See `docs/DATA_PROVENANCE.md`, `docs/ROADMAP.md`, and `docs/HANDOFF_REPORT.md` before using data in a public/commercial deployment.

## Run the current website prototype

```bash
pip install -e ".[dev]"
make web
```

Open the forwarded port **8000** in Codespaces.

The website/service is a prototype, not a validated live product. It must distinguish LIVE, REPLAY, FALLBACK, STALE, and DISCONNECTED states. Historical or fallback predictions must never be presented as live observations. The real-time provider and end-to-end live inference path are still pending validation.
