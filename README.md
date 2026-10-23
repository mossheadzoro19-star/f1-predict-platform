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

**Preferred recovery path when Jolpica returns persistent HTTP 429:** use the static TracingInsights/RaceData archive. It is a public release ZIP, needs no API key, and is cached locally after the first download:

```bash
python -m f1_predict.ingestion.static_race_state \
  --start-year 2023 \
  --end-year 2025 \
  --output data/processed/features/jolpica_race_state_enriched.parquet
```

The archive provides historical race metadata, results, lap times, lap-end positions and pit stops. The source repository identifies its dataset license as CC0 1.0. This path does not call Jolpica endpoints.

The original API-based path remains available when the provider is healthy:

```bash
python -m f1_predict.ingestion.jolpica_race_state \
  --start-year 2023 \
  --end-year 2025 \
  --output data/processed/features/jolpica_race_state_enriched.parquet
```

The static archive's race-state timestamps are approximate race-day anchors plus cumulative lap times, not official timing-feed timestamps. Weather and tyre-compound features remain missing unless explicitly present in the source. Audit the resulting dataset before model comparison, and do not present it as live timing.

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
