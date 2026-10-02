# F1 Predict — Race Intelligence Platform

Production-style F1 race intelligence platform for research into dynamically changing race-winning probabilities.

## Milestone 1

- Repository and Codespaces foundation
- Data-source adapters for Jolpica-F1 and OpenF1
- Historical ingestion entry point
- Data provenance documentation
- Test scaffold
- Raw/processed data separation

## Data architecture

Jolpica-F1 is used for structured historical championship/race records.
FastF1 will be integrated for richer historical session/lap/weather processing.
OpenF1 is the planned live-data provider, with historical replay as a fallback.

The ML pipeline is designed around point-in-time correctness: a prediction may only use information that would have been available at that prediction timestamp.

## Quick start in Codespaces

```bash
python --version
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m f1_predict.ingestion.historical --year 2025 --source jolpica
```

See `docs/DATA_PROVENANCE.md` and `docs/ROADMAP.md` before using data in a public/commercial deployment.
