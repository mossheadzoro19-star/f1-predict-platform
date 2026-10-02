# F1 Predict — Race Intelligence Platform

Production-style F1 race intelligence platform for research into dynamically changing race-winning probabilities.

## Milestone 1

- Repository and Codespaces foundation
- Jolpica-F1 historical season ingestion
- OpenF1 historical/live adapter foundation
- Data-source provenance documentation
- Raw/processed data separation
- Point-in-time ML design

## Historical ingestion

The current Jolpica ingestion downloads three core season-level datasets:

- race schedule and circuit metadata
- all race driver results
- all qualifying results

The Jolpica API documents a maximum page size of 100, so the adapter automatically paginates and combines all returned race records before writing the raw JSON files.

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

Raw provider data is intentionally ignored by git. The repository stores the ingestion code and provenance rules, not a large copy of provider responses.

## Quick start in Codespaces

```bash
python --version
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
python -m f1_predict.ingestion.historical --year 2025 --source jolpica
```

See `docs/DATA_PROVENANCE.md` and `docs/ROADMAP.md` before using data in a public/commercial deployment.


## Run the race-intelligence website

Install the runtime and start the MVP dashboard:

~~~bash
pip install -e ".[dev]"
make web
~~~

Open the forwarded port **8000** in Codespaces.

The website has three runtime behaviors:

- **LIVE** — when an active OpenF1 race session is available.
- **REPLAY** — when the provider exposes a historical race session or live data is unavailable.
- **Fallback** — the latest historical ML prediction remains available if the provider cannot be reached.

The current MVP combines the historical ML winner probability with live/replayed track-position evidence. This is deliberately an operational first version; the live-state model will be trained and evaluated separately before we replace this overlay with a learned race-state model.

OpenF1 currently documents historical access as free and real-time access as requiring the appropriate live-data access/subscription. Keep `OPENF1_API_TOKEN` on the backend and never expose it in frontend code.
