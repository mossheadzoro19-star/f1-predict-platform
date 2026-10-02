install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

ingest-2025:
	python -m f1_predict.ingestion.historical --year 2025 --source jolpica

ingest-openf1-2025:
	python -m f1_predict.ingestion.historical --year 2025 --source openf1
