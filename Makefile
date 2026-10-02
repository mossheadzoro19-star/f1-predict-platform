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

web:
	uvicorn f1_predict.web:app --host 0.0.0.0 --port 8000 --reload
