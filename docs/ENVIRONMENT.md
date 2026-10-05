# Environment

## Runtime
- Python >=3.12
- Node 22 where frontend tooling is used.
- GitHub Codespaces is the primary development environment.

## Python dependencies
Declared in `pyproject.toml`: pandas, pyarrow, scikit-learn, XGBoost, FastAPI, Uvicorn, httpx and development tools.

## Variables
Private provider credentials, when required, belong in environment variables only.

Example:
```
OPENF1_API_TOKEN=
```

The project must not make a paid provider token a required dependency.

## Local web
```bash
make web
```
The FastAPI service listens on port 8000.
