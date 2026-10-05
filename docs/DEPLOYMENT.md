# Deployment

## Development
GitHub Codespaces + FastAPI/Uvicorn.

## Pre-deployment
1. Run tests.
2. Run Ruff.
3. Run mypy where applicable.
4. Validate model/data artifacts.
5. Re-check provider terms.
6. Confirm secrets are server-side.
7. Run API smoke tests.

## Runtime
Frontend → API → prediction service → stored model/features → data providers.

Live provider access remains server-side.

## Rollback
Version the API and model artifacts so a previous known-good version can be restored without rewriting historical datasets.

## Production checklist
- [ ] Tests pass
- [ ] Lint/type checks pass
- [ ] Secrets configured
- [ ] CORS restricted
- [ ] Rate limiting enabled
- [ ] Health endpoint verified
- [ ] Stale/disconnected states verified
- [ ] Provider terms reviewed
