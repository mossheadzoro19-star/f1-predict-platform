# Security

## Secrets
- Provider tokens stay server-side.
- Use environment variables.
- Never commit real credentials.
- Never expose private environment variables to browser code.

## API
- Validate all query parameters.
- Restrict CORS when the application is deployed publicly.
- Apply rate limiting before public deployment.
- Do not expose stack traces or provider credentials.

## Data
Historical F1 data and derived ML data must respect each provider's current terms and redistribution rights.

## Logging
Never log API tokens or credentials. Log provider state, errors, latency and relevant operational identifiers.

## Checklist
- [ ] Secrets absent from git
- [ ] Provider access server-side
- [ ] Input validation
- [ ] Safe error responses
- [ ] Dependency review
- [ ] Provider licensing review before public deployment
