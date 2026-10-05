# API

## GET /api/health
Returns service health.

## GET /api/snapshot
Returns the current or requested replay snapshot.

Query:
- `replay_offset_seconds` optional, non-negative.

Response:
```json
{
  "mode": "LIVE|REPLAY|FALLBACK|STALE|DISCONNECTED",
  "source": "provider-or-dataset",
  "session": {},
  "updated_at": "ISO-8601",
  "drivers": [],
  "race_state": {},
  "note": "",
  "replay_available": true
}
```

## Driver output
Each driver record may contain:
- driver
- code
- team
- position
- gap_to_leader
- compound
- tyre_age
- win_probability

## API rules
- Never expose provider credentials.
- Validate replay offsets.
- Return explicit provider state.
- Keep probability values in [0,1].
- Return consistent JSON.
