# Data provenance and licensing

## Jolpica-F1

Purpose: structured historical Formula 1 records.

Project policy: use the API for non-commercial research while complying with the API terms, attribution requirements, rate limits, and CC BY-NC-SA 4.0 licensing described by the project.

Reference:
- https://github.com/jolpica/jolpica-f1

## FastF1

Purpose: richer historical F1 session/lap/telemetry/weather processing.

FastF1 is software distributed under the MIT license. Its upstream motorsport data sources may have separate terms; do not assume the software license grants redistribution rights to all retrieved data.

Reference:
- https://github.com/theOehrly/Fast-F1

## OpenF1

Purpose: live race timing and historical replay/live data.

OpenF1 documentation and terms distinguish historical access from live access and describe different access plans. Before any public/commercial deployment, re-check current provider terms and required access tier.

Reference:
- https://openf1.org/
- https://openf1.org/docs/

## Repository policy

1. Keep raw provider responses separate from derived ML datasets.
2. Store retrieval timestamps and source identifiers.
3. Never commit provider credentials.
4. Re-check provider terms before enabling public data redistribution.
5. Preserve point-in-time correctness when creating training examples.
