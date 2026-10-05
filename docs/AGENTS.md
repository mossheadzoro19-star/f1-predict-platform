# AI Coding Agent Instructions

## Source of truth
Read relevant files in `/docs` and existing implementation before significant changes.

## Rules
1. Inspect before editing.
2. Make the smallest correct change.
3. Preserve existing behavior unless the task explicitly changes it.
4. Reuse existing functions and dependencies.
5. Do not invent requirements.
6. Do not add paid APIs.
7. Do not fake live data.
8. Do not introduce deep learning or infrastructure without evidence.
9. Never bypass leakage controls.
10. Do not modify unrelated files.

## ML rules
- Keep chronological evaluation fixed unless explicitly justified.
- 2025 test remains untouched for model selection.
- Primary metric is LogLoss.
- Treat accuracy and maximum probability as secondary diagnostics, never as equivalent to model correctness.

## UI rules
- Follow DESIGN_SYSTEM.md.
- Use Pixel Code/pixel-grid visual language.
- Prefer CSS/Unicode procedural F1-car motifs over heavy image assets.
- Always distinguish LIVE, REPLAY, FALLBACK, STALE and DISCONNECTED.

## Workflow
Before non-trivial work:
1. Read docs.
2. Inspect relevant code.
3. Identify affected files.
4. Create a concise plan.
5. Implement.
6. Run tests/lint/type checks.
7. Review diff.
8. Update documentation if behavior changed.
