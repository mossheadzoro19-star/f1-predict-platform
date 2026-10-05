# Code Style & Conventions

## Principles
- Prefer readable, small functions.
- One logical responsibility per function.
- Reuse existing utilities.
- Avoid unnecessary abstractions.
- Avoid large codedumps and speculative infrastructure.
- Prefer library primitives when they solve the problem cleanly.

## Python
- Python >=3.12.
- Type hints on public functions.
- pandas operations should be vectorized where practical.
- Keep provider I/O separate from transformations.
- Use `pathlib.Path`.
- Follow Ruff configuration in `pyproject.toml`.

## Naming
- snake_case: functions, variables, modules.
- PascalCase: classes.
- UPPER_SNAKE_CASE: constants.

## Error handling
Raise meaningful exceptions at library boundaries; convert provider failures to explicit service states at the API layer.

## Comments
Explain why, not what. Document non-obvious point-in-time/leakage rules.

## Dependencies
Do not add a dependency when a standard library function or an existing dependency is sufficient.
