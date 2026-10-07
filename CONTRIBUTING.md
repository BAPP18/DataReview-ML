# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Before opening a pull request

Run:

```bash
ruff check .
ruff format --check .
python -m unittest discover -s tests -v
```

Or:

```bash
make check
```

## Engineering expectations

- Keep model training separate from app inference.
- Do not commit secrets or private datasets.
- Preserve provenance fields when adding ingestion sources.
- Add tests for behavior changes.
- Update model metadata and the model card when changing a released artifact.
- Avoid presenting model probabilities as ground truth.
- Keep reviewer decisions auditable and explicit.

## Commit style

Prefer focused commits such as:

- `fix: prevent duplicate review transitions`
- `test: cover malformed source records`
- `docs: document matcher artifact lifecycle`
- `refactor: isolate dashboard data loading`
