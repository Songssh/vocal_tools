# vocal_tools

`vocal_tools` is a headless Python audio engine package for source separation, RVC conversion, audio mixing, model catalogs, and reusable cover-pipeline helpers.

It is meant to be imported by other Python projects. It does not include a FastAPI server, web UI, database, or job queue.

## Documentation

- Main docs: [docs/README.md](docs/README.md)
- Final project summary: [docs/project_summary.md](docs/project_summary.md)
- CLI usage: [docs/cli.md](docs/cli.md)
- Release checklist: [docs/release_checklist.md](docs/release_checklist.md)

## Quick Install

```powershell
python -m pip install -e .
```

## Quick CLI Checks

```powershell
vocal-tools models list --models-dir test_models
vocal-tools voices list --models-dir test_models
python -m vocal_tools.cli --help
```
