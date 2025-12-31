# Flight Delay Analysis Project

Analysis comparing flight delay patterns between 1993 and 2003.

## Setup

### Virtual Environment

This project uses a Python virtual environment for dependency isolation.

**Activate the virtual environment:**
```bash
source .venv/bin/activate
```

**Deactivate when done:**
```bash
deactivate
```

### Dependencies

All dependencies are listed in `pyproject.toml` and have been installed in the virtual environment:

- `duckdb` - Database for analytics
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `plotly` - Interactive visualizations
- `kaleido` - Static image export for Plotly
- `scikit-learn` - Machine learning
- `lightgbm` - Gradient boosting framework
- `pyyaml` - YAML configuration parsing

**To reinstall dependencies:**
```bash
source .venv/bin/activate
pip install duckdb pandas numpy plotly kaleido scikit-learn lightgbm pyyaml
```

## Pipeline Execution

Run pipeline stages in order:

```bash
source .venv/bin/activate
python scripts/00_run_manifest.py
python scripts/01_ingest.py
python scripts/02_audit.py
# ... and so on
```

## Project Structure

See `project_structure.md` and `project_pipeline.md` for detailed documentation.

