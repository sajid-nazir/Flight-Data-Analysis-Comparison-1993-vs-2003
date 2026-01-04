# Flight Delay Analysis Project

Comprehensive analysis comparing flight delay patterns between 1993 and 2003, using machine learning models to predict on-time arrival and assess cross-year generalization.

## Project Status: ✅ COMPLETE

All 14 pipeline stages have been executed successfully:
- ✅ Data ingestion and audit
- ✅ Data cleaning and quality analysis
- ✅ Exploratory data analysis (EDA)
- ✅ Comparative analysis (1993 vs 2003)
- ✅ Feature engineering with target encoding
- ✅ Within-year model training (Logistic Regression + LightGBM)
- ✅ Cross-year generalization analysis
- ✅ Model interpretability (feature importance, rank shifts)
- ✅ System-level visualizations
- ✅ Drilldown tables for interactive exploration
- ✅ Visualization registry
- ✅ Final report generation

**Key Deliverables:**
- 50+ interactive visualizations (Plotly JSON + PNG)
- 16 drilldown tables for web app integration
- Trained models (within-year and cross-year) with hyperparameter tuning
- SHAP interpretability analysis (directional feature effects)
- Baseline comparison analysis
- Comprehensive final report with research findings
- Complete data quality analysis

**Enhanced Features:**
- **Hyperparameter Tuning**: Optuna-based Bayesian optimization for LightGBM
- **SHAP Values**: Advanced interpretability showing how features affect predictions
- **Baseline Comparison**: Context for model performance vs. naive baselines
- **Enhanced Reporting**: "Why" explanations for observed patterns

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
- `optuna` - Hyperparameter optimization
- `shap` - Model interpretability (SHAP values)
- `pyarrow` - Parquet file support

**To reinstall dependencies:**
```bash
source .venv/bin/activate
pip install duckdb pandas numpy plotly kaleido scikit-learn lightgbm pyyaml optuna shap pyarrow
```

## Pipeline Execution

Run pipeline stages in order:

```bash
source .venv/bin/activate
python scripts/00_run_manifest.py
python scripts/01_ingest.py
python scripts/02_audit.py
python scripts/03_clean.py
python scripts/04_eda.py
python scripts/05_compare.py
python scripts/06_features.py
python scripts/07_train_within_year.py
python scripts/08_train_cross_year.py
python scripts/09_interpret.py
python scripts/10_system_viz.py
python scripts/12_build_drilldowns.py
python scripts/11_build_registry.py
python scripts/13_build_report.py
```

## Key Findings

### Research Question 1: Predictability from Departure-Time Information
✅ **Yes** - Models achieve strong predictive performance (ROC-AUC > 0.64) using only information available at scheduled departure time.

### Research Question 2: 1993 vs 2003 Prediction Environment Difference
✅ **Yes** - Meaningful differences exist, with cross-year generalization loss of 5-13%, indicating non-trivial distribution shifts.

**Model Performance:**
- 1993 LightGBM: ROC-AUC = 0.6566, PR-AUC = 0.9002
- 2003 LightGBM: ROC-AUC = 0.6303, PR-AUC = 0.8911
- 1993 Logistic Regression: ROC-AUC = 0.6367, PR-AUC = 0.8916
- 2003 Logistic Regression: ROC-AUC = 0.6118, PR-AUC = 0.8835
- Cross-year generalization loss: 4.76% - 12.46%
- Hyperparameter tuning: 10 trials (configurable, currently set to 10 for faster iteration)

## Outputs

### Visualizations
- **50+ charts** available in both interactive (Plotly JSON) and static (PNG) formats
- Located in: `outputs/viz/` and `outputs/figures/`
- Registry: `outputs/registry/viz_registry.json`
- Includes: ROC curves, PR curves, calibration plots, baseline comparisons, SHAP summaries

### Tables
- Cleaning ledger, null counts, model metrics, feature importance
- Located in: `outputs/tables/`

### Models
- Within-year models: `models/within_year/`
- Cross-year models: `models/cross_year/`
- Encoders: `encoders/`

### Report
- Final report: `reports/final_report.md`
- Includes executive summary, research questions, findings, and methodology

## Data Quality

**Raw Data:**
- 1993: 5,070,501 flights → 4,975,929 after cleaning
- 2003: 6,488,540 flights → 6,331,131 after cleaning

**Null Patterns:**
- 1993: 10 columns 100% null (not collected in 1993)
- 2003: Delay breakdown columns ~41% null (cancelled/diverted flights)
- Both years: ArrDelay nulls ~1.4-1.7% (cancelled/diverted flights)

See `outputs/tables/audit/tbl_null_counts_raw_*.csv` for detailed null analysis.

## Project Structure

See `project_structure.md` and `project_pipeline.md` for detailed documentation.

