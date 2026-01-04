# Pipeline Changes Summary Based on Analysis Strategy

## Overview

Based on the comprehensive analysis strategy recommendations, we've implemented key changes to ensure fair, unbiased comparison between 1993 and 2003 data.

## ✅ Implemented Changes

### Stage 03 (Clean) - COMPLETED

**Dual Data Versions Created:**

1. **`clean_common`** (Primary for Comparison)
   - **1993**: 19 columns (common columns only)
   - **2003**: 23 columns (common columns only, excludes delay breakdown)
   - **Location**: `parquet/clean/common/year=YYYY/`
   - **Use for**: Main comparison, modeling, feature engineering

2. **`clean_full`** (2003-Only Diagnostic)
   - **2003**: 30 columns (includes delay breakdown columns)
   - **Location**: `parquet/clean/full/year=2003/`
   - **Use for**: 2003-only delay breakdown analysis (Q3-Q4)

**Key Features:**
- ✅ Excludes delay breakdown columns from common version
- ✅ Excludes `has_delay_breakdown` flag from common version (to avoid temporal bias)
- ✅ Both versions exclude cancelled/diverted flights
- ✅ Both versions have same cleaning filters applied

## 📋 Required Pipeline Updates

### Stage 04 (EDA) - NEEDS UPDATE

**Current**: Uses `parquet/clean/year=YYYY/clean.parquet`

**Should Use**:
- **Main comparison**: `parquet/clean/common/year=YYYY/` (both years)
- **2003 diagnostic**: `parquet/clean/full/year=2003/` (for delay breakdown analysis)

**Add**:
- Route-matched analysis (routes present in both years)
- Airport-matched analysis (top N common airports)
- Slice comparisons by all dimensions (hour, day, month, carrier, airport, route, distance)

### Stage 05 (Compare) - NEEDS UPDATE

**Current**: Standardized deltas with volume controls

**Should Add**:
- **"As-operated" comparison**: Overall rates by dimension (descriptive)
- **"Like-for-like" comparison**:
  - Route-matched analysis (within-route comparison)
  - Airport-matched analysis (within-airport comparison)
  - Optional: Reweighting/standardization analysis

**Input**: Use `parquet/clean/common/` for main comparison

### Stage 06 (Features) - NEEDS UPDATE

**Current**: Uses `parquet/clean/year=YYYY/clean.parquet`

**Should Use**: `parquet/clean/common/year=YYYY/` (common columns only)

**Feature Exclusions** (already documented in `config/feature_exclusions.yaml`):
- ❌ Delay breakdown columns
- ❌ `has_delay_breakdown` flag
- ❌ Ex-post operational characteristics (DepDelay, TaxiOut, TaxiIn, AirTime) - unless using "at_departure" prediction moment

**Features to Include** (ex-ante, common columns):
- ✅ Month, DayOfWeek, DepHour (binned)
- ✅ UniqueCarrier, Origin, Dest, Route
- ✅ Distance (binned)
- ✅ CRSElapsedTime
- ✅ Origin hourly volume, Dest hourly volume (congestion proxies)

### Stage 07-08 (Modeling) - NEEDS UPDATE

**Current**: Within-year and cross-year modeling

**Should Use**: `parquet/features/` built from `clean_common/` data

**Key Requirements**:
- Use identical feature sets for both years
- Compare feature importance/coefficients between years
- Document composition differences
- Use common columns only

### Stage 09+ (Interpretation, Reports) - NEEDS UPDATE

**Should Include**:
- Main comparison using common columns
- Route-matched and airport-matched analyses
- 2003-only delay breakdown analysis (appendix, Q3-Q4 only)
- Bias documentation (temporal patterns, data collection changes)

## 📊 Data Structure

```
parquet/
├── clean/
│   ├── common/          # Common columns (fair comparison)
│   │   ├── year=1993/
│   │   └── year=2003/
│   └── full/            # Full columns (2003 diagnostic only)
│       └── year=2003/
└── features/            # Built from common/ data
    ├── year=1993/
    └── year=2003/
```

## 🎯 Analysis Principles

1. **Fairness**: Same features, same definitions, same methods
2. **Transparency**: Document all composition differences
3. **Robustness**: Multiple comparison methods (as-operated + like-for-like)
4. **Interpretability**: Simple slices + models for different audiences
5. **Bias Awareness**: Acknowledge temporal/data collection biases

## 📝 Next Steps

1. ✅ Stage 03: Dual versions created
2. ⏳ Stage 04: Update to use `clean_common/` and add route/airport matching
3. ⏳ Stage 05: Add like-for-like comparison methods
4. ⏳ Stage 06: Update to use `clean_common/` and exclude delay breakdown
5. ⏳ Stage 07-08: Ensure models use common features
6. ⏳ Stage 09+: Include route-matched analysis in reports

## 🔍 Key Files

- **Analysis Strategy**: `outputs/tables/cleaning/tbl_13_analysis_strategy.md`
- **Feature Exclusions**: `config/feature_exclusions.yaml`
- **Modeling Bias Concern**: `outputs/tables/cleaning/tbl_12_modeling_bias_concern.md`
- **Cleaning Decisions**: `outputs/tables/cleaning/tbl_11_cleaning_decisions.md`

