# Data Cleaning Decisions and Rationale

## Overview

This document describes the cleaning decisions made in Stage 03 and their rationale.

## 1. Dropped Columns (100% Missing in 1993)

**Decision**: Drop columns that are 100% missing in 1993 data.

**Columns Dropped**:
- `TailNum`, `AirTime`, `TaxiIn`, `TaxiOut`
- `CancellationCode`
- `CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay`

**Rationale**:
- These columns provide no information for 1993 analysis
- Keeping them would create schema inconsistencies between years
- They are not available in 1993, so cannot be used for cross-year comparisons

**Impact**:
- 1993: 19 columns (down from 29)
- 2003: 29 columns (these columns exist but have missingness patterns)

## 2. Dropped Rows (Low Missingness Columns)

**Decision**: Drop rows where columns with <2% missingness have NULL values.

**Columns Affected**:
- `Distance` (1993): <2% missing
- `AirTime` (2003): 1.74% missing

**Rationale**:
- Low missingness (<2%) suggests data quality issues rather than systematic patterns
- These are critical operational fields (distance, air time) that should be available
- Small enough impact to drop without significant data loss
- Better to have complete records than impute critical operational metrics

**Impact**:
- 1993: ~76,914 rows dropped (1.5% of data after other filters)
- 2003: Minimal impact (AirTime missingness handled)

## 3. Delay Columns Missingness (2003)

**Decision**: Keep delay columns as NULL (no imputation).

**Columns**:
- `CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay`
- `CancellationCode`

**Missingness Pattern**:
- Q1 2003: 100% missing
- Q2 2003: 66.42% missing
- Q3-Q4 2003: 0% missing (fully available)
- Overall: 41.10% missing

**Rationale**:
- This is a **systematic data collection change**, not random missingness
- Delay columns were not collected in Q1-Q2 2003, then started in Q3
- Imputing with 0 would be misleading (we don't know if delays were 0 or just not collected)
- Tree-based models (LightGBM, XGBoost) handle missing values natively
- Preserves data integrity and allows models to learn from availability patterns

**Impact**:
- Delay breakdown analysis only valid for Q3-Q4 2003 (50% of year)
- Overall delay analysis (ArrDelay, DepDelay) not significantly impacted
- Models can use `has_delay_breakdown` flag to understand data availability

## 4. Feature Flag: `has_delay_breakdown`

**Decision**: Add binary feature flag indicating delay breakdown data availability.

**Implementation**:
- Added to `clean_2003` table only (1993 doesn't have delay columns)
- `1`: At least one delay column is not NULL (data available)
- `0`: All delay columns are NULL (data not available)

**Rationale**:
- Helps models understand when delay breakdown data is available
- Can be used as a feature to improve predictions
- Documents the systematic missingness pattern
- Enables conditional feature usage in modeling

**Usage in Modeling**:
- Can be used as a feature to indicate data quality/availability
- Models can learn different patterns for records with/without delay breakdown
- Helps interpret model behavior across different data collection periods

## 5. Extreme ArrDelay Values

**Decision**: Drop rows with ArrDelay outside the range [-80, +150] minutes and save them to separate extremes tables.

**Implementation**:
- Filter: `ArrDelay >= -80 AND ArrDelay <= 150`
- Extreme values saved to `extremes_1993` and `extremes_2003` tables
- Extremes tables saved to `parquet/clean/extremes/` for separate analysis

**Rationale**:
- Arrival delays >2.5 hours (150 min) are extreme and likely data errors or exceptional circumstances
- Early arrivals >80 minutes are also likely data errors
- More aggressive filter ensures main analysis focuses on typical operational delays
- Saving extremes separately allows for later investigation without contaminating main analysis

**Impact**:
- 1993: 17,658 rows removed (0.354% of data)
  - ArrDelay < -80: 11 rows
  - ArrDelay > 150: 17,647 rows
- 2003: 40,773 rows removed (0.640% of data)
  - ArrDelay < -80: 26 rows
  - ArrDelay > 150: 40,747 rows

**Extreme Values Saved**:
- Tables: `extremes_1993`, `extremes_2003` (in DuckDB)
- Parquet: `parquet/clean/extremes/year=1993/`, `parquet/clean/extremes/year=2003/`
- Can be analyzed separately to understand patterns in extreme delays

## 6. Winsorization

**Decision**: Apply winsorization to extreme outliers (if configured).

**Implementation**:
- Clips values at 0.5th and 99.5th percentiles
- Applied to `ArrDelay` and `DepDelay` (if exists)
- Configurable via `winsorize` parameter in config
- **Note**: Currently configured but not applied (ArrDelay extremes handled via filtering instead)

**Rationale**:
- Extreme outliers (e.g., delays >24 hours) are likely data errors
- Winsorization preserves distribution shape while removing extreme values
- Better than dropping outliers (preserves sample size)

## Summary of Cleaning Steps

1. ✅ Remove invalid time values (outside 0-2400 range)
2. ✅ Remove cancelled flights
3. ✅ Remove diverted flights
4. ✅ Remove rows missing ArrDelay
5. ✅ Drop columns 100% missing in 1993
6. ✅ Drop rows with missing values in low-missingness columns (<2%)
7. ✅ Add `has_delay_breakdown` feature flag (2003 only)
8. ✅ Extract and remove extreme ArrDelay values (outside [-60, +360] minutes)
9. ✅ Apply winsorization (if configured, currently not applied)

## Final Data Quality

**1993**:
- Initial: 5,070,501 rows
- Final: ~4,975,929 rows (after ArrDelay filter, ~98.1% retained)
- Extreme ArrDelay rows: 17,658 (saved to `extremes_1993`)
- Columns: 19 (10 dropped)

**2003**:
- Initial: 6,488,540 rows
- Final: ~6,331,131 rows (after ArrDelay filter, ~97.6% retained)
- Extreme ArrDelay rows: 40,773 (saved to `extremes_2003`)
- Columns: 30 (includes `has_delay_breakdown` flag)
- Delay breakdown available: ~58.9% of records (Q3-Q4)

## Model Implications

1. **Missing Values**: Models must handle missing delay columns (tree-based models recommended)
2. **Feature Engineering**: Can use `has_delay_breakdown` as a feature
3. **Temporal Patterns**: Delay breakdown analysis only valid for Q3-Q4 2003
4. **Cross-Year Comparison**: Delay breakdowns only comparable for Q3-Q4 2003 vs 1993 (if applicable)

