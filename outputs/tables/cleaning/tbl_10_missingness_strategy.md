# Missingness Strategy for 2003 Delay Columns

## Summary

After cleaning, we have identified a systematic missingness pattern in 2003 data for delay-related columns.

## Columns Dropped from 1993 (100% Missing)

The following 10 columns were **dropped** from 1993 data as they are 100% missing:
- `TailNum`
- `AirTime`
- `TaxiIn`
- `TaxiOut`
- `CancellationCode`
- `CarrierDelay`
- `WeatherDelay`
- `NASDelay`
- `SecurityDelay`
- `LateAircraftDelay`

## 2003 Missingness Pattern

### Columns with No Missingness (0%)
- `TailNum`: 0% missing (fully available)
- `TaxiIn`: 0% missing (fully available)
- `TaxiOut`: 0% missing (fully available)

### Columns with Low Missingness (~1.74%)
- `AirTime`: 1.74% missing (likely random missingness, can be imputed or handled in modeling)

### Columns with Systematic Missingness (41.19% overall)

The following delay-related columns show a **systematic pattern** by quarter:
- `CancellationCode`
- `CarrierDelay`
- `WeatherDelay`
- `NASDelay`
- `SecurityDelay`
- `LateAircraftDelay`

**Missingness by Quarter:**
- **Q1 (Jan-Mar)**: 100% missing
- **Q2 (Apr-Jun)**: 66.42% missing
- **Q3 (Jul-Sep)**: 0% missing (fully available)
- **Q4 (Oct-Dec)**: 0% missing (fully available)

## Interpretation

This is **NOT random missingness** but a **data collection change**:
- Delay breakdown columns were not collected in Q1 2003
- Partial collection started in Q2 2003 (33.58% available)
- Full collection began in Q3 2003 and continued through Q4

## Recommendations

### Option 1: Keep as NULL (Recommended for Modeling)
- **Pros**: 
  - Preserves data integrity
  - Models can handle missing values
  - No false assumptions about delay values
- **Cons**: 
  - Models need to handle missingness
  - Some algorithms require imputation
- **Best for**: Tree-based models (LightGBM, XGBoost) that handle missing values natively

### Option 2: Impute with 0 (NOT Recommended)
- **Pros**: 
  - Simple, no missing values
- **Cons**: 
  - **Misleading**: We don't know if delay was actually 0 or just not collected
  - Introduces bias (assumes no delays in Q1/Q2)
  - Violates data integrity
- **Best for**: Only if you're certain these flights had no delays (unlikely)

### Option 3: Create Availability Flags
- **Pros**: 
  - Preserves original data
  - Models can learn from availability pattern
  - Can use flags as features
- **Cons**: 
  - Adds complexity
  - Still need to handle missing values
- **Best for**: Advanced modeling where data collection patterns matter

### Option 4: Drop Q1/Q2 Records for Delay Analysis (NOT Recommended)
- **Pros**: 
  - Clean delay data
- **Cons**: 
  - Loses ~40% of 2003 data
  - Other columns (ArrDelay, DepDelay) are still valid
  - Breaks temporal continuity
- **Best for**: Only if delay breakdown is critical and can't handle missingness

## Recommended Approach (IMPLEMENTED)

**For the current pipeline:**
1. ✅ **Drop rows with missing values in columns with <2% missingness** (e.g., AirTime, Distance)
   - These are likely data quality issues, not systematic patterns
   - Small enough impact to drop without significant data loss
2. ✅ **Keep delay columns as NULL** in cleaned data (no imputation)
   - Preserves data integrity
   - Avoids false assumptions about delay values
3. ✅ **Added `has_delay_breakdown` feature flag** (1 if delay columns available, 0 otherwise)
   - Indicates whether delay breakdown data was collected
   - Can be used as a feature in modeling
   - Helps models understand data availability patterns
4. ✅ **Document this pattern** for model interpretation

**Implementation Details:**
- **Low missingness threshold**: 2% (columns with <2% missingness have rows dropped)
- **Delay columns**: Kept as NULL where missing (systematic pattern, not random)
- **Feature flag**: `has_delay_breakdown` added to clean_2003 table
  - `1`: Delay breakdown data available (Q3/Q4 2003)
  - `0`: Delay breakdown data not available (Q1/Q2 2003)

## Impact on Analysis

- **Overall delay analysis**: Not significantly impacted (ArrDelay and DepDelay are available)
- **Delay breakdown analysis**: Only valid for Q3/Q4 2003 (50% of year)
- **1993 vs 2003 comparison**: Delay breakdowns only comparable for Q3/Q4 2003
- **Modeling**: Models should handle missing delay breakdowns gracefully

