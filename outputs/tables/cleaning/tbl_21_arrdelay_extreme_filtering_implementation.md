# ArrDelay Extreme Value Filtering - Implementation Summary

## Overview

Implemented filtering of extreme ArrDelay values outside the range [-60, +360] minutes. Extreme values are **dropped from the main cleaned dataset** but **saved to separate extremes tables** for later analysis.

---

## Implementation Details

### 1. Filter Range
- **Minimum**: -80 minutes (early arrivals >80 min are likely data errors)
- **Maximum**: +150 minutes (delays >2.5 hours are likely data errors or exceptional circumstances)

### 2. Code Changes

#### `src/flight_delay/clean.py`
- **Modified `create_clean_table()`**:
  - Added parameters: `arrdelay_min`, `arrdelay_max`, `apply_arrdelay_filter`
  - Added ArrDelay range filter to WHERE clause when `apply_arrdelay_filter=True`
  
- **New function `extract_extreme_arrdelay_values()`**:
  - Extracts rows with ArrDelay outside [-60, +360] to separate table
  - Returns count of extreme rows
  - Prints breakdown (too negative vs too positive)

#### `scripts/03_clean.py`
- **Modified cleaning workflow**:
  1. Create temp table with all filters EXCEPT ArrDelay range
  2. Extract extreme ArrDelay values to `extremes_{year}` table
  3. Create final clean table with ArrDelay filter applied
  4. Save extremes tables to Parquet
  
- **Updated ledger**:
  - Added "Remove Extreme ArrDelay" step to cleaning ledger
  - Tracks rows removed and percentage

### 3. Data Storage

**DuckDB Tables:**
- `extremes_1993`: Extreme ArrDelay rows from 1993
- `extremes_2003`: Extreme ArrDelay rows from 2003

**Parquet Files:**
- `parquet/clean/extremes/year=1993/`: Partitioned Parquet for 1993 extremes
- `parquet/clean/extremes/year=2003/`: Partitioned Parquet for 2003 extremes

---

## Impact Analysis

### 1993 Data
- **Total rows before filter**: 4,993,587
- **Rows removed**: 17,658 (0.354%)
  - ArrDelay < -80: 11 rows
  - ArrDelay > 150: 17,647 rows
- **Final rows**: ~4,975,929 (99.646% retained)

### 2003 Data
- **Total rows before filter**: 6,371,904
- **Rows removed**: 40,773 (0.640%)
  - ArrDelay < -80: 26 rows
  - ArrDelay > 150: 40,747 rows
- **Final rows**: ~6,331,131 (99.360% retained)

### Summary
- **Total extreme rows**: 58,431 (0.515% of combined dataset)
- **Data loss**: Small but acceptable (<1%)
- **Extreme values preserved**: Yes, in separate tables for analysis

---

## Rationale

1. **Volume is small**: <1% of data, so dropping has acceptable impact
2. **Likely data errors**: Delays >2.5 hours or early arrivals >80 min are likely errors or exceptional circumstances
3. **Preserve for analysis**: Extreme values saved separately allow investigation without contaminating main analysis
4. **Realistic bounds**: 
   - -80 min: Reasonable early arrival limit (beyond this likely data errors)
   - +150 min: 2.5 hours is a reasonable maximum delay for typical operational analysis (beyond this likely exceptional circumstances or errors)

---

## Usage

### Accessing Extreme Values

**In DuckDB:**
```sql
SELECT * FROM extremes_1993;
SELECT * FROM extremes_2003;
```

**From Parquet:**
```python
import pandas as pd
extremes_1993 = pd.read_parquet('parquet/clean/extremes/year=1993/')
extremes_2003 = pd.read_parquet('parquet/clean/extremes/year=2003/')
```

### Analysis Opportunities

The extremes tables can be used to:
1. Investigate patterns in extreme delays
2. Identify data quality issues
3. Understand exceptional circumstances (weather events, system failures, etc.)
4. Validate data quality improvements

---

## Next Steps

1. ✅ **Implementation complete**
2. ⏳ **Re-run Stage 03** to apply the filter
3. ⏳ **Re-run Stage 04** (EDA) to regenerate summaries with filtered data
4. ⏳ **Optional**: Analyze extremes tables to understand patterns

---

## Files Modified

1. `src/flight_delay/clean.py` - Added filtering logic and extremes extraction
2. `scripts/03_clean.py` - Updated workflow to extract and save extremes
3. `outputs/tables/cleaning/tbl_11_cleaning_decisions.md` - Updated documentation
4. `outputs/tables/cleaning/tbl_20_arrdelay_extreme_values_analysis.md` - Analysis document
5. `outputs/tables/cleaning/tbl_21_arrdelay_extreme_filtering_implementation.md` - This document

---

## Testing

- ✅ Code syntax validated
- ✅ Imports successful
- ⏳ Full pipeline test (requires re-running Stage 03)

