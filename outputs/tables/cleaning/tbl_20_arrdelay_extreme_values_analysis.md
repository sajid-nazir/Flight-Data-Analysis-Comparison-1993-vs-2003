# ArrDelay Extreme Values Analysis

## Summary

Winsorization is **configured** in `config/params.yaml` (`winsorize: true`, `winsor_q_low: 0.005`, `winsor_q_high: 0.995`) but **NOT actually applied** in the cleaning stage. This document analyzes extreme ArrDelay values and provides recommendations.

---

## Extreme Value Counts

### 1993 Data (4,993,587 total flights)

**Positive Extreme Delays:**
- `ArrDelay > 200 min`: 7,189 flights (0.1440%)
- `ArrDelay > 300 min`: 1,346 flights (0.0270%)
- `ArrDelay > 400 min`: 323 flights (0.0065%)
- `ArrDelay > 500 min`: 116 flights (0.0023%)
- `ArrDelay > 600 min`: 54 flights (0.0011%)
- `ArrDelay > 800 min`: 28 flights (0.0006%)
- `ArrDelay > 1000 min`: 8 flights (0.0002%)

**Negative Extreme Delays (Early Arrivals):**
- `ArrDelay < -50 min`: 243 flights (0.0049%)
- `ArrDelay < -100 min`: 11 flights (0.0002%)
- `ArrDelay < -200 min`: 10 flights (0.0002%)
- `ArrDelay < -500 min`: 8 flights (0.0002%)

**Distribution:**
- Min: -829 minutes
- Max: 1,291 minutes
- 99th percentile: 101 minutes
- 99.5th percentile: 133 minutes
- 99.9th percentile: 222 minutes

**Top 5 Extreme Delays (>600 min):**
1. 1,291 min (SLC → DFW, Month 2)
2. 1,191 min (CLE → BUF, Month 8)
3. 1,159 min (EGE → DFW, Month 1)
4. 1,023 min (SEA → PDX, Month 1)
5. 1,023 min (RDU → LGA, Month 2)

---

### 2003 Data (6,371,904 total flights)

**Positive Extreme Delays:**
- `ArrDelay > 200 min`: 17,813 flights (0.2796%)
- `ArrDelay > 300 min`: 4,571 flights (0.0717%)
- `ArrDelay > 400 min`: 1,622 flights (0.0255%)
- `ArrDelay > 500 min`: 872 flights (0.0137%)
- `ArrDelay > 600 min`: 633 flights (0.0099%)
- `ArrDelay > 800 min`: 385 flights (0.0060%)
- `ArrDelay > 1000 min`: 245 flights (0.0038%)

**Negative Extreme Delays (Early Arrivals):**
- `ArrDelay < -50 min`: 1,072 flights (0.0168%)
- `ArrDelay < -100 min`: 20 flights (0.0003%)
- `ArrDelay < -200 min`: 12 flights (0.0002%)
- `ArrDelay < -500 min`: 10 flights (0.0002%)

**Distribution:**
- Min: -937 minutes
- Max: 1,612 minutes
- 99th percentile: 126 minutes
- 99.5th percentile: 165 minutes
- 99.9th percentile: 274 minutes

**Top 5 Extreme Delays (>600 min):**
1. 1,612 min (IAD → LGA, Month 4)
2. 1,584 min (SJU → MSP, Month 12)
3. 1,517 min (GEG → MSP, Month 8)
4. 1,494 min (MIA → MSP, Month 4)
5. 1,454 min (HNL → MSP, Month 4)

---

## Analysis & Recommendations

### Volume Assessment

**1993:**
- Extreme delays (>600 min): 54 flights (0.0011%) - **Very small, can cap or drop**
- Very high delays (>300 min): 1,346 flights (0.027%) - **Small, can cap**
- High delays (>200 min): 7,189 flights (0.144%) - **Small, can cap**

**2003:**
- Extreme delays (>600 min): 633 flights (0.0099%) - **Small, can cap or drop**
- Very high delays (>300 min): 4,571 flights (0.072%) - **Small, can cap**
- High delays (>200 min): 17,813 flights (0.280%) - **Small, can cap**

### Recommended Approach

Based on volume analysis, **recommended approach: Filter ArrDelay to [-80, +150] minutes and save extremes separately**

**Rationale:**
1. **Volume is small**: Extreme values (>150 min) represent <1% of data
2. **More aggressive filter**: Focuses main analysis on typical operational delays
3. **Preserve extremes**: Save extreme values to separate tables for investigation
4. **Realistic bounds**: 
   - -80 min: Reasonable early arrival limit
   - +150 min: 2.5 hours is reasonable maximum for typical operational analysis

### Alternative Approaches

**Option 1: Winsorization (RECOMMENDED)**
- **Pros**: Preserves all rows, maintains distribution shape, handles outliers gracefully
- **Cons**: Changes extreme values (but they're likely data errors anyway)
- **Implementation**: Apply winsorization at 0.5th and 99.5th percentiles

**Option 2: Drop Extreme Values**
- **Pros**: Removes clearly erroneous data
- **Cons**: Loses ~0.01% of data, may introduce bias if errors are systematic
- **Implementation**: Drop rows where `ArrDelay > 600` or `ArrDelay < -500`

**Option 3: Cap at Fixed Thresholds**
- **Pros**: Simple, interpretable
- **Cons**: May be too aggressive or too lenient
- **Implementation**: Cap at ±300 minutes (or ±200 minutes)

**Option 4: Hybrid Approach**
- **Pros**: Handles different severity levels differently
- **Cons**: More complex
- **Implementation**: 
  - Winsorize at 99.5th percentile for high delays
  - Drop only the most extreme (>1000 min) as likely data errors

---

## Recommended Implementation

### Step 1: Apply ArrDelay Filter

Filter ArrDelay to [-80, +150] minutes and extract extremes:

**1993:**
- Lower bound: -80 minutes
- Upper bound: +150 minutes
- Expected extremes: ~17,658 rows (0.354%)

**2003:**
- Lower bound: -80 minutes
- Upper bound: +150 minutes
- Expected extremes: ~40,773 rows (0.640%)

### Step 2: Verify Impact

After filtering:
- Check that extreme values are removed from main dataset
- Verify extremes are saved to separate tables
- Confirm acceptable data loss (<1%)

### Step 3: Document Decision

Update cleaning documentation to note:
- ArrDelay filter applied: [-80, +150] minutes
- Extreme values saved to separate tables
- Rationale: Focus main analysis on typical operational delays, preserve extremes for investigation

---

## Next Steps

1. ✅ **Implementation complete**: ArrDelay filter implemented
2. ✅ **Filter range**: [-80, +150] minutes
3. ⏳ **Re-run Stage 03** (Clean) to apply filter
4. ⏳ **Re-run Stage 04** (EDA) to regenerate summaries with filtered data
5. ⏳ **Verify** that extreme values are saved to extremes tables

---

## Impact Assessment

**Before Filter:**
- Max delay: 1,291 min (1993), 1,612 min (2003)
- Extreme outliers present

**After Filter (expected):**
- Max delay: 150 min (both years)
- Min delay: -80 min (both years)
- Distribution focused on typical operational delays
- Extreme rows saved to separate tables

**Data Loss:**
- 1993: 17,658 rows (0.354%) moved to extremes table
- 2003: 40,773 rows (0.640%) moved to extremes table
- Total: 58,431 rows (0.515% of combined dataset)
- **Extreme values preserved** in separate tables for investigation


