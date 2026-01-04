# Invalid Time Values Investigation - 2003 Data

## Summary

Found invalid time values in `DepTime` and `ArrTime` columns where values are outside the standard 0-2400 range (HHMM format).

**Note**: 1993 data has **0 invalid time values**, while 2003 has these issues. This suggests a data collection or formatting change between the two years.

## Findings

### DepTime (Departure Time)
- **Total invalid records**: 571 out of 6,387,071 (0.01%)
- **Value range**: 2401-3000 (mostly 2401-2500)
- **Cancellation rate**: 0% (not related to cancellations)
- **Diversion rate**: 1.1% (6 out of 571)

### ArrTime (Arrival Time)
- **Total invalid records**: 3,785 out of 6,375,690 (0.06%)
- **Value range**: 2401-3000 (mostly 2401-2500)
- **Cancellation rate**: 0% (not related to cancellations)
- **Diversion rate**: 0% (not related to diversions)

## Analysis

### Value Distribution

**DepTime:**
- 2401-2500: 472 records (82.7%)
- 2501-3000: 99 records (17.3%)

**ArrTime:**
- 2401-2500: 3,084 records (81.5%)
- 2501-3000: 701 records (18.5%)

### Interpretation

1. **2401-2500 range**: These are likely legitimate times after midnight (e.g., 2405 = 00:05 next day). In aviation data, times crossing midnight are sometimes represented as 2400+ instead of wrapping to 0-59.

2. **2501-3000 range**: These are likely data entry errors or system issues, as they don't correspond to valid 24-hour time representations.

3. **Not related to cancellations/diversions**: The invalid times are not primarily from cancelled or diverted flights, suggesting they represent actual flight times that crossed midnight.

## Recommendations

1. **Data Cleaning**: Consider converting times > 2400 to their equivalent 0-59 format:
   - 2405 → 5 (00:05)
   - 2415 → 15 (00:15)
   - 2500 → 100 (01:00)
   - etc.

2. **Validation**: For values > 2500, investigate further as these may be data quality issues.

3. **Documentation**: Note that 0.01-0.06% of records have this issue, which is a very small percentage and may not significantly impact analysis.

## Sample Records

See `scripts/investigate_invalid_times.py` for detailed sample records and full investigation results.

