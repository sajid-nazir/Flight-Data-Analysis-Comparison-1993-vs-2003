# Analysis Strategy: Fair 1993 vs 2003 Comparison

## Overview

This document outlines the analysis strategy to ensure fair, unbiased comparison between 1993 and 2003 flight data, addressing composition changes, temporal bias, and data availability differences.

## 1. Locked On-Time Definition

### Primary Outcome: OnTime15
- **Definition**: `ArrDelay <= 15` minutes
- **Applied to**: Completed flights only (cancelled and diverted already excluded in cleaning)
- **Consistent across**: Both years use identical definition

### Secondary Outcome: CompletedAsScheduled (Optional)
- **Definition**: Flight completed as scheduled (not cancelled, not diverted)
- **Purpose**: Detect if system "looks" on-time by cancelling difficult flights
- **Note**: Already handled in cleaning (cancelled/diverted removed), but can track separately if needed

## 2. Dual Data Versions for 2003

### Version A: `clean_common` (Primary for Comparison)
- **Purpose**: Common-columns comparison (fair cross-year analysis)
- **Columns**: Only columns available in both 1993 and 2003
- **Excludes**: Delay breakdown columns (CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay, CancellationCode)
- **Excludes**: `has_delay_breakdown` flag (to avoid temporal bias)
- **Use for**: Main 1993 vs 2003 comparison, modeling, feature engineering

### Version B: `clean_full` (2003-Only Diagnostic)
- **Purpose**: 2003-only deep dive with delay breakdown
- **Columns**: All available columns including delay breakdown
- **Includes**: Delay breakdown columns for Q3-Q4 2003
- **Use for**: 
  - 2003-only diagnostic analysis
  - Understanding delay contributors (Q3-Q4 only)
  - Appendix/exploratory analysis

## 3. Common Columns (Ex-Ante Characteristics)

### Primary Features for Comparison
These are known **before** the flight operates (at "scheduled" moment):

1. **Temporal**:
   - `Month` / Season
   - `DayOfWeek`
   - `DepHour` (binned: early morning, peak, late night)

2. **Operational**:
   - `UniqueCarrier` (carrier)
   - `Origin` (origin airport)
   - `Dest` (destination airport)
   - `Route` (Origin-Dest combination)
   - `Distance` (binned: short/medium/long-haul)
   - `CRSElapsedTime` (scheduled elapsed time - proxy for schedule padding)

3. **Derived**:
   - Origin hourly volume (congestion proxy)
   - Dest hourly volume (congestion proxy)
   - Route frequency

### Excluded from Main Comparison
- Delay breakdown columns (not in 1993, temporal bias in 2003)
- Ex-post operational characteristics (DepDelay, TaxiOut, TaxiIn, AirTime) - use as secondary lens only

## 4. Composition Adjustment Methods

### A) "As-Operated" Comparison (Descriptive)
- Compare overall on-time rates and patterns by carrier/time/airport/distance in each year
- Shows what actually happened
- **Use for**: Understanding overall system performance

### B) "Like-for-Like" Comparison (Fairer)
Three approaches (implement at least one):

#### B1) Route-Matched Analysis
- **Method**: Restrict to routes present in both years
- **Compare**: Within-route on-time rates
- **Output**: Route-level comparison
- **Use for**: Fair comparison controlling for route differences

#### B2) Airport-Matched Analysis
- **Method**: Restrict to top N airports common to both years
- **Compare**: Within-airport on-time rates
- **Output**: Airport-level comparison
- **Use for**: Understanding hub/airport-specific changes

#### B3) Reweighting/Standardization
- **Method**: Reweight 1993 to match 2003 distribution of key covariates (or vice versa)
- **Compare**: Adjusted on-time rates
- **Output**: Standardized comparison tables
- **Use for**: Overall adjusted comparison controlling for composition

## 5. Two Complementary Analysis Methods

### Method 1: Simple, Interpretable Slices (Recommended)
For each year separately, compute OnTime15 rate by:
- Hour-of-day bin
- Day-of-week
- Month/season
- Distance bin
- Carrier
- Origin airport (top 20)
- Destination airport (top 20)
- Route (top routes)

**Compare**:
- Rankings (e.g., "early morning consistently best in both years")
- Gaps (e.g., "carrier differences widened")
- Patterns (e.g., "seasonal patterns similar/different")

**Output**: Slice comparison tables and visualizations

### Method 2: Single Model Per Year (Identical Features)
Fit a model (Logistic Regression or LightGBM) in **each year** using:
- **Same predictors** (common columns only)
- **Same feature engineering** (identical transformations)

**Compare**:
- Top drivers (feature importance/coefficients)
- Directionality (e.g., "late-night departures hurt more in 2003")
- Strength (magnitude of effects)

**Output**: Model comparison, feature importance comparison, coefficient comparison

## 6. Delay Breakdown Columns: 2003-Only Diagnostic

### Safe Uses (2003-Only)
1. **2003-only story**: "Among late flights (ArrDelay>15) with breakdown available (Q3-Q4), the largest contributors were..."
2. **Bias-aware appendix**: Show that breakdown-available flights are systematically more delayed, so breakdown analyses are conditional on that subset

### NOT Used For
- ❌ Defining "on-time characteristics" across 1993 vs 2003
- ❌ Main comparison features
- ❌ Cross-year modeling

## 7. Implementation Plan

### Stage 03 (Clean) - UPDATED
- Create `clean_common` version (without delay breakdown) for both years
- Create `clean_full` version (with delay breakdown) for 2003 only
- Both versions exclude cancelled/diverted

### Stage 04 (EDA) - UPDATED
- Use `clean_common` for main comparison
- Add route-matched analysis
- Add airport-matched analysis
- Compute slices by all dimensions
- Use `clean_full` for 2003-only delay breakdown analysis (appendix)

### Stage 05 (Compare) - UPDATED
- "As-operated" comparison using `clean_common`
- "Like-for-like" comparison:
  - Route-matched analysis
  - Airport-matched analysis
  - Optional: Reweighting analysis
- Delta tables with volume controls

### Stage 06 (Features) - UPDATED
- Use `clean_common` only
- Exclude delay breakdown columns
- Use common columns only
- Create identical feature sets for both years

### Stage 07-08 (Modeling) - UPDATED
- Train models on `clean_common` data
- Use identical features for both years
- Compare feature importance/coefficients
- Cross-year generalization tests

## 8. Expected Outputs

### Main Comparison (Common Columns)
- Overall on-time rates (as-operated and like-for-like)
- Slice comparisons by dimension
- Model comparisons (feature importance, coefficients)
- Route-matched and airport-matched analyses

### 2003-Only Diagnostic (Full Columns)
- Delay breakdown analysis (Q3-Q4 only)
- Delay contributor analysis
- Bias documentation (breakdown-available vs not-available)

## 9. Key Principles

1. **Fairness**: Use same features, same definitions, same methods
2. **Transparency**: Document all composition differences
3. **Robustness**: Multiple comparison methods (as-operated + like-for-like)
4. **Interpretability**: Simple slices + models for different audiences
5. **Bias Awareness**: Acknowledge and document temporal/data collection biases

