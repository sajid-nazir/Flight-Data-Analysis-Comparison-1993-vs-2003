# Modeling Bias Concern: Delay Breakdown Columns

## The Problem

You've identified a critical issue: **temporal bias in delay breakdown availability could skew model training**.

### Data Pattern

**2003 Delay Breakdown Availability:**
- **Q1**: 100% missing (1,570,786 rows)
- **Q2**: 66.38% missing (1,048,200 rows), 33.62% available (530,926 rows)
- **Q3-Q4**: 100% available (3,221,992 rows)

**Delay Statistics by Availability:**
- **Not Available (Q1-Q2)**: 
  - Avg Delay: 1.99 min
  - Median Delay: -3.00 min
  - OnTime15: 85.38%
  
- **Available (Q2-Q4)**:
  - Avg Delay: 4.07 min
  - Median Delay: -2.00 min
  - OnTime15: 83.45%

### The Risk

If we use delay breakdown columns (`CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay`) as features:

1. **Temporal Confounding**: The model might learn that `has_delay_breakdown=0` (Q1-Q2) predicts lower delays, not because of the missing data, but because Q1-Q2 actually had better on-time performance.

2. **Seasonal Bias**: The model could learn seasonal patterns (Q1-Q2 vs Q3-Q4) through the delay breakdown availability flag, which is not a causal relationship.

3. **Cross-Year Generalization**: Since 1993 has NO delay breakdown columns, models trained on 2003 with delay breakdown features won't generalize to 1993.

4. **Train/Test Split Issues**: If train/test splits are by month, and Q1-Q2 are in training while Q3-Q4 are in testing (or vice versa), the model will see a different data distribution.

## Recommended Solutions

### Option 1: Exclude Delay Breakdown Columns from Features (RECOMMENDED)

**Decision**: Do NOT use delay breakdown columns as features for prediction.

**Rationale**:
- These columns are not available in 1993 (cross-year comparison impossible)
- Only available in 50% of 2003 data (Q3-Q4)
- Introduces temporal/seasonal confounding
- The `has_delay_breakdown` flag itself could introduce bias

**Implementation**:
- In Stage 06 (Features), exclude these columns from feature engineering:
  - `CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay`
  - `CancellationCode` (also not in 1993)
  - `has_delay_breakdown` flag (optional - could keep if we want to model data quality)

**Pros**:
- Clean, unbiased features
- Works for both 1993 and 2003
- No temporal confounding
- Simpler model interpretation

**Cons**:
- Loses potentially useful delay breakdown information for Q3-Q4 2003
- But: this information is only available for 50% of data, so not reliable anyway

### Option 2: Separate Models for Different Data Availability Periods

**Decision**: Train separate models:
- Model A: Using all features except delay breakdown (works for all data)
- Model B: Using delay breakdown features (only for Q3-Q4 2003)

**Rationale**:
- Maximizes use of available data
- Model A can be used for all periods, Model B only for Q3-Q4

**Pros**:
- Uses all available information
- Clear separation of concerns

**Cons**:
- More complex
- Model B only works for 50% of 2003 data
- Can't compare 1993 vs 2003 using Model B

### Option 3: Use Delay Breakdown Only for Q3-Q4 2003

**Decision**: Only use delay breakdown columns for Q3-Q4 2003 records.

**Rationale**:
- Avoids temporal bias by only using data where it's consistently available
- Still loses 50% of 2003 data

**Pros**:
- Uses delay breakdown where available
- Avoids Q1-Q2 bias

**Cons**:
- Loses 50% of 2003 data
- Still can't use for 1993
- Complex feature engineering

### Option 4: Keep Delay Breakdown but Control for Temporal Effects

**Decision**: Use delay breakdown columns but add temporal features (quarter, month) to control for seasonality.

**Rationale**:
- Model can learn both delay breakdown effects and seasonal effects separately

**Pros**:
- Uses all available information
- Controls for temporal confounding

**Cons**:
- Still has missingness issues
- Model complexity increases
- Harder to interpret
- Still can't use for 1993

## Final Recommendation

**Use Option 1: Exclude Delay Breakdown Columns from Features**

### Justification:

1. **Cross-Year Consistency**: 1993 has no delay breakdown columns. For fair comparison, we should use the same feature set for both years.

2. **Data Availability**: Delay breakdown is only available for 50% of 2003 data. Using features that are missing 50% of the time is problematic.

3. **Temporal Bias**: The correlation between delay breakdown availability and delay patterns is likely seasonal, not causal. Using these features would introduce confounding.

4. **Model Simplicity**: Simpler models are easier to interpret and less prone to overfitting.

5. **Prediction Moment**: According to the pipeline, we're predicting at "scheduled" moment (before departure). Delay breakdown columns are only known AFTER the flight, so they shouldn't be used for prediction anyway!

### What to Keep:

- **Core features**: Month, DayOfWeek, DepHour, Carrier, Origin, Dest, Distance
- **Congestion proxies**: Origin hourly volume, Dest hourly volume
- **Route features**: Origin-Dest combinations
- **Temporal features**: Month, DayOfWeek (these are always available)

### What to Exclude:

- ❌ `CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay`
- ❌ `CancellationCode`
- ❌ `has_delay_breakdown` (optional - could keep as data quality indicator, but probably not needed)

### Implementation:

Update Stage 06 (Features) to explicitly exclude delay breakdown columns from feature engineering.

## Impact on Analysis

- **Delay Breakdown Analysis**: Can still be done separately for Q3-Q4 2003 as a descriptive analysis, but not as model features.
- **Model Training**: Will use consistent feature set across both years.
- **Model Interpretation**: Cleaner, no temporal confounding.
- **Cross-Year Comparison**: Fair comparison possible since both years use same features.

