# Stage 04 — EDA: Detailed Implementation Plan

## Overview

Stage 04 performs **Exploratory Data Analysis** on the cleaned common-columns datasets to:
1. Compute core KPIs for both years
2. Analyze on-time performance across multiple dimensions
3. Identify common routes and airports for fair comparison
4. Generate descriptive visualizations for 1993 vs 2003 comparison

## Key Principles (from Analysis Strategy)

- **Use `clean_common` datasets**: Both years have identical 19 columns for fair comparison
- **Route-matched analysis**: Compare only routes that exist in both years
- **Airport-matched analysis**: Compare only airports that exist in both years
- **Volume controls**: Report sample sizes to ensure meaningful comparisons
- **Exclude delay breakdown columns**: Not available in 1993, so not used in main comparison

---

## Inputs

### Data Sources
- `parquet/clean/common/year=1993/**` (partitioned Parquet files)
- `parquet/clean/common/year=2003/**` (partitioned Parquet files)

### Configuration
- `config/params.yaml`:
  - `on_time_threshold_min: 15` (defines on-time as ArrDelay <= 15 minutes)
  - `export_png: true` (generate static images)
  - `plotly_template: "plotly_white"`

### Data Access Method
- Use DuckDB to scan Parquet files directly (no need to load into memory)
- Query pattern: `SELECT * FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')`

---

## Detailed Actions

### 1. Core KPIs by Year

**Query Logic:**
```sql
SELECT 
    Year,
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay,
    MEDIAN(ArrDelay) as median_arr_delay,
    AVG(DepDelay) as mean_dep_delay,
    MEDIAN(DepDelay) as median_dep_delay,
    AVG(Distance) as mean_distance,
    MEDIAN(Distance) as median_distance,
    MIN(ArrDelay) as min_arr_delay,
    MAX(ArrDelay) as max_arr_delay,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY ArrDelay) as p25_arr_delay,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY ArrDelay) as p75_arr_delay
FROM read_parquet('parquet/clean/common/year=*/**/*.parquet')
GROUP BY Year
```

**Output:** `tbl_10_core_kpis_by_year.csv`

**Metrics:**
- Total flights
- On-time rate (%)
- Mean/median arrival delay
- Mean/median departure delay
- Mean/median distance
- Min/max delays
- Quartiles (25th, 75th percentile)

---

### 2. On-Time Rates by Month

**Query Logic:**
```sql
SELECT 
    Year,
    Month,
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay,
    MEDIAN(ArrDelay) as median_arr_delay
FROM read_parquet('parquet/clean/common/year=*/**/*.parquet')
GROUP BY Year, Month
ORDER BY Year, Month
```

**Output:** `tbl_11_ontime_by_month.parquet`

**Visualization:** Line chart comparing monthly on-time rates (1993 vs 2003)

---

### 3. On-Time Rates by Day of Week

**Query Logic:**
```sql
SELECT 
    Year,
    DayOfWeek,
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay,
    MEDIAN(ArrDelay) as median_arr_delay
FROM read_parquet('parquet/clean/common/year=*/**/*.parquet')
GROUP BY Year, DayOfWeek
ORDER BY Year, DayOfWeek
```

**Output:** `tbl_12_ontime_by_dow.parquet`

**Visualization:** Bar chart comparing day-of-week patterns (1993 vs 2003)

**Note:** DayOfWeek: 1=Monday, 7=Sunday (or as defined in data)

---

### 4. On-Time Rates by Departure Hour

**Query Logic:**
```sql
SELECT 
    Year,
    CAST(DepTime / 100 AS INTEGER) as dep_hour,  -- Extract hour from HHMM format
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay,
    MEDIAN(ArrDelay) as median_arr_delay
FROM read_parquet('parquet/clean/common/year=*/**/*.parquet')
WHERE DepTime IS NOT NULL
GROUP BY Year, dep_hour
ORDER BY Year, dep_hour
```

**Output:** `tbl_13_ontime_by_dep_hour.parquet`

**Visualization:** Line chart showing hourly patterns (1993 vs 2003)

**Note:** Handle NULL DepTime values. Consider binning hours (e.g., early morning, peak, late night) for visualization.

---

### 5. Carrier Summary

**Query Logic:**
```sql
SELECT 
    Year,
    UniqueCarrier,
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay,
    MEDIAN(ArrDelay) as median_arr_delay,
    AVG(Distance) as mean_distance
FROM read_parquet('parquet/clean/common/year=*/**/*.parquet')
GROUP BY Year, UniqueCarrier
ORDER BY Year, total_flights DESC
```

**Output:** `tbl_14_carrier_summary.parquet`

**Visualization:** 
- Top 10 carriers by volume for each year
- Bar charts showing on-time rates
- Compare carrier rankings between years

---

### 6. Origin Airport Summary

**Query Logic:**
```sql
SELECT 
    Year,
    Origin,
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay,
    MEDIAN(ArrDelay) as median_arr_delay,
    COUNT(DISTINCT Dest) as unique_destinations
FROM read_parquet('parquet/clean/common/year=*/**/*.parquet')
GROUP BY Year, Origin
ORDER BY Year, total_flights DESC
```

**Output:** `tbl_15_origin_airport_summary.parquet`

**Visualization:** Top 20 origin airports by volume (for each year)

---

### 7. Destination Airport Summary

**Query Logic:**
```sql
SELECT 
    Year,
    Dest,
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay,
    MEDIAN(ArrDelay) as median_arr_delay,
    COUNT(DISTINCT Origin) as unique_origins
FROM read_parquet('parquet/clean/common/year=*/**/*.parquet')
GROUP BY Year, Dest
ORDER BY Year, total_flights DESC
```

**Output:** `tbl_16_dest_airport_summary.parquet`

**Visualization:** Top 20 destination airports by volume (for each year)

---

### 8. Route Summary

**Query Logic:**
```sql
SELECT 
    Year,
    Origin,
    Dest,
    CONCAT(Origin, '-', Dest) as route,
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay,
    MEDIAN(ArrDelay) as median_arr_delay,
    AVG(Distance) as mean_distance,
    MEDIAN(Distance) as median_distance
FROM read_parquet('parquet/clean/common/year=*/**/*.parquet')
GROUP BY Year, Origin, Dest, route
ORDER BY Year, total_flights DESC
```

**Output:** `tbl_17_route_summary.parquet`

**Visualization:** Top routes by volume (for each year)

---

### 9. Route-Matched Analysis (NEW)

**Purpose:** Compare only routes that exist in both years for fair comparison.

**Step 1: Identify Common Routes**
```sql
-- Get routes present in both years
WITH routes_1993 AS (
    SELECT DISTINCT CONCAT(Origin, '-', Dest) as route
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
),
routes_2003 AS (
    SELECT DISTINCT CONCAT(Origin, '-', Dest) as route
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
)
SELECT route
FROM routes_1993
INTERSECT
SELECT route
FROM routes_2003
```

**Step 2: Compute Metrics for Common Routes**
```sql
SELECT 
    Year,
    route,
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay,
    MEDIAN(ArrDelay) as median_arr_delay
FROM read_parquet('parquet/clean/common/year=*/**/*.parquet')
WHERE CONCAT(Origin, '-', Dest) IN (
    -- Common routes from Step 1
)
GROUP BY Year, route
ORDER BY Year, total_flights DESC
```

**Output:** `tbl_18_route_matched_summary.parquet`

**Visualization:** 
- Scatter plot: 1993 on-time rate vs 2003 on-time rate (for common routes)
- Top 20 common routes by volume with side-by-side comparison

---

### 10. Airport-Matched Analysis (NEW)

**Purpose:** Compare only airports (origin or destination) that exist in both years.

**Step 1: Identify Common Airports**
```sql
-- Get airports present in both years (as origin OR destination)
WITH airports_1993 AS (
    SELECT DISTINCT Origin as airport FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    UNION
    SELECT DISTINCT Dest as airport FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
),
airports_2003 AS (
    SELECT DISTINCT Origin as airport FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
    UNION
    SELECT DISTINCT Dest as airport FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
)
SELECT airport
FROM airports_1993
INTERSECT
SELECT airport
FROM airports_2003
```

**Step 2: Compute Metrics for Common Airports (as Origin)**
```sql
SELECT 
    Year,
    Origin as airport,
    'origin' as airport_role,
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay
FROM read_parquet('parquet/clean/common/year=*/**/*.parquet')
WHERE Origin IN (
    -- Common airports from Step 1
)
GROUP BY Year, Origin
ORDER BY Year, total_flights DESC
```

**Step 3: Compute Metrics for Common Airports (as Destination)**
```sql
SELECT 
    Year,
    Dest as airport,
    'destination' as airport_role,
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay
FROM read_parquet('parquet/clean/common/year=*/**/*.parquet')
WHERE Dest IN (
    -- Common airports from Step 1
)
GROUP BY Year, Dest
ORDER BY Year, total_flights DESC
```

**Output:** `tbl_19_airport_matched_summary.parquet`

**Visualization:**
- Top 20 common airports (by total volume across both years)
- Side-by-side comparison of on-time rates (1993 vs 2003)
- Separate views for origin and destination airports

---

## Output Files

### Tables (CSV/Parquet)

1. `outputs/tables/eda/tbl_10_core_kpis_by_year.csv`
2. `outputs/tables/eda/tbl_11_ontime_by_month.parquet`
3. `outputs/tables/eda/tbl_12_ontime_by_dow.parquet`
4. `outputs/tables/eda/tbl_13_ontime_by_dep_hour.parquet`
5. `outputs/tables/eda/tbl_14_carrier_summary.parquet`
6. `outputs/tables/eda/tbl_15_origin_airport_summary.parquet`
7. `outputs/tables/eda/tbl_16_dest_airport_summary.parquet`
8. `outputs/tables/eda/tbl_17_route_summary.parquet`
9. `outputs/tables/eda/tbl_18_route_matched_summary.parquet` (NEW)
10. `outputs/tables/eda/tbl_19_airport_matched_summary.parquet` (NEW)

### Visualizations (JSON + PNG)

1. `outputs/viz/eda/viz_11_kpi_panel_1993_vs_2003.plotly.json` + PNG
   - Multi-panel dashboard showing core KPIs side-by-side

2. `outputs/viz/eda/viz_12_ontime_by_month_1993_vs_2003.plotly.json` + PNG
   - Line chart: Monthly on-time rates (1993 vs 2003)

3. `outputs/viz/eda/viz_13_ontime_by_dow_1993_vs_2003.plotly.json` + PNG
   - Grouped bar chart: Day-of-week patterns (1993 vs 2003)

4. `outputs/viz/eda/viz_14_ontime_by_dep_hour_1993_vs_2003.plotly.json` + PNG
   - Line chart: Hourly patterns (1993 vs 2003)

5. `outputs/viz/eda/viz_15_top10_carriers_1993.plotly.json` + PNG
   - Bar chart: Top 10 carriers by volume and on-time rate (1993)

6. `outputs/viz/eda/viz_16_top10_carriers_2003.plotly.json` + PNG
   - Bar chart: Top 10 carriers by volume and on-time rate (2003)

7. `outputs/viz/eda/viz_17_route_matched_comparison.plotly.json` + PNG (NEW)
   - Scatter plot: 1993 vs 2003 on-time rates for common routes
   - Top 20 common routes side-by-side comparison

8. `outputs/viz/eda/viz_18_airport_matched_comparison.plotly.json` + PNG (NEW)
   - Top 20 common airports side-by-side comparison (origin and destination)

---

## Implementation Structure

### Script: `scripts/04_eda.py`

**Main Function Flow:**
1. Load configuration
2. Connect to DuckDB
3. Verify clean_common Parquet files exist
4. Compute all aggregate tables (1-10 above)
5. Generate all visualizations
6. Save all outputs (tables + JSON + PNG)

### Module: `src/flight_delay/eda.py`

**Functions to Create:**
- `compute_core_kpis(conn, parquet_path_1993, parquet_path_2003) -> pd.DataFrame`
- `compute_ontime_by_month(conn, parquet_path_1993, parquet_path_2003) -> pd.DataFrame`
- `compute_ontime_by_dow(conn, parquet_path_1993, parquet_path_2003) -> pd.DataFrame`
- `compute_ontime_by_dep_hour(conn, parquet_path_1993, parquet_path_2003) -> pd.DataFrame`
- `compute_carrier_summary(conn, parquet_path_1993, parquet_path_2003) -> pd.DataFrame`
- `compute_airport_summary(conn, parquet_path_1993, parquet_path_2003, role='origin') -> pd.DataFrame`
- `compute_route_summary(conn, parquet_path_1993, parquet_path_2003) -> pd.DataFrame`
- `compute_route_matched_summary(conn, parquet_path_1993, parquet_path_2003) -> pd.DataFrame`
- `compute_airport_matched_summary(conn, parquet_path_1993, parquet_path_2003) -> pd.DataFrame`

**Visualization Functions:**
- `create_kpi_panel(kpi_df) -> go.Figure`
- `create_monthly_comparison(monthly_df) -> go.Figure`
- `create_dow_comparison(dow_df) -> go.Figure`
- `create_hourly_comparison(hourly_df) -> go.Figure`
- `create_carrier_charts(carrier_df, year) -> go.Figure`
- `create_route_matched_chart(route_matched_df) -> go.Figure`
- `create_airport_matched_chart(airport_matched_df) -> go.Figure`

---

## Key Considerations

### 1. Parquet Path Handling
- Use glob patterns: `parquet/clean/common/year=1993/**/*.parquet`
- DuckDB's `read_parquet()` supports glob patterns
- Ensure paths are absolute or relative to project root

### 2. NULL Handling
- `DepTime` may have NULLs (handle in hour extraction)
- Use `WHERE DepTime IS NOT NULL` for hourly analysis
- Consider binning NULLs separately if significant

### 3. Volume Controls
- Report sample sizes (total_flights) in all summaries
- Filter out groups with very low volume (< min_group_volume) if needed
- Document volume thresholds in output tables

### 4. Route Definition
- Route = `CONCAT(Origin, '-', Dest)`
- Ensure consistent formatting (uppercase, no spaces)

### 5. Departure Hour Extraction
- `DepTime` is in HHMM format (e.g., 1430 = 2:30 PM)
- Extract hour: `CAST(DepTime / 100 AS INTEGER)`
- Handle edge cases: 2400 = midnight (hour 24 or 0?)

### 6. Visualization Aesthetics
- Use consistent color scheme: 1993 = `#2E86AB`, 2003 = `#A23B72`
- Include volume annotations (flight counts) where relevant
- Use Plotly's `plotly_white` template
- Save both JSON (interactive) and PNG (static)

### 7. Performance
- Use DuckDB's efficient Parquet scanning
- Consider materializing intermediate results if queries are slow
- Use appropriate aggregation functions (MEDIAN may be slower than PERCENTILE_CONT)

---

## Validation Checks

Before completing Stage 04, verify:

1. ✅ All output tables exist and have data
2. ✅ All visualizations are generated (JSON + PNG)
3. ✅ Route-matched analysis includes only routes present in both years
4. ✅ Airport-matched analysis includes only airports present in both years
5. ✅ On-time rates are calculated correctly (ArrDelay <= 15)
6. ✅ Sample sizes (total_flights) are reasonable for all groups
7. ✅ No NULL values in key aggregations (unless expected)
8. ✅ Visualizations are aesthetically pleasing and readable

---

## Next Stage

After Stage 04, proceed to **Stage 05 — Compare: standardized deltas with volume controls**, which will:
- Compute weighted overall deltas
- Create per-dimension delta tables
- Generate "delta vs volume" visualizations
- Use the EDA summaries as inputs


