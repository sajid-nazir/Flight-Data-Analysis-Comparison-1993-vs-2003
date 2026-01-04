-- Stage 05: Compare 1993 vs 2003 with standardized deltas

-- Note: Most comparison logic is done in Python using pandas operations on pre-computed EDA tables.
-- These SQL queries show the conceptual logic for computing deltas.

-- 1. Overall weighted delta (conceptual - actual computation done in Python)
-- This shows the logic for computing flight-weighted overall delta
WITH kpi_1993 AS (
    SELECT 
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
),
kpi_2003 AS (
    SELECT 
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
)
SELECT 
    1993 as year,
    kpi_1993.total_flights,
    kpi_1993.ontime_rate_pct as value
FROM kpi_1993
UNION ALL
SELECT 
    2003 as year,
    kpi_2003.total_flights,
    kpi_2003.ontime_rate_pct as value
FROM kpi_2003;

-- Delta calculation (done in Python):
-- delta_absolute = value_2003 - value_1993
-- delta_percent = (delta_absolute / value_1993) * 100
-- weighted_avg = (value_1993 * total_flights_1993 + value_2003 * total_flights_2003) / (total_flights_1993 + total_flights_2003)

-- 2. Delta by Carrier (conceptual - uses pre-computed carrier summary)
-- The actual computation uses the carrier summary table from Stage 04
-- and computes deltas in Python. This shows the underlying data:

WITH carrier_1993 AS (
    SELECT 
        UniqueCarrier,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    GROUP BY UniqueCarrier
),
carrier_2003 AS (
    SELECT 
        UniqueCarrier,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
    GROUP BY UniqueCarrier
)
SELECT 
    COALESCE(c1.UniqueCarrier, c2.UniqueCarrier) as UniqueCarrier,
    COALESCE(c1.total_flights, 0) as total_flights_1993,
    COALESCE(c2.total_flights, 0) as total_flights_2003,
    COALESCE(c1.total_flights, 0) + COALESCE(c2.total_flights, 0) as total_flights_both,
    COALESCE(c1.ontime_rate_pct, 0) as ontime_rate_pct_1993,
    COALESCE(c2.ontime_rate_pct, 0) as ontime_rate_pct_2003,
    COALESCE(c2.ontime_rate_pct, 0) - COALESCE(c1.ontime_rate_pct, 0) as delta_absolute
FROM carrier_1993 c1
FULL OUTER JOIN carrier_2003 c2 ON c1.UniqueCarrier = c2.UniqueCarrier
WHERE (COALESCE(c1.total_flights, 0) + COALESCE(c2.total_flights, 0)) >= 10000  -- min_group_volume
ORDER BY total_flights_both DESC;

-- 3. Delta by Origin Airport (conceptual)
WITH origin_1993 AS (
    SELECT 
        Origin,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    GROUP BY Origin
),
origin_2003 AS (
    SELECT 
        Origin,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
    GROUP BY Origin
)
SELECT 
    COALESCE(o1.Origin, o2.Origin) as Origin,
    COALESCE(o1.ontime_rate_pct, 0) as ontime_rate_pct_1993,
    COALESCE(o2.ontime_rate_pct, 0) as ontime_rate_pct_2003,
    COALESCE(o2.ontime_rate_pct, 0) - COALESCE(o1.ontime_rate_pct, 0) as delta_absolute,
    COALESCE(o1.total_flights, 0) + COALESCE(o2.total_flights, 0) as total_flights_both
FROM origin_1993 o1
FULL OUTER JOIN origin_2003 o2 ON o1.Origin = o2.Origin
WHERE (COALESCE(o1.total_flights, 0) + COALESCE(o2.total_flights, 0)) >= 10000
ORDER BY total_flights_both DESC;

-- 4. Delta by Route (conceptual)
WITH route_1993 AS (
    SELECT 
        CONCAT(Origin, '-', Dest) as route,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    GROUP BY route
),
route_2003 AS (
    SELECT 
        CONCAT(Origin, '-', Dest) as route,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
    GROUP BY route
)
SELECT 
    COALESCE(r1.route, r2.route) as route,
    COALESCE(r1.ontime_rate_pct, 0) as ontime_rate_pct_1993,
    COALESCE(r2.ontime_rate_pct, 0) as ontime_rate_pct_2003,
    COALESCE(r2.ontime_rate_pct, 0) - COALESCE(r1.ontime_rate_pct, 0) as delta_absolute,
    COALESCE(r1.total_flights, 0) + COALESCE(r2.total_flights, 0) as total_flights_both
FROM route_1993 r1
FULL OUTER JOIN route_2003 r2 ON r1.route = r2.route
WHERE (COALESCE(r1.total_flights, 0) + COALESCE(r2.total_flights, 0)) >= 10000
ORDER BY total_flights_both DESC;

-- 5. Delta by Month
WITH monthly_1993 AS (
    SELECT 
        Month,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    GROUP BY Month
),
monthly_2003 AS (
    SELECT 
        Month,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
    GROUP BY Month
)
SELECT 
    COALESCE(m1.Month, m2.Month) as Month,
    COALESCE(m1.ontime_rate_pct, 0) as ontime_rate_pct_1993,
    COALESCE(m2.ontime_rate_pct, 0) as ontime_rate_pct_2003,
    COALESCE(m2.ontime_rate_pct, 0) - COALESCE(m1.ontime_rate_pct, 0) as delta_absolute
FROM monthly_1993 m1
FULL OUTER JOIN monthly_2003 m2 ON m1.Month = m2.Month
ORDER BY Month;

-- 6. Delta by Departure Hour
WITH hourly_1993 AS (
    SELECT 
        CAST(CRSDepTime / 100 AS INTEGER) as dep_hour,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    WHERE CRSDepTime IS NOT NULL AND CRSDepTime >= 0 AND CRSDepTime <= 2400
    GROUP BY dep_hour
),
hourly_2003 AS (
    SELECT 
        CAST(CRSDepTime / 100 AS INTEGER) as dep_hour,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
    WHERE CRSDepTime IS NOT NULL AND CRSDepTime >= 0 AND CRSDepTime <= 2400
    GROUP BY dep_hour
)
SELECT 
    COALESCE(h1.dep_hour, h2.dep_hour) as dep_hour,
    COALESCE(h1.ontime_rate_pct, 0) as ontime_rate_pct_1993,
    COALESCE(h2.ontime_rate_pct, 0) as ontime_rate_pct_2003,
    COALESCE(h2.ontime_rate_pct, 0) - COALESCE(h1.ontime_rate_pct, 0) as delta_absolute
FROM hourly_1993 h1
FULL OUTER JOIN hourly_2003 h2 ON h1.dep_hour = h2.dep_hour
ORDER BY dep_hour;
