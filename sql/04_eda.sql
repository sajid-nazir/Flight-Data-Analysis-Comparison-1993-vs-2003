-- Stage 04: Exploratory Data Analysis queries

-- 1. Core KPIs by Year
WITH all_data AS (
    SELECT Year, ArrDelay, DepDelay, Distance
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    UNION ALL
    SELECT Year, ArrDelay, DepDelay, Distance
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
)
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
FROM all_data
GROUP BY Year
ORDER BY Year;

-- 2. On-Time Rates by Month
WITH all_data AS (
    SELECT Year, Month, ArrDelay
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    UNION ALL
    SELECT Year, Month, ArrDelay
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
)
SELECT 
    Year,
    Month,
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay,
    MEDIAN(ArrDelay) as median_arr_delay
FROM all_data
GROUP BY Year, Month
ORDER BY Year, Month;

-- 3. On-Time Rates by Day of Week
WITH all_data AS (
    SELECT Year, DayOfWeek, ArrDelay
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    UNION ALL
    SELECT Year, DayOfWeek, ArrDelay
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
)
SELECT 
    Year,
    DayOfWeek,
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay,
    MEDIAN(ArrDelay) as median_arr_delay
FROM all_data
GROUP BY Year, DayOfWeek
ORDER BY Year, DayOfWeek;

-- 4. On-Time Rates by Departure Hour
WITH all_data AS (
    SELECT Year, DepTime, ArrDelay
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    WHERE DepTime IS NOT NULL
    UNION ALL
    SELECT Year, DepTime, ArrDelay
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
    WHERE DepTime IS NOT NULL
),
with_hour AS (
    SELECT 
        Year,
        CAST(DepTime / 100 AS INTEGER) as dep_hour,
        ArrDelay
    FROM all_data
    WHERE DepTime >= 0 AND DepTime <= 2400
)
SELECT 
    Year,
    dep_hour,
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay,
    MEDIAN(ArrDelay) as median_arr_delay
FROM with_hour
GROUP BY Year, dep_hour
ORDER BY Year, dep_hour;

-- 5. Carrier Summary
WITH all_data AS (
    SELECT Year, UniqueCarrier, ArrDelay, Distance
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    UNION ALL
    SELECT Year, UniqueCarrier, ArrDelay, Distance
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
)
SELECT 
    Year,
    UniqueCarrier,
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay,
    MEDIAN(ArrDelay) as median_arr_delay,
    AVG(Distance) as mean_distance
FROM all_data
GROUP BY Year, UniqueCarrier
ORDER BY Year, total_flights DESC;

-- 6. Origin Airport Summary
WITH all_data AS (
    SELECT Year, Origin, Dest, ArrDelay
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    UNION ALL
    SELECT Year, Origin, Dest, ArrDelay
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
)
SELECT 
    Year,
    Origin as airport,
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay,
    MEDIAN(ArrDelay) as median_arr_delay,
    COUNT(DISTINCT Dest) as unique_destinations
FROM all_data
GROUP BY Year, Origin
ORDER BY Year, total_flights DESC;

-- 7. Destination Airport Summary
WITH all_data AS (
    SELECT Year, Origin, Dest, ArrDelay
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    UNION ALL
    SELECT Year, Origin, Dest, ArrDelay
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
)
SELECT 
    Year,
    Dest as airport,
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay,
    MEDIAN(ArrDelay) as median_arr_delay,
    COUNT(DISTINCT Origin) as unique_origins
FROM all_data
GROUP BY Year, Dest
ORDER BY Year, total_flights DESC;

-- 8. Route Summary
WITH all_data AS (
    SELECT Year, Origin, Dest, ArrDelay, Distance
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    UNION ALL
    SELECT Year, Origin, Dest, ArrDelay, Distance
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
)
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
FROM all_data
GROUP BY Year, Origin, Dest, route
ORDER BY Year, total_flights DESC;

-- 9. Route-Matched Analysis (only routes present in both years)
WITH routes_1993 AS (
    SELECT DISTINCT CONCAT(Origin, '-', Dest) as route
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
),
routes_2003 AS (
    SELECT DISTINCT CONCAT(Origin, '-', Dest) as route
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
),
common_routes AS (
    SELECT route FROM routes_1993
    INTERSECT
    SELECT route FROM routes_2003
),
all_data AS (
    SELECT Year, Origin, Dest, ArrDelay
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    UNION ALL
    SELECT Year, Origin, Dest, ArrDelay
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
)
SELECT 
    Year,
    CONCAT(Origin, '-', Dest) as route,
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay,
    MEDIAN(ArrDelay) as median_arr_delay
FROM all_data
WHERE CONCAT(Origin, '-', Dest) IN (SELECT route FROM common_routes)
GROUP BY Year, route
ORDER BY Year, total_flights DESC;

-- 10. Airport-Matched Analysis (only airports present in both years)
WITH airports_1993 AS (
    SELECT DISTINCT Origin as airport FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    UNION
    SELECT DISTINCT Dest as airport FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
),
airports_2003 AS (
    SELECT DISTINCT Origin as airport FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
    UNION
    SELECT DISTINCT Dest as airport FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
),
common_airports AS (
    SELECT airport FROM airports_1993
    INTERSECT
    SELECT airport FROM airports_2003
),
origin_data AS (
    SELECT Year, Origin as airport, 'origin' as airport_role, ArrDelay
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    WHERE Origin IN (SELECT airport FROM common_airports)
    UNION ALL
    SELECT Year, Origin as airport, 'origin' as airport_role, ArrDelay
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
    WHERE Origin IN (SELECT airport FROM common_airports)
),
dest_data AS (
    SELECT Year, Dest as airport, 'destination' as airport_role, ArrDelay
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    WHERE Dest IN (SELECT airport FROM common_airports)
    UNION ALL
    SELECT Year, Dest as airport, 'destination' as airport_role, ArrDelay
    FROM read_parquet('parquet/clean/common/year=2003/**/*.parquet')
    WHERE Dest IN (SELECT airport FROM common_airports)
),
combined_data AS (
    SELECT * FROM origin_data
    UNION ALL
    SELECT * FROM dest_data
)
SELECT 
    Year,
    airport,
    airport_role,
    COUNT(*) as total_flights,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
    AVG(ArrDelay) as mean_arr_delay,
    MEDIAN(ArrDelay) as median_arr_delay
FROM combined_data
GROUP BY Year, airport, airport_role
ORDER BY Year, airport_role, total_flights DESC;
