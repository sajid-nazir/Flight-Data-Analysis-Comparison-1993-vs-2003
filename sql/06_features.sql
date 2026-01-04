-- Stage 06: Build feature tables for modeling

-- 1. Build base features from clean parquet data
SELECT 
    Year,
    Month,
    DayOfWeek,
    DepTime,
    CRSDepTime,
    UniqueCarrier,
    Origin,
    Dest,
    Distance,
    CRSElapsedTime,
    ArrDelay,
    -- Build target variable
    CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END as ontime15,
    -- Build route feature
    Origin || '_' || Dest as route,
    -- Build departure hour from CRSDepTime
    CAST(FLOOR(CRSDepTime / 100) AS INTEGER) as dep_hour_raw
FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
WHERE ArrDelay IS NOT NULL;

-- Note: Binning is done in Python:
-- dep_hour_bin: [0,6)='late_night', [6,10)='early_morning', [10,14)='mid_morning',
--                [14,18)='afternoon', [18,22)='evening', [22,24]='night'
-- distance_bin: [0,500)='short', [500,1000)='medium', [1000,1500)='long',
--                [1500,2000)='very_long', [2000,10000]='ultra_long'

-- 2. Compute congestion features (hourly volumes at origin and destination)
WITH base_data AS (
    SELECT 
        Year,
        Month,
        DayOfWeek,
        CAST(FLOOR(CRSDepTime / 100) AS INTEGER) as dep_hour,
        Origin,
        Dest
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    WHERE CRSDepTime IS NOT NULL
),
origin_hourly AS (
    SELECT 
        Year, Month, DayOfWeek, dep_hour, Origin,
        COUNT(*) as origin_hourly_volume
    FROM base_data
    GROUP BY Year, Month, DayOfWeek, dep_hour, Origin
),
dest_hourly AS (
    SELECT 
        Year, Month, DayOfWeek, dep_hour, Dest,
        COUNT(*) as dest_hourly_volume
    FROM base_data
    GROUP BY Year, Month, DayOfWeek, dep_hour, Dest
),
all_combinations AS (
    SELECT DISTINCT Year, Month, DayOfWeek, dep_hour, Origin, Dest
    FROM base_data
)
SELECT 
    a.Year, a.Month, a.DayOfWeek, a.dep_hour, a.Origin, a.Dest,
    COALESCE(o.origin_hourly_volume, 0) as origin_hourly_volume,
    COALESCE(d.dest_hourly_volume, 0) as dest_hourly_volume
FROM all_combinations a
LEFT JOIN origin_hourly o
    ON a.Year = o.Year 
    AND a.Month = o.Month 
    AND a.DayOfWeek = o.DayOfWeek 
    AND a.dep_hour = o.dep_hour
    AND a.Origin = o.Origin
LEFT JOIN dest_hourly d
    ON a.Year = d.Year 
    AND a.Month = d.Month 
    AND a.DayOfWeek = d.DayOfWeek 
    AND a.dep_hour = d.dep_hour
    AND a.Dest = d.Dest;

-- 3. Create train/test split assignments (based on months)
-- Note: Split assignment is done in Python based on config:
-- train_months: [1,2,3,4,5,6,7,8,9]
-- test_months: [10,11,12]
-- This query shows the logic:
SELECT 
    Year,
    Month,
    CASE 
        WHEN Month IN (1,2,3,4,5,6,7,8,9) THEN 'train'
        WHEN Month IN (10,11,12) THEN 'test'
        ELSE 'other'
    END as split
FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
GROUP BY Year, Month
ORDER BY Year, Month;

-- 4. Fit target encoders (mean encoding) on training data only
-- This computes the mean of ontime15 for each category value
-- Example for UniqueCarrier:
SELECT 
    UniqueCarrier,
    AVG(ontime15) as target_encoded_value,
    COUNT(*) as count
FROM (
    SELECT 
        UniqueCarrier,
        CASE WHEN ArrDelay <= 15 THEN 1.0 ELSE 0.0 END as ontime15
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    WHERE ArrDelay IS NOT NULL
    AND Month IN (1,2,3,4,5,6,7,8,9)  -- Training months only
)
GROUP BY UniqueCarrier
ORDER BY count DESC;

-- Example for Origin:
SELECT 
    Origin,
    AVG(ontime15) as target_encoded_value,
    COUNT(*) as count
FROM (
    SELECT 
        Origin,
        CASE WHEN ArrDelay <= 15 THEN 1.0 ELSE 0.0 END as ontime15
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    WHERE ArrDelay IS NOT NULL
    AND Month IN (1,2,3,4,5,6,7,8,9)
)
GROUP BY Origin
ORDER BY count DESC;

-- Example for route:
SELECT 
    Origin || '_' || Dest as route,
    AVG(ontime15) as target_encoded_value,
    COUNT(*) as count
FROM (
    SELECT 
        Origin,
        Dest,
        CASE WHEN ArrDelay <= 15 THEN 1.0 ELSE 0.0 END as ontime15
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    WHERE ArrDelay IS NOT NULL
    AND Month IN (1,2,3,4,5,6,7,8,9)
)
GROUP BY route
ORDER BY count DESC;

-- 5. Apply target encoders to create model-ready features
-- This would join the encoded values back to the base features
-- Example structure (actual implementation done in Python):
WITH base_features AS (
    SELECT 
        Year, Month, DayOfWeek, CRSDepTime, dep_hour_raw, Distance, CRSElapsedTime,
        UniqueCarrier, Origin, Dest, route, dep_hour_bin, distance_bin,
        origin_hourly_volume, dest_hourly_volume,
        CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END as ontime15,
        CASE 
            WHEN Month IN (1,2,3,4,5,6,7,8,9) THEN 'train'
            WHEN Month IN (10,11,12) THEN 'test'
            ELSE 'other'
        END as split
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    WHERE ArrDelay IS NOT NULL
),
carrier_encoding AS (
    SELECT 
        UniqueCarrier,
        AVG(ontime15) as carrier_freq
    FROM base_features
    WHERE split = 'train'
    GROUP BY UniqueCarrier
),
origin_encoding AS (
    SELECT 
        Origin,
        AVG(ontime15) as origin_freq
    FROM base_features
    WHERE split = 'train'
    GROUP BY Origin
),
dest_encoding AS (
    SELECT 
        Dest,
        AVG(ontime15) as dest_freq
    FROM base_features
    WHERE split = 'train'
    GROUP BY Dest
),
route_encoding AS (
    SELECT 
        route,
        AVG(ontime15) as route_freq
    FROM base_features
    WHERE split = 'train'
    GROUP BY route
)
SELECT 
    bf.Year, bf.Month, bf.DayOfWeek, bf.split, bf.ontime15,
    bf.CRSDepTime, bf.dep_hour_raw, bf.Distance, bf.CRSElapsedTime,
    bf.origin_hourly_volume, bf.dest_hourly_volume,
    COALESCE(ce.carrier_freq, 0.5) as UniqueCarrier_freq,
    COALESCE(oe.origin_freq, 0.5) as Origin_freq,
    COALESCE(de.dest_freq, 0.5) as Dest_freq,
    COALESCE(re.route_freq, 0.5) as route_freq
FROM base_features bf
LEFT JOIN carrier_encoding ce ON bf.UniqueCarrier = ce.UniqueCarrier
LEFT JOIN origin_encoding oe ON bf.Origin = oe.Origin
LEFT JOIN dest_encoding de ON bf.Dest = de.Dest
LEFT JOIN route_encoding re ON bf.route = re.route;

-- 6. Feature summary statistics
SELECT 
    'ontime15' as feature,
    COUNT(*) as count,
    AVG(ontime15) as mean,
    STDDEV(ontime15) as stddev,
    MIN(ontime15) as min,
    MAX(ontime15) as max,
    SUM(CASE WHEN ontime15 IS NULL THEN 1 ELSE 0 END) as missing_count
FROM (
    SELECT 
        CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END as ontime15
    FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet')
    WHERE ArrDelay IS NOT NULL
)
UNION ALL
SELECT 
    'Distance' as feature,
    COUNT(*) as count,
    AVG(Distance) as mean,
    STDDEV(Distance) as stddev,
    MIN(Distance) as min,
    MAX(Distance) as max,
    SUM(CASE WHEN Distance IS NULL THEN 1 ELSE 0 END) as missing_count
FROM read_parquet('parquet/clean/common/year=1993/**/*.parquet');
