-- Stage 03: Clean data and build clean Parquet

-- 1. Get cleaning statistics (row counts at each step)
-- Initial count
SELECT COUNT(*) as initial FROM raw_1993;

-- After removing invalid times
SELECT COUNT(*) as after_invalid_times
FROM raw_1993
WHERE (DepTime IS NULL OR (DepTime >= 0 AND DepTime <= 2400))
AND (ArrTime IS NULL OR (ArrTime >= 0 AND ArrTime <= 2400))
AND (CRSDepTime IS NULL OR (CRSDepTime >= 0 AND CRSDepTime <= 2400))
AND (CRSArrTime IS NULL OR (CRSArrTime >= 0 AND CRSArrTime <= 2400));

-- After removing cancelled
SELECT COUNT(*) as after_cancelled
FROM raw_1993
WHERE (DepTime IS NULL OR (DepTime >= 0 AND DepTime <= 2400))
AND (ArrTime IS NULL OR (ArrTime >= 0 AND ArrTime <= 2400))
AND (CRSDepTime IS NULL OR (CRSDepTime >= 0 AND CRSDepTime <= 2400))
AND (CRSArrTime IS NULL OR (CRSArrTime >= 0 AND CRSArrTime <= 2400))
AND (Cancelled IS NULL OR Cancelled = 0);

-- After removing diverted
SELECT COUNT(*) as after_diverted
FROM raw_1993
WHERE (DepTime IS NULL OR (DepTime >= 0 AND DepTime <= 2400))
AND (ArrTime IS NULL OR (ArrTime >= 0 AND ArrTime <= 2400))
AND (CRSDepTime IS NULL OR (CRSDepTime >= 0 AND CRSDepTime <= 2400))
AND (CRSArrTime IS NULL OR (CRSArrTime >= 0 AND CRSArrTime <= 2400))
AND (Cancelled IS NULL OR Cancelled = 0)
AND (Diverted IS NULL OR Diverted = 0);

-- After removing missing ArrDelay
SELECT COUNT(*) as after_missing_arrdelay
FROM raw_1993
WHERE (DepTime IS NULL OR (DepTime >= 0 AND DepTime <= 2400))
AND (ArrTime IS NULL OR (ArrTime >= 0 AND ArrTime <= 2400))
AND (CRSDepTime IS NULL OR (CRSDepTime >= 0 AND CRSDepTime <= 2400))
AND (CRSArrTime IS NULL OR (CRSArrTime >= 0 AND CRSArrTime <= 2400))
AND (Cancelled IS NULL OR Cancelled = 0)
AND (Diverted IS NULL OR Diverted = 0)
AND ArrDelay IS NOT NULL;

-- 2. Identify columns that are 100% missing (to drop)
SELECT 
    column_name,
    COUNT(*) as total,
    SUM(CASE WHEN column_name IS NULL THEN 1 ELSE 0 END) as missing
FROM raw_1993
GROUP BY column_name
HAVING missing = total;

-- 3. Identify columns with low missingness (<2%) - drop rows where these are NULL
SELECT 
    column_name,
    COUNT(*) as total,
    SUM(CASE WHEN column_name IS NULL THEN 1 ELSE 0 END) as missing,
    ROUND(SUM(CASE WHEN column_name IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as missing_pct
FROM raw_1993
WHERE (DepTime IS NULL OR (DepTime >= 0 AND DepTime <= 2400))
AND (ArrTime IS NULL OR (ArrTime >= 0 AND ArrTime <= 2400))
AND (Cancelled IS NULL OR Cancelled = 0)
AND (Diverted IS NULL OR Diverted = 0)
AND ArrDelay IS NOT NULL
GROUP BY column_name
HAVING missing_pct > 0 AND missing_pct < 2;

-- 4. Create clean table with all filters applied
CREATE OR REPLACE TABLE clean_1993 AS
SELECT 
    -- Select all columns except those that are 100% missing
    Year, Month, DayOfWeek, DayofMonth, DepTime, CRSDepTime, ArrTime, CRSArrTime,
    UniqueCarrier, FlightNum, TailNum, ActualElapsedTime, CRSElapsedTime,
    AirTime, ArrDelay, DepDelay, Origin, Dest, Distance, TaxiIn, TaxiOut,
    Cancelled, CancellationCode, Diverted, CarrierDelay, WeatherDelay,
    NASDelay, SecurityDelay, LateAircraftDelay
FROM raw_1993
WHERE (DepTime IS NULL OR (DepTime >= 0 AND DepTime <= 2400))
AND (ArrTime IS NULL OR (ArrTime >= 0 AND ArrTime <= 2400))
AND (CRSDepTime IS NULL OR (CRSDepTime >= 0 AND CRSDepTime <= 2400))
AND (CRSArrTime IS NULL OR (CRSArrTime >= 0 AND CRSArrTime <= 2400))
AND (Cancelled IS NULL OR Cancelled = 0)
AND (Diverted IS NULL OR Diverted = 0)
AND ArrDelay IS NOT NULL
AND Distance IS NOT NULL  -- Example: low missingness column
AND ArrDelay >= -80 AND ArrDelay <= 150;  -- Extreme ArrDelay filter

-- 5. Extract extreme ArrDelay values to separate table
CREATE OR REPLACE TABLE extremes_1993 AS
SELECT *
FROM raw_1993
WHERE ArrDelay IS NOT NULL
AND (ArrDelay < -80 OR ArrDelay > 150);

-- Count extremes
SELECT 
    COUNT(*) as total_extremes,
    SUM(CASE WHEN ArrDelay < -80 THEN 1 ELSE 0 END) as too_negative,
    SUM(CASE WHEN ArrDelay > 150 THEN 1 ELSE 0 END) as too_positive
FROM extremes_1993;

-- 6. Create common-columns version (excludes delay breakdown columns)
CREATE OR REPLACE TABLE clean_common_1993 AS
SELECT 
    Year, Month, DayOfWeek, DayofMonth, DepTime, CRSDepTime, ArrTime, CRSArrTime,
    UniqueCarrier, FlightNum, ActualElapsedTime, CRSElapsedTime,
    ArrDelay, DepDelay, Origin, Dest, Distance,
    Cancelled, Diverted
    -- Excludes: TailNum, AirTime, TaxiIn, TaxiOut, CancellationCode,
    --          CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay,
    --          has_delay_breakdown
FROM clean_1993;

-- 7. Add has_delay_breakdown flag for 2003 (if delay columns available)
-- This is only for 2003, not 1993
ALTER TABLE clean_2003 ADD COLUMN has_delay_breakdown INTEGER;

UPDATE clean_2003
SET has_delay_breakdown = CASE
    WHEN (CarrierDelay IS NOT NULL 
          OR WeatherDelay IS NOT NULL 
          OR NASDelay IS NOT NULL 
          OR SecurityDelay IS NOT NULL 
          OR LateAircraftDelay IS NOT NULL) THEN 1
    ELSE 0
END;

-- 8. Calculate KPIs for raw vs clean comparison
-- Raw KPIs
SELECT 
    COUNT(*) as total_flights,
    AVG(ArrDelay) as avg_arrdelay,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ArrDelay) as median_arrdelay,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime15_pct
FROM raw_1993
WHERE ArrDelay IS NOT NULL;

-- Clean KPIs
SELECT 
    COUNT(*) as total_flights,
    AVG(ArrDelay) as avg_arrdelay,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ArrDelay) as median_arrdelay,
    SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime15_pct
FROM clean_1993;

-- 9. Write cleaned Parquet files (partitioned by year and month)
-- Note: This is done via Python's write_partitioned_parquet() function
COPY (
    SELECT * FROM clean_common_1993
) TO 'parquet/clean/common/year=1993/month=01/part-000.parquet' 
(FORMAT PARQUET, PARTITION_BY (year, month));
