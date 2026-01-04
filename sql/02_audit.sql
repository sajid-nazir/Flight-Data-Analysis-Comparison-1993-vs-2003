-- Stage 02: Audit missingness, validity, and comparability

-- 1. Missingness calculation by column
SELECT 
    column_name,
    COUNT(*) as total_count,
    SUM(CASE WHEN column_name IS NULL THEN 1 ELSE 0 END) as missing_count,
    ROUND(SUM(CASE WHEN column_name IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as missing_pct
FROM raw_1993
GROUP BY column_name;

-- For each column individually:
SELECT 
    'DepTime' as column_name,
    COUNT(*) as total_count,
    SUM(CASE WHEN DepTime IS NULL THEN 1 ELSE 0 END) as missing_count,
    ROUND(SUM(CASE WHEN DepTime IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as missing_pct
FROM raw_1993;

-- 2. Range check: Invalid time values
SELECT COUNT(*) as invalid_count
FROM raw_1993
WHERE DepTime IS NOT NULL AND (DepTime < 0 OR DepTime > 2400);

SELECT COUNT(*) as invalid_count
FROM raw_1993
WHERE ArrTime IS NOT NULL AND (ArrTime < 0 OR ArrTime > 2400);

SELECT COUNT(*) as invalid_count
FROM raw_1993
WHERE CRSDepTime IS NOT NULL AND (CRSDepTime < 0 OR CRSDepTime > 2400);

SELECT COUNT(*) as invalid_count
FROM raw_1993
WHERE CRSArrTime IS NOT NULL AND (CRSArrTime < 0 OR CRSArrTime > 2400);

-- 3. Range check: Invalid distance
SELECT COUNT(*) as invalid_count
FROM raw_1993
WHERE Distance IS NOT NULL AND Distance <= 0;

-- 4. Range check: Invalid month
SELECT COUNT(*) as invalid_count
FROM raw_1993
WHERE Month IS NOT NULL AND (Month < 1 OR Month > 12);

-- 5. Range check: Invalid day of month
SELECT COUNT(*) as invalid_count
FROM raw_1993
WHERE DayofMonth IS NOT NULL AND (DayofMonth < 1 OR DayofMonth > 31);

-- 6. Range check: Invalid day of week
SELECT COUNT(*) as invalid_count
FROM raw_1993
WHERE DayOfWeek IS NOT NULL AND (DayOfWeek < 1 OR DayOfWeek > 7);

-- 7. Range check: Extreme delays (> 24 hours = 1440 minutes)
SELECT COUNT(*) as invalid_count
FROM raw_1993
WHERE ArrDelay IS NOT NULL AND ABS(ArrDelay) > 1440;

SELECT COUNT(*) as invalid_count
FROM raw_1993
WHERE DepDelay IS NOT NULL AND ABS(DepDelay) > 1440;

-- 8. Cancellation rate
SELECT 
    COUNT(*) as total_flights,
    SUM(Cancelled) as cancelled_count,
    ROUND(SUM(Cancelled) * 100.0 / COUNT(*), 2) as cancel_rate_pct
FROM raw_1993;

-- 9. Diversion rate
SELECT 
    COUNT(*) as total_flights,
    SUM(Diverted) as diverted_count,
    ROUND(SUM(Diverted) * 100.0 / COUNT(*), 2) as divert_rate_pct
FROM raw_1993;

-- 10. Cancellation codes distribution
SELECT 
    CancellationCode, 
    COUNT(*) as cnt
FROM raw_1993
WHERE Cancelled = 1 AND CancellationCode IS NOT NULL
GROUP BY CancellationCode
ORDER BY cnt DESC;

-- 11. Feature availability matrix (comparing 1993 vs 2003)
-- Get columns from 1993
SELECT column_name FROM (
    DESCRIBE raw_1993
);

-- Get columns from 2003
SELECT column_name FROM (
    DESCRIBE raw_2003
);

-- Check which columns exist in both
WITH cols_1993 AS (
    SELECT column_name FROM (DESCRIBE raw_1993)
),
cols_2003 AS (
    SELECT column_name FROM (DESCRIBE raw_2003)
)
SELECT 
    COALESCE(c1.column_name, c2.column_name) as column_name,
    CASE WHEN c1.column_name IS NOT NULL THEN 1 ELSE 0 END as in_1993,
    CASE WHEN c2.column_name IS NOT NULL THEN 1 ELSE 0 END as in_2003,
    CASE WHEN c1.column_name IS NOT NULL AND c2.column_name IS NOT NULL THEN 1 ELSE 0 END as in_both
FROM cols_1993 c1
FULL OUTER JOIN cols_2003 c2 ON c1.column_name = c2.column_name
ORDER BY column_name;

-- 12. ArrDelay distribution statistics
SELECT 
    MIN(ArrDelay) as min_delay,
    MAX(ArrDelay) as max_delay,
    AVG(ArrDelay) as mean_delay,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ArrDelay) as median_delay,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY ArrDelay) as p25_delay,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY ArrDelay) as p75_delay
FROM raw_1993
WHERE ArrDelay IS NOT NULL;
