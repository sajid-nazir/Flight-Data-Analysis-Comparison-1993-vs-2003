-- Stage 02: Audit missingness, validity, and comparability

-- Missingness calculation example
-- SELECT 
--     column_name,
--     COUNT(*) as total_count,
--     SUM(CASE WHEN column_name IS NULL THEN 1 ELSE 0 END) as missing_count,
--     ROUND(SUM(CASE WHEN column_name IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as missing_pct
-- FROM raw_1993
-- GROUP BY column_name;

-- Range check: Invalid time values
-- SELECT COUNT(*) as invalid_count
-- FROM raw_1993
-- WHERE DepTime IS NOT NULL AND (DepTime < 0 OR DepTime > 2400);

-- Range check: Invalid distance
-- SELECT COUNT(*) as invalid_count
-- FROM raw_1993
-- WHERE Distance IS NOT NULL AND Distance <= 0;

-- Cancellation rate
-- SELECT 
--     COUNT(*) as total_flights,
--     SUM(Cancelled) as cancelled_count,
--     ROUND(SUM(Cancelled) * 100.0 / COUNT(*), 2) as cancel_rate_pct
-- FROM raw_1993;

-- Diversion rate
-- SELECT 
--     COUNT(*) as total_flights,
--     SUM(Diverted) as diverted_count,
--     ROUND(SUM(Diverted) * 100.0 / COUNT(*), 2) as divert_rate_pct
-- FROM raw_1993;
