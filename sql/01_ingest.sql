-- Stage 01: Ingest CSV files into DuckDB and write raw Parquet

-- 1. Read CSV into DuckDB table
-- Note: This is done via Python's read_csv_to_duckdb() function which uses:
CREATE TABLE raw_1993 AS SELECT * FROM read_csv_auto('data_raw/1993.csv');
CREATE TABLE raw_2003 AS SELECT * FROM read_csv_auto('data_raw/2003.csv');

-- 2. Normalize column types (fix VARCHAR columns that should be BIGINT)
-- For columns that are 100% missing in 1993, they may be inferred as VARCHAR
-- but should be BIGINT to match 2003 types
CREATE OR REPLACE TABLE raw_1993_temp AS
SELECT 
    -- Cast numeric columns that are VARCHAR to BIGINT
    CAST(AirTime AS BIGINT) AS AirTime,
    CAST(TaxiIn AS BIGINT) AS TaxiIn,
    CAST(TaxiOut AS BIGINT) AS TaxiOut,
    CAST(CarrierDelay AS BIGINT) AS CarrierDelay,
    CAST(WeatherDelay AS BIGINT) AS WeatherDelay,
    CAST(NASDelay AS BIGINT) AS NASDelay,
    CAST(SecurityDelay AS BIGINT) AS SecurityDelay,
    CAST(LateAircraftDelay AS BIGINT) AS LateAircraftDelay,
    -- Keep all other columns as-is
    *
FROM raw_1993
WHERE AirTime IS NULL;  -- Only apply if column is 100% NULL

-- 3. Extract month from date column
-- Try extracting from date column first
ALTER TABLE raw_1993 ADD COLUMN month INTEGER;
UPDATE raw_1993 
SET month = EXTRACT(MONTH FROM CAST(FlightDate AS DATE))
WHERE FlightDate IS NOT NULL;

-- If date parsing fails, extract from string format
UPDATE raw_1993 
SET month = CAST(SUBSTRING(CAST(FlightDate AS VARCHAR), 5, 2) AS INTEGER)
WHERE month IS NULL 
AND FlightDate IS NOT NULL 
AND LENGTH(CAST(FlightDate AS VARCHAR)) >= 6;

-- Or use Month column directly if it exists
UPDATE raw_1993 
SET month = CAST(Month AS INTEGER)
WHERE month IS NULL AND Month IS NOT NULL;

-- 4. Write partitioned Parquet files
-- Note: This is done via Python's write_partitioned_parquet() function
-- which uses COPY command:
COPY (
    SELECT * FROM raw_1993
) TO 'parquet/raw/year=1993/month=01/part-000.parquet' (FORMAT PARQUET, PARTITION_BY (year, month));

-- 5. Get row counts by partition
SELECT 
    year,
    month,
    COUNT(*) as row_count
FROM raw_1993
GROUP BY year, month
ORDER BY year, month;

-- 6. Get table schema
DESCRIBE raw_1993;
DESCRIBE raw_2003;
