"""
Exploratory Data Analysis utilities
"""
import pandas as pd
import duckdb
from typing import Dict, List, Tuple
from pathlib import Path


def compute_core_kpis(
    conn: duckdb.DuckDBPyConnection,
    parquet_path_1993: str,
    parquet_path_2003: str,
    on_time_threshold: int = 15
) -> pd.DataFrame:
    """
    Compute core KPIs for both years.
    
    Args:
        conn: DuckDB connection
        parquet_path_1993: Path pattern to 1993 Parquet files
        parquet_path_2003: Path pattern to 2003 Parquet files
        on_time_threshold: Minutes threshold for on-time (default 15)
        
    Returns:
        DataFrame with core KPIs by year
    """
    query = f"""
    WITH all_data AS (
        SELECT Year, ArrDelay, DepDelay, Distance
        FROM read_parquet('{parquet_path_1993}')
        UNION ALL
        SELECT Year, ArrDelay, DepDelay, Distance
        FROM read_parquet('{parquet_path_2003}')
    )
    SELECT 
        Year,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= {on_time_threshold} THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
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
    ORDER BY Year
    """
    
    result = conn.execute(query).fetchdf()
    return result


def compute_ontime_by_month(
    conn: duckdb.DuckDBPyConnection,
    parquet_path_1993: str,
    parquet_path_2003: str,
    on_time_threshold: int = 15
) -> pd.DataFrame:
    """
    Compute on-time rates by month for both years.
    
    Args:
        conn: DuckDB connection
        parquet_path_1993: Path pattern to 1993 Parquet files
        parquet_path_2003: Path pattern to 2003 Parquet files
        on_time_threshold: Minutes threshold for on-time (default 15)
        
    Returns:
        DataFrame with on-time rates by year and month
    """
    query = f"""
    WITH all_data AS (
        SELECT Year, Month, ArrDelay
        FROM read_parquet('{parquet_path_1993}')
        UNION ALL
        SELECT Year, Month, ArrDelay
        FROM read_parquet('{parquet_path_2003}')
    )
    SELECT 
        Year,
        Month,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= {on_time_threshold} THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
        AVG(ArrDelay) as mean_arr_delay,
        MEDIAN(ArrDelay) as median_arr_delay
    FROM all_data
    GROUP BY Year, Month
    ORDER BY Year, Month
    """
    
    result = conn.execute(query).fetchdf()
    return result


def compute_ontime_by_dow(
    conn: duckdb.DuckDBPyConnection,
    parquet_path_1993: str,
    parquet_path_2003: str,
    on_time_threshold: int = 15
) -> pd.DataFrame:
    """
    Compute on-time rates by day of week for both years.
    
    Args:
        conn: DuckDB connection
        parquet_path_1993: Path pattern to 1993 Parquet files
        parquet_path_2003: Path pattern to 2003 Parquet files
        on_time_threshold: Minutes threshold for on-time (default 15)
        
    Returns:
        DataFrame with on-time rates by year and day of week
    """
    query = f"""
    WITH all_data AS (
        SELECT Year, DayOfWeek, ArrDelay
        FROM read_parquet('{parquet_path_1993}')
        UNION ALL
        SELECT Year, DayOfWeek, ArrDelay
        FROM read_parquet('{parquet_path_2003}')
    )
    SELECT 
        Year,
        DayOfWeek,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= {on_time_threshold} THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
        AVG(ArrDelay) as mean_arr_delay,
        MEDIAN(ArrDelay) as median_arr_delay
    FROM all_data
    GROUP BY Year, DayOfWeek
    ORDER BY Year, DayOfWeek
    """
    
    result = conn.execute(query).fetchdf()
    return result


def compute_ontime_by_dep_hour(
    conn: duckdb.DuckDBPyConnection,
    parquet_path_1993: str,
    parquet_path_2003: str,
    on_time_threshold: int = 15
) -> pd.DataFrame:
    """
    Compute on-time rates by departure hour for both years.
    
    Args:
        conn: DuckDB connection
        parquet_path_1993: Path pattern to 1993 Parquet files
        parquet_path_2003: Path pattern to 2003 Parquet files
        on_time_threshold: Minutes threshold for on-time (default 15)
        
    Returns:
        DataFrame with on-time rates by year and departure hour
    """
    query = f"""
    WITH all_data AS (
        SELECT Year, DepTime, ArrDelay
        FROM read_parquet('{parquet_path_1993}')
        WHERE DepTime IS NOT NULL
        UNION ALL
        SELECT Year, DepTime, ArrDelay
        FROM read_parquet('{parquet_path_2003}')
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
        SUM(CASE WHEN ArrDelay <= {on_time_threshold} THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
        AVG(ArrDelay) as mean_arr_delay,
        MEDIAN(ArrDelay) as median_arr_delay
    FROM with_hour
    GROUP BY Year, dep_hour
    ORDER BY Year, dep_hour
    """
    
    result = conn.execute(query).fetchdf()
    return result


def compute_carrier_summary(
    conn: duckdb.DuckDBPyConnection,
    parquet_path_1993: str,
    parquet_path_2003: str,
    on_time_threshold: int = 15
) -> pd.DataFrame:
    """
    Compute carrier summary statistics for both years.
    
    Args:
        conn: DuckDB connection
        parquet_path_1993: Path pattern to 1993 Parquet files
        parquet_path_2003: Path pattern to 2003 Parquet files
        on_time_threshold: Minutes threshold for on-time (default 15)
        
    Returns:
        DataFrame with carrier summaries by year
    """
    query = f"""
    WITH all_data AS (
        SELECT Year, UniqueCarrier, ArrDelay, Distance
        FROM read_parquet('{parquet_path_1993}')
        UNION ALL
        SELECT Year, UniqueCarrier, ArrDelay, Distance
        FROM read_parquet('{parquet_path_2003}')
    )
    SELECT 
        Year,
        UniqueCarrier,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= {on_time_threshold} THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
        AVG(ArrDelay) as mean_arr_delay,
        MEDIAN(ArrDelay) as median_arr_delay,
        AVG(Distance) as mean_distance
    FROM all_data
    GROUP BY Year, UniqueCarrier
    ORDER BY Year, total_flights DESC
    """
    
    result = conn.execute(query).fetchdf()
    return result


def compute_airport_summary(
    conn: duckdb.DuckDBPyConnection,
    parquet_path_1993: str,
    parquet_path_2003: str,
    role: str = 'origin',
    on_time_threshold: int = 15
) -> pd.DataFrame:
    """
    Compute airport summary statistics for both years.
    
    Args:
        conn: DuckDB connection
        parquet_path_1993: Path pattern to 1993 Parquet files
        parquet_path_2003: Path pattern to 2003 Parquet files
        role: 'origin' or 'dest' (default 'origin')
        on_time_threshold: Minutes threshold for on-time (default 15)
        
    Returns:
        DataFrame with airport summaries by year
    """
    airport_col = 'Origin' if role == 'origin' else 'Dest'
    other_col = 'Dest' if role == 'origin' else 'Origin'
    
    query = f"""
    WITH all_data AS (
        SELECT Year, {airport_col} as airport, {other_col} as other_airport, ArrDelay
        FROM read_parquet('{parquet_path_1993}')
        UNION ALL
        SELECT Year, {airport_col} as airport, {other_col} as other_airport, ArrDelay
        FROM read_parquet('{parquet_path_2003}')
    )
    SELECT 
        Year,
        airport,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= {on_time_threshold} THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
        AVG(ArrDelay) as mean_arr_delay,
        MEDIAN(ArrDelay) as median_arr_delay,
        COUNT(DISTINCT other_airport) as unique_{other_col.lower()}_count
    FROM all_data
    GROUP BY Year, airport
    ORDER BY Year, total_flights DESC
    """
    
    result = conn.execute(query).fetchdf()
    return result


def compute_route_summary(
    conn: duckdb.DuckDBPyConnection,
    parquet_path_1993: str,
    parquet_path_2003: str,
    on_time_threshold: int = 15
) -> pd.DataFrame:
    """
    Compute route summary statistics for both years.
    
    Args:
        conn: DuckDB connection
        parquet_path_1993: Path pattern to 1993 Parquet files
        parquet_path_2003: Path pattern to 2003 Parquet files
        on_time_threshold: Minutes threshold for on-time (default 15)
        
    Returns:
        DataFrame with route summaries by year
    """
    query = f"""
    WITH all_data AS (
        SELECT Year, Origin, Dest, ArrDelay, Distance
        FROM read_parquet('{parquet_path_1993}')
        UNION ALL
        SELECT Year, Origin, Dest, ArrDelay, Distance
        FROM read_parquet('{parquet_path_2003}')
    )
    SELECT 
        Year,
        Origin,
        Dest,
        CONCAT(Origin, '-', Dest) as route,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= {on_time_threshold} THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
        AVG(ArrDelay) as mean_arr_delay,
        MEDIAN(ArrDelay) as median_arr_delay,
        AVG(Distance) as mean_distance,
        MEDIAN(Distance) as median_distance
    FROM all_data
    GROUP BY Year, Origin, Dest, route
    ORDER BY Year, total_flights DESC
    """
    
    result = conn.execute(query).fetchdf()
    return result


def compute_route_matched_summary(
    conn: duckdb.DuckDBPyConnection,
    parquet_path_1993: str,
    parquet_path_2003: str,
    on_time_threshold: int = 15
) -> pd.DataFrame:
    """
    Compute route summary for routes that exist in both years (matched analysis).
    
    Args:
        conn: DuckDB connection
        parquet_path_1993: Path pattern to 1993 Parquet files
        parquet_path_2003: Path pattern to 2003 Parquet files
        on_time_threshold: Minutes threshold for on-time (default 15)
        
    Returns:
        DataFrame with route summaries for common routes only
    """
    query = f"""
    WITH routes_1993 AS (
        SELECT DISTINCT CONCAT(Origin, '-', Dest) as route
        FROM read_parquet('{parquet_path_1993}')
    ),
    routes_2003 AS (
        SELECT DISTINCT CONCAT(Origin, '-', Dest) as route
        FROM read_parquet('{parquet_path_2003}')
    ),
    common_routes AS (
        SELECT route FROM routes_1993
        INTERSECT
        SELECT route FROM routes_2003
    ),
    all_data AS (
        SELECT Year, Origin, Dest, ArrDelay
        FROM read_parquet('{parquet_path_1993}')
        UNION ALL
        SELECT Year, Origin, Dest, ArrDelay
        FROM read_parquet('{parquet_path_2003}')
    )
    SELECT 
        Year,
        CONCAT(Origin, '-', Dest) as route,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= {on_time_threshold} THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
        AVG(ArrDelay) as mean_arr_delay,
        MEDIAN(ArrDelay) as median_arr_delay
    FROM all_data
    WHERE CONCAT(Origin, '-', Dest) IN (SELECT route FROM common_routes)
    GROUP BY Year, route
    ORDER BY Year, total_flights DESC
    """
    
    result = conn.execute(query).fetchdf()
    return result


def compute_airport_matched_summary(
    conn: duckdb.DuckDBPyConnection,
    parquet_path_1993: str,
    parquet_path_2003: str,
    on_time_threshold: int = 15
) -> pd.DataFrame:
    """
    Compute airport summary for airports that exist in both years (matched analysis).
    Includes both origin and destination roles.
    
    Args:
        conn: DuckDB connection
        parquet_path_1993: Path pattern to 1993 Parquet files
        parquet_path_2003: Path pattern to 2003 Parquet files
        on_time_threshold: Minutes threshold for on-time (default 15)
        
    Returns:
        DataFrame with airport summaries for common airports only (both origin and dest)
    """
    query = f"""
    WITH airports_1993 AS (
        SELECT DISTINCT Origin as airport FROM read_parquet('{parquet_path_1993}')
        UNION
        SELECT DISTINCT Dest as airport FROM read_parquet('{parquet_path_1993}')
    ),
    airports_2003 AS (
        SELECT DISTINCT Origin as airport FROM read_parquet('{parquet_path_2003}')
        UNION
        SELECT DISTINCT Dest as airport FROM read_parquet('{parquet_path_2003}')
    ),
    common_airports AS (
        SELECT airport FROM airports_1993
        INTERSECT
        SELECT airport FROM airports_2003
    ),
    origin_data AS (
        SELECT Year, Origin as airport, 'origin' as airport_role, ArrDelay
        FROM read_parquet('{parquet_path_1993}')
        WHERE Origin IN (SELECT airport FROM common_airports)
        UNION ALL
        SELECT Year, Origin as airport, 'origin' as airport_role, ArrDelay
        FROM read_parquet('{parquet_path_2003}')
        WHERE Origin IN (SELECT airport FROM common_airports)
    ),
    dest_data AS (
        SELECT Year, Dest as airport, 'destination' as airport_role, ArrDelay
        FROM read_parquet('{parquet_path_1993}')
        WHERE Dest IN (SELECT airport FROM common_airports)
        UNION ALL
        SELECT Year, Dest as airport, 'destination' as airport_role, ArrDelay
        FROM read_parquet('{parquet_path_2003}')
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
        SUM(CASE WHEN ArrDelay <= {on_time_threshold} THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
        AVG(ArrDelay) as mean_arr_delay,
        MEDIAN(ArrDelay) as median_arr_delay
    FROM combined_data
    GROUP BY Year, airport, airport_role
    ORDER BY Year, airport_role, total_flights DESC
    """
    
    result = conn.execute(query).fetchdf()
    return result
