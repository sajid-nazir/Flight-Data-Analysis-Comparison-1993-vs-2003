"""
Data cleaning utilities
"""
import pandas as pd
import duckdb
from typing import Dict, List, Tuple


def get_cleaning_stats(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    year: int
) -> Dict[str, int]:
    """
    Get statistics for cleaning steps.
    
    Args:
        conn: DuckDB connection
        table_name: Name of the table to analyze
        year: Year for labeling
        
    Returns:
        Dictionary with row counts at each cleaning step
    """
    stats = {}
    
    # Initial count
    stats['initial'] = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    
    # After removing invalid times
    stats['after_invalid_times'] = conn.execute(f"""
        SELECT COUNT(*) FROM {table_name}
        WHERE (DepTime IS NULL OR (DepTime >= 0 AND DepTime <= 2400))
        AND (ArrTime IS NULL OR (ArrTime >= 0 AND ArrTime <= 2400))
        AND (CRSDepTime IS NULL OR (CRSDepTime >= 0 AND CRSDepTime <= 2400))
        AND (CRSArrTime IS NULL OR (CRSArrTime >= 0 AND CRSArrTime <= 2400))
    """).fetchone()[0]
    
    # After removing cancelled
    stats['after_cancelled'] = conn.execute(f"""
        SELECT COUNT(*) FROM {table_name}
        WHERE (DepTime IS NULL OR (DepTime >= 0 AND DepTime <= 2400))
        AND (ArrTime IS NULL OR (ArrTime >= 0 AND ArrTime <= 2400))
        AND (CRSDepTime IS NULL OR (CRSDepTime >= 0 AND CRSDepTime <= 2400))
        AND (CRSArrTime IS NULL OR (CRSArrTime >= 0 AND CRSArrTime <= 2400))
        AND (Cancelled IS NULL OR Cancelled = 0)
    """).fetchone()[0]
    
    # After removing diverted
    stats['after_diverted'] = conn.execute(f"""
        SELECT COUNT(*) FROM {table_name}
        WHERE (DepTime IS NULL OR (DepTime >= 0 AND DepTime <= 2400))
        AND (ArrTime IS NULL OR (ArrTime >= 0 AND ArrTime <= 2400))
        AND (CRSDepTime IS NULL OR (CRSDepTime >= 0 AND CRSDepTime <= 2400))
        AND (CRSArrTime IS NULL OR (CRSArrTime >= 0 AND CRSArrTime <= 2400))
        AND (Cancelled IS NULL OR Cancelled = 0)
        AND (Diverted IS NULL OR Diverted = 0)
    """).fetchone()[0]
    
    # After removing missing ArrDelay
    stats['after_missing_arrdelay'] = conn.execute(f"""
        SELECT COUNT(*) FROM {table_name}
        WHERE (DepTime IS NULL OR (DepTime >= 0 AND DepTime <= 2400))
        AND (ArrTime IS NULL OR (ArrTime >= 0 AND ArrTime <= 2400))
        AND (CRSDepTime IS NULL OR (CRSDepTime >= 0 AND CRSDepTime <= 2400))
        AND (CRSArrTime IS NULL OR (CRSArrTime >= 0 AND CRSArrTime <= 2400))
        AND (Cancelled IS NULL OR Cancelled = 0)
        AND (Diverted IS NULL OR Diverted = 0)
        AND ArrDelay IS NOT NULL
    """).fetchone()[0]
    
    return stats


def apply_winsorization(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    column: str,
    q_low: float,
    q_high: float
) -> Tuple[float, float]:
    """
    Calculate winsorization bounds for a column.
    
    Args:
        conn: DuckDB connection
        table_name: Name of the table
        column: Column name to winsorize
        q_low: Lower quantile (e.g., 0.005)
        q_high: Upper quantile (e.g., 0.995)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    result = conn.execute(f"""
        SELECT 
            PERCENTILE_CONT({q_low}) WITHIN GROUP (ORDER BY {column}) as lower_bound,
            PERCENTILE_CONT({q_high}) WITHIN GROUP (ORDER BY {column}) as upper_bound
        FROM {table_name}
        WHERE {column} IS NOT NULL
    """).fetchone()
    
    return (result[0] if result[0] is not None else float('-inf'),
            result[1] if result[1] is not None else float('inf'))


def get_columns_to_drop(conn: duckdb.DuckDBPyConnection, table_name: str) -> List[str]:
    """
    Get list of columns that are 100% missing and should be dropped.
    
    Args:
        conn: DuckDB connection
        table_name: Name of the table to check
        
    Returns:
        List of column names to drop
    """
    # Get all columns
    schema_info = conn.execute(f"DESCRIBE {table_name}").fetchall()
    all_columns = [col[0] for col in schema_info]
    
    columns_to_drop = []
    
    for col in all_columns:
        try:
            # Check if column is 100% missing
            result = conn.execute(f"""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) as missing
                FROM {table_name}
            """).fetchone()
            
            total = result[0]
            missing = result[1]
            
            if total > 0 and missing == total:
                columns_to_drop.append(col)
        except:
            # Column might not exist or have issues, skip it
            pass
    
    return columns_to_drop


def get_low_missingness_columns(conn: duckdb.DuckDBPyConnection, table_name: str, threshold: float = 0.02) -> List[str]:
    """
    Get list of columns with less than threshold missingness (e.g., <2%).
    These columns should have no NULL values in cleaned data.
    
    Args:
        conn: DuckDB connection
        table_name: Name of the table to check
        threshold: Missingness threshold (default 0.02 = 2%)
        
    Returns:
        List of column names with <threshold missingness
    """
    # Get all columns
    schema_info = conn.execute(f"DESCRIBE {table_name}").fetchall()
    all_columns = [col[0] for col in schema_info]
    
    total_rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    if total_rows == 0:
        return []
    
    low_missingness_cols = []
    
    for col in all_columns:
        try:
            # Check missingness percentage
            result = conn.execute(f"""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) as missing
                FROM {table_name}
            """).fetchone()
            
            total = result[0]
            missing = result[1]
            missing_pct = missing / total if total > 0 else 1.0
            
            if 0 < missing_pct < threshold:  # Between 0% and threshold (e.g., 2%)
                low_missingness_cols.append(col)
        except:
            # Column might not exist or have issues, skip it
            pass
    
    return low_missingness_cols


def create_clean_table(
    conn: duckdb.DuckDBPyConnection,
    source_table: str,
    target_table: str,
    year: int,
    winsorize: bool = False,
    winsor_q_low: float = 0.005,
    winsor_q_high: float = 0.995,
    drop_100pct_missing: bool = True,
    drop_low_missingness_rows: bool = True,
    low_missingness_threshold: float = 0.02,
    arrdelay_min: float = -80,
    arrdelay_max: float = 150,
    apply_arrdelay_filter: bool = True
) -> Dict[str, int]:
    """
    Create a cleaned table with all filters applied.
    
    Args:
        conn: DuckDB connection
        source_table: Source table name
        target_table: Target table name
        year: Year for labeling
        winsorize: Whether to apply winsorization
        winsor_q_low: Lower quantile for winsorization
        winsor_q_high: Upper quantile for winsorization
        drop_100pct_missing: Whether to drop columns that are 100% missing
        drop_low_missingness_rows: Whether to drop rows with missing values in low-missingness columns
        low_missingness_threshold: Threshold for low missingness (default 0.02 = 2%)
        arrdelay_min: Minimum ArrDelay value to keep (default -80)
        arrdelay_max: Maximum ArrDelay value to keep (default 150)
        apply_arrdelay_filter: Whether to filter ArrDelay to [arrdelay_min, arrdelay_max] range
        
    Returns:
        Dictionary with cleaning statistics
    """
    # Get cleaning stats
    stats = get_cleaning_stats(conn, source_table, year)
    
    # Get columns to drop (100% missing)
    columns_to_drop = []
    if drop_100pct_missing:
        columns_to_drop = get_columns_to_drop(conn, source_table)
        if columns_to_drop:
            print(f"    Dropping {len(columns_to_drop)} columns that are 100% missing: {', '.join(columns_to_drop)}")
    
    # Get all columns except those to drop
    schema_info = conn.execute(f"DESCRIBE {source_table}").fetchall()
    all_columns = [col[0] for col in schema_info if col[0] not in columns_to_drop]
    
    # Get columns with low missingness (<2%) - we'll drop rows where these are NULL
    low_missingness_cols = []
    if drop_low_missingness_rows:
        # Check missingness on the filtered dataset (after basic filters)
        temp_table = f"temp_filtered_{year}"
        
        # Apply basic filters first
        basic_where = [
            "(DepTime IS NULL OR (DepTime >= 0 AND DepTime <= 2400))",
            "(ArrTime IS NULL OR (ArrTime >= 0 AND ArrTime <= 2400))",
            "(CRSDepTime IS NULL OR (CRSDepTime >= 0 AND CRSDepTime <= 2400))",
            "(CRSArrTime IS NULL OR (CRSArrTime >= 0 AND CRSArrTime <= 2400))",
            "(Cancelled IS NULL OR Cancelled = 0)",
            "(Diverted IS NULL OR Diverted = 0)",
            "ArrDelay IS NOT NULL"
        ]
        basic_where_clause = " AND ".join(basic_where)
        
        select_cols = ", ".join(all_columns)
        conn.execute(f"""
            CREATE OR REPLACE TABLE {temp_table} AS
            SELECT {select_cols}
            FROM {source_table}
            WHERE {basic_where_clause}
        """)
        
        low_missingness_cols = get_low_missingness_columns(conn, temp_table, low_missingness_threshold)
        if low_missingness_cols:
            print(f"    Dropping rows with missing values in low-missingness columns (<{low_missingness_threshold*100}%): {', '.join(low_missingness_cols)}")
        
        # Clean up temp table
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
    
    select_columns = ", ".join(all_columns)
    
    # Build the WHERE clause for filtering
    where_clauses = [
        "(DepTime IS NULL OR (DepTime >= 0 AND DepTime <= 2400))",
        "(ArrTime IS NULL OR (ArrTime >= 0 AND ArrTime <= 2400))",
        "(CRSDepTime IS NULL OR (CRSDepTime >= 0 AND CRSDepTime <= 2400))",
        "(CRSArrTime IS NULL OR (CRSArrTime >= 0 AND CRSArrTime <= 2400))",
        "(Cancelled IS NULL OR Cancelled = 0)",
        "(Diverted IS NULL OR Diverted = 0)",
        "ArrDelay IS NOT NULL"
    ]
    
    # Add ArrDelay range filter if requested
    if apply_arrdelay_filter:
        where_clauses.append(f"ArrDelay >= {arrdelay_min} AND ArrDelay <= {arrdelay_max}")
    
    # Add conditions to drop rows with missing values in low-missingness columns
    for col in low_missingness_cols:
        if col in all_columns:  # Make sure column wasn't dropped
            where_clauses.append(f"{col} IS NOT NULL")
    
    where_clause = " AND ".join(where_clauses)
    
    # Create cleaned table
    conn.execute(f"""
        CREATE OR REPLACE TABLE {target_table} AS
        SELECT {select_columns}
        FROM {source_table}
        WHERE {where_clause}
    """)
    
    # Add has_delay_breakdown flag for 2003 (delay columns available starting Q3)
    if year == 2003:
        # Check if delay columns exist
        delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
        existing_delay_cols = [col for col in delay_cols if col in all_columns]
        
        if existing_delay_cols:
            # Check if column already exists in target table
            schema_info = conn.execute(f"DESCRIBE {target_table}").fetchall()
            target_columns = [col[0] for col in schema_info]
            
            delay_check = " OR ".join([f"{col} IS NOT NULL" for col in existing_delay_cols])
            
            if "has_delay_breakdown" not in target_columns:
                # Create flag: 1 if any delay column is not NULL, 0 otherwise
                # This indicates delay breakdown data is available
                conn.execute(f"""
                    ALTER TABLE {target_table} ADD COLUMN has_delay_breakdown INTEGER;
                """)
                
                conn.execute(f"""
                    UPDATE {target_table}
                    SET has_delay_breakdown = CASE
                        WHEN ({delay_check}) THEN 1
                        ELSE 0
                    END
                """)
                
                print(f"    Added 'has_delay_breakdown' feature flag")
            else:
                # Column already exists, just update it
                conn.execute(f"""
                    UPDATE {target_table}
                    SET has_delay_breakdown = CASE
                        WHEN ({delay_check}) THEN 1
                        ELSE 0
                    END
                """)
            
            conn.execute(f"""
                UPDATE {target_table}
                SET has_delay_breakdown = CASE
                    WHEN ({delay_check}) THEN 1
                    ELSE 0
                END
            """)
            
            print(f"    Added 'has_delay_breakdown' feature flag")
    
    return stats


def create_common_columns_table(
    conn: duckdb.DuckDBPyConnection,
    source_table: str,
    target_table: str,
    year: int,
    reference_table: str = None,
    delay_breakdown_cols: List[str] = None
) -> None:
    """
    Create a table with only common columns (columns that exist in both years).
    This ensures fair comparison between 1993 and 2003.
    
    Args:
        conn: DuckDB connection
        source_table: Source table name (should be clean table)
        target_table: Target table name
        year: Year for labeling
        reference_table: Reference table to get common columns from (e.g., clean_common_1993)
        delay_breakdown_cols: List of delay breakdown columns to exclude
    """
    if delay_breakdown_cols is None:
        delay_breakdown_cols = [
            'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 
            'LateAircraftDelay', 'CancellationCode', 'has_delay_breakdown'
        ]
    
    # Get all columns from source table
    schema_info = conn.execute(f"DESCRIBE {source_table}").fetchall()
    all_columns = [col[0] for col in schema_info]
    
    # If reference table is provided, use its columns as the common set
    if reference_table:
        try:
            ref_schema_info = conn.execute(f"DESCRIBE {reference_table}").fetchall()
            reference_columns = set([col[0] for col in ref_schema_info])
            # Only keep columns that exist in both source and reference
            common_columns = [col for col in all_columns 
                            if col in reference_columns and col not in delay_breakdown_cols]
        except:
            # If reference table doesn't exist yet, fall back to excluding delay breakdown only
            common_columns = [col for col in all_columns if col not in delay_breakdown_cols]
    else:
        # Filter out delay breakdown columns only
        common_columns = [col for col in all_columns if col not in delay_breakdown_cols]
    
    # Also exclude columns that were 100% missing in 1993 (for 2003)
    columns_not_in_1993 = ['TailNum', 'AirTime', 'TaxiIn', 'TaxiOut']
    common_columns = [col for col in common_columns if col not in columns_not_in_1993]
    
    select_columns = ", ".join(common_columns)
    
    # Create common columns table
    conn.execute(f"""
        CREATE OR REPLACE TABLE {target_table} AS
        SELECT {select_columns}
        FROM {source_table}
    """)
    
    excluded_count = len(all_columns) - len(common_columns)
    print(f"    Created common-columns version: {len(common_columns)} columns (excluded {excluded_count} columns)")


def extract_extreme_arrdelay_values(
    conn: duckdb.DuckDBPyConnection,
    source_table: str,
    target_table: str,
    arrdelay_min: float = -80,
    arrdelay_max: float = 150
) -> int:
    """
    Extract rows with extreme ArrDelay values (outside the specified range) to a separate table.
    
    Args:
        conn: DuckDB connection
        source_table: Source table name (should be the cleaned table before ArrDelay filtering)
        target_table: Target table name for extreme values
        arrdelay_min: Minimum ArrDelay value (default -80)
        arrdelay_max: Maximum ArrDelay value (default 150)
        
    Returns:
        Number of rows with extreme ArrDelay values
    """
    # Get all columns from source table
    schema_info = conn.execute(f"DESCRIBE {source_table}").fetchall()
    all_columns = [col[0] for col in schema_info]
    select_columns = ", ".join(all_columns)
    
    # Create table with extreme ArrDelay values
    conn.execute(f"""
        CREATE OR REPLACE TABLE {target_table} AS
        SELECT {select_columns}
        FROM {source_table}
        WHERE ArrDelay IS NOT NULL
        AND (ArrDelay < {arrdelay_min} OR ArrDelay > {arrdelay_max})
    """)
    
    count = conn.execute(f"SELECT COUNT(*) FROM {target_table}").fetchone()[0]
    
    if count > 0:
        too_negative = conn.execute(f"""
            SELECT COUNT(*) FROM {target_table} WHERE ArrDelay < {arrdelay_min}
        """).fetchone()[0]
        too_positive = conn.execute(f"""
            SELECT COUNT(*) FROM {target_table} WHERE ArrDelay > {arrdelay_max}
        """).fetchone()[0]
        print(f"    Extracted {count:,} extreme ArrDelay rows to {target_table}:")
        print(f"      ArrDelay < {arrdelay_min}: {too_negative:,}")
        print(f"      ArrDelay > {arrdelay_max}: {too_positive:,}")
    
    return count
