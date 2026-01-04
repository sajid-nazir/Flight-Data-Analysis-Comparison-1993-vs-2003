"""
Data auditing utilities
"""
import pandas as pd
import duckdb
from typing import Dict, List, Any


def calculate_missingness(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    year: int
) -> pd.DataFrame:
    """
    Calculate missingness percentage for each column.
    
    Args:
        conn: DuckDB connection
        table_name: Name of the table to analyze
        year: Year for labeling
        
    Returns:
        DataFrame with columns: column_name, year, missing_count, total_count, missing_pct
    """
    # Get all column names
    schema_info = conn.execute(f"DESCRIBE {table_name}").fetchall()
    column_names = [col[0] for col in schema_info]
    
    missingness_data = []
    total_rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    
    for col_name in column_names:
        # Count NULL values
        missing_count = conn.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE {col_name} IS NULL"
        ).fetchone()[0]
        
        missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0
        
        missingness_data.append({
            'column_name': col_name,
            'year': year,
            'missing_count': missing_count,
            'total_count': total_rows,
            'missing_pct': round(missing_pct, 2)
        })
    
    return pd.DataFrame(missingness_data)


def perform_range_checks(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    year: int
) -> pd.DataFrame:
    """
    Perform range checks on various columns.
    
    Args:
        conn: DuckDB connection
        table_name: Name of the table to analyze
        year: Year for labeling
        
    Returns:
        DataFrame with columns: check_name, column, year, invalid_count, total_count, invalid_pct
    """
    total_rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    range_checks = []
    
    # Time column checks (should be 0-2400 or NULL)
    time_columns = ['DepTime', 'ArrTime', 'CRSDepTime', 'CRSArrTime']
    for col in time_columns:
        try:
            invalid_count = conn.execute(
                f"""
                SELECT COUNT(*) FROM {table_name}
                WHERE {col} IS NOT NULL 
                AND ({col} < 0 OR {col} > 2400)
                """
            ).fetchone()[0]
            
            invalid_pct = (invalid_count / total_rows * 100) if total_rows > 0 else 0
            range_checks.append({
                'check_name': 'time_range',
                'column': col,
                'year': year,
                'invalid_count': invalid_count,
                'total_count': total_rows,
                'invalid_pct': round(invalid_pct, 2)
            })
        except:
            pass  # Column might not exist
    
    # Distance check (must be > 0)
    try:
        invalid_count = conn.execute(
            f"""
            SELECT COUNT(*) FROM {table_name}
            WHERE Distance IS NOT NULL AND Distance <= 0
            """
        ).fetchone()[0]
        
        invalid_pct = (invalid_count / total_rows * 100) if total_rows > 0 else 0
        range_checks.append({
            'check_name': 'distance_positive',
            'column': 'Distance',
            'year': year,
            'invalid_count': invalid_count,
            'total_count': total_rows,
            'invalid_pct': round(invalid_pct, 2)
        })
    except:
        pass
    
    # Month check (1-12)
    try:
        invalid_count = conn.execute(
            f"""
            SELECT COUNT(*) FROM {table_name}
            WHERE Month IS NOT NULL AND (Month < 1 OR Month > 12)
            """
        ).fetchone()[0]
        
        invalid_pct = (invalid_count / total_rows * 100) if total_rows > 0 else 0
        range_checks.append({
            'check_name': 'month_range',
            'column': 'Month',
            'year': year,
            'invalid_count': invalid_count,
            'total_count': total_rows,
            'invalid_pct': round(invalid_pct, 2)
        })
    except:
        pass
    
    # DayofMonth check (1-31)
    try:
        invalid_count = conn.execute(
            f"""
            SELECT COUNT(*) FROM {table_name}
            WHERE DayofMonth IS NOT NULL AND (DayofMonth < 1 OR DayofMonth > 31)
            """
        ).fetchone()[0]
        
        invalid_pct = (invalid_count / total_rows * 100) if total_rows > 0 else 0
        range_checks.append({
            'check_name': 'day_range',
            'column': 'DayofMonth',
            'year': year,
            'invalid_count': invalid_count,
            'total_count': total_rows,
            'invalid_pct': round(invalid_pct, 2)
        })
    except:
        pass
    
    # DayOfWeek check (1-7)
    try:
        invalid_count = conn.execute(
            f"""
            SELECT COUNT(*) FROM {table_name}
            WHERE DayOfWeek IS NOT NULL AND (DayOfWeek < 1 OR DayOfWeek > 7)
            """
        ).fetchone()[0]
        
        invalid_pct = (invalid_count / total_rows * 100) if total_rows > 0 else 0
        range_checks.append({
            'check_name': 'dow_range',
            'column': 'DayOfWeek',
            'year': year,
            'invalid_count': invalid_count,
            'total_count': total_rows,
            'invalid_pct': round(invalid_pct, 2)
        })
    except:
        pass
    
    # Extreme delay check (> 24 hours = 1440 minutes)
    delay_columns = ['ArrDelay', 'DepDelay']
    for col in delay_columns:
        try:
            invalid_count = conn.execute(
                f"""
                SELECT COUNT(*) FROM {table_name}
                WHERE {col} IS NOT NULL AND ABS({col}) > 1440
                """
            ).fetchone()[0]
            
            invalid_pct = (invalid_count / total_rows * 100) if total_rows > 0 else 0
            range_checks.append({
                'check_name': 'extreme_delay',
                'column': col,
                'year': year,
                'invalid_count': invalid_count,
                'total_count': total_rows,
                'invalid_pct': round(invalid_pct, 2)
            })
        except:
            pass
    
    return pd.DataFrame(range_checks)


def calculate_cancel_divert_rates(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    year: int
) -> Dict[str, Any]:
    """
    Calculate cancellation and diversion rates.
    
    Args:
        conn: DuckDB connection
        table_name: Name of the table to analyze
        year: Year for labeling
        
    Returns:
        Dictionary with cancellation and diversion statistics
    """
    total_rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    
    # Cancellation rate
    try:
        cancelled_count = conn.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE Cancelled = 1"
        ).fetchone()[0]
        cancel_rate = (cancelled_count / total_rows * 100) if total_rows > 0 else 0
    except:
        cancelled_count = 0
        cancel_rate = 0
    
    # Diversion rate
    try:
        diverted_count = conn.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE Diverted = 1"
        ).fetchone()[0]
        divert_rate = (diverted_count / total_rows * 100) if total_rows > 0 else 0
    except:
        diverted_count = 0
        divert_rate = 0
    
    # Cancellation codes distribution
    cancel_codes = {}
    try:
        code_counts = conn.execute(
            f"""
            SELECT CancellationCode, COUNT(*) as cnt
            FROM {table_name}
            WHERE Cancelled = 1 AND CancellationCode IS NOT NULL
            GROUP BY CancellationCode
            ORDER BY cnt DESC
            """
        ).fetchall()
        
        for code, count in code_counts:
            cancel_codes[code] = count
    except:
        pass
    
    return {
        'year': year,
        'total_flights': total_rows,
        'cancelled_count': cancelled_count,
        'cancelled_pct': round(cancel_rate, 2),
        'diverted_count': diverted_count,
        'diverted_pct': round(divert_rate, 2),
        'cancel_codes': cancel_codes
    }


def create_availability_matrix(
    conn: duckdb.DuckDBPyConnection,
    table_1993: str,
    table_2003: str
) -> pd.DataFrame:
    """
    Create feature availability matrix comparing columns between years.
    
    Args:
        conn: DuckDB connection
        table_1993: Name of 1993 table
        table_2003: Name of 2003 table
        
    Returns:
        DataFrame with columns: column_name, in_1993, in_2003, in_both
    """
    # Get columns from each table
    cols_1993 = set(
        col[0] for col in conn.execute(f"DESCRIBE {table_1993}").fetchall()
    )
    cols_2003 = set(
        col[0] for col in conn.execute(f"DESCRIBE {table_2003}").fetchall()
    )
    
    all_cols = cols_1993 | cols_2003
    
    availability_data = []
    for col in sorted(all_cols):
        availability_data.append({
            'column_name': col,
            'in_1993': 1 if col in cols_1993 else 0,
            'in_2003': 1 if col in cols_2003 else 0,
            'in_both': 1 if (col in cols_1993 and col in cols_2003) else 0
        })
    
    return pd.DataFrame(availability_data)
