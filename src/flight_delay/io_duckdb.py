"""
DuckDB I/O utilities
"""
import duckdb
from pathlib import Path
from typing import Optional, Dict, List, Any


def get_duckdb_connection(db_path: str = "db/flights.duckdb") -> duckdb.DuckDBPyConnection:
    """
    Create or open DuckDB database connection.
    
    Args:
        db_path: Path to DuckDB database file
        
    Returns:
        DuckDB connection object
        
    Raises:
        OSError: If database file cannot be created/opened
    """
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    
    conn = duckdb.connect(str(db_file))
    return conn


def read_csv_to_duckdb(
    conn: duckdb.DuckDBPyConnection,
    csv_path: str,
    table_name: str,
    year: Optional[int] = None
) -> None:
    """
    Read CSV file into DuckDB table.
    
    Args:
        conn: DuckDB connection
        csv_path: Path to CSV file
        table_name: Name for the table in DuckDB
        year: Optional year to add as a column
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV with automatic schema detection
    # Handle "NA" as null string and use larger sample size for better type detection
    query = f"""
    CREATE OR REPLACE TABLE {table_name} AS
    SELECT * FROM read_csv_auto('{csv_path}', 
        nullstr='NA',
        sample_size=-1,
        ignore_errors=false
    )
    """
    
    conn.execute(query)
    
    # Add year column if provided and not already present
    if year is not None:
        # Check if year column exists
        columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
        column_names = [col[0].lower() for col in columns]
        
        if 'year' not in column_names:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN year INTEGER")
            conn.execute(f"UPDATE {table_name} SET year = {year}")
        else:
            # Update year if it exists but might be wrong
            conn.execute(f"UPDATE {table_name} SET year = {year} WHERE year IS NULL OR year != {year}")


def write_partitioned_parquet(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    output_dir: str,
    partition_cols: List[str],
    overwrite: bool = True
) -> None:
    """
    Write DuckDB table to partitioned Parquet files.
    
    Args:
        conn: DuckDB connection
        table_name: Name of the table to export
        output_dir: Base directory for output Parquet files
        partition_cols: List of column names to partition by
        overwrite: Whether to overwrite existing files (default: True)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build partition clause
    partition_clause = ", ".join(partition_cols)
    
    # Use COPY TO with PARTITION_BY
    overwrite_clause = "OVERWRITE_OR_IGNORE" if overwrite else ""
    query = f"""
    COPY (SELECT * FROM {table_name}) 
    TO '{output_dir}' 
    (FORMAT PARQUET, PARTITION_BY ({partition_clause}), {overwrite_clause})
    """
    
    conn.execute(query)


def get_table_schema(conn: duckdb.DuckDBPyConnection, table_name: str) -> Dict[str, Any]:
    """
    Get schema information for a table.
    
    Args:
        conn: DuckDB connection
        table_name: Name of the table
        
    Returns:
        Dictionary with schema information (column names and types)
    """
    schema_info = conn.execute(f"DESCRIBE {table_name}").fetchall()
    
    schema = {
        "columns": [],
        "column_count": len(schema_info)
    }
    
    for col_name, col_type, null, key, default, extra in schema_info:
        schema["columns"].append({
            "name": col_name,
            "type": str(col_type),
            "nullable": null == "YES"
        })
    
    return schema


def get_row_count(conn: duckdb.DuckDBPyConnection, table_name: str) -> int:
    """
    Get total row count from a table.
    
    Args:
        conn: DuckDB connection
        table_name: Name of the table
        
    Returns:
        Total number of rows
    """
    result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    return result[0] if result else 0


def get_row_count_by_partition(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    partition_cols: List[str]
) -> List[Dict[str, Any]]:
    """
    Get row counts grouped by partition columns.
    
    Args:
        conn: DuckDB connection
        table_name: Name of the table
        partition_cols: List of column names to group by
        
    Returns:
        List of dictionaries with partition values and row counts
    """
    group_by_cols = ", ".join(partition_cols)
    select_cols = ", ".join(partition_cols + ["COUNT(*) as row_count"])
    
    query = f"""
    SELECT {select_cols}
    FROM {table_name}
    GROUP BY {group_by_cols}
    ORDER BY {group_by_cols}
    """
    
    results = conn.execute(query).fetchall()
    
    # Convert to list of dictionaries
    rows = []
    for row in results:
        row_dict = {}
        for i, col in enumerate(partition_cols):
            row_dict[col] = row[i]
        row_dict["row_count"] = row[len(partition_cols)]
        rows.append(row_dict)
    
    return rows
