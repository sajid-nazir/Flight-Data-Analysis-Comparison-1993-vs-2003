"""
Feature engineering utilities
"""
import pandas as pd
import duckdb
from typing import Dict, List, Tuple
import json
from pathlib import Path
import numpy as np


def build_base_features(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str,
    year: int,
    on_time_threshold: int = 15
) -> pd.DataFrame:
    """
    Build base features from clean parquet data.
    
    Args:
        conn: DuckDB connection
        parquet_path: Path pattern to clean parquet files
        year: Year for labeling
        on_time_threshold: Minutes threshold for on-time (default 15)
        
    Returns:
        DataFrame with base features and target
    """
    query = f"""
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
        -- Build target
        CASE WHEN ArrDelay <= {on_time_threshold} THEN 1 ELSE 0 END as ontime15,
        -- Build route
        Origin || '_' || Dest as route,
        -- Build dep hour from CRSDepTime
        CAST(FLOOR(CRSDepTime / 100) AS INTEGER) as dep_hour_raw
    FROM read_parquet('{parquet_path}')
    WHERE ArrDelay IS NOT NULL
    """
    
    df = conn.execute(query).fetchdf()
    
    # Bin departure hour
    df['dep_hour_bin'] = pd.cut(
        df['dep_hour_raw'],
        bins=[0, 6, 10, 14, 18, 22, 24],
        labels=['late_night', 'early_morning', 'mid_morning', 'afternoon', 'evening', 'night'],
        include_lowest=True
    ).astype(str)
    
    # Bin distance
    df['distance_bin'] = pd.cut(
        df['Distance'],
        bins=[0, 500, 1000, 1500, 2000, 10000],
        labels=['short', 'medium', 'long', 'very_long', 'ultra_long'],
        include_lowest=True
    ).astype(str)
    
    return df


def compute_congestion_features(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str
) -> pd.DataFrame:
    """
    Compute congestion proxy features (hourly volumes at origin and destination).
    
    Args:
        conn: DuckDB connection
        parquet_path: Path pattern to clean parquet files
        
    Returns:
        DataFrame with congestion features
    """
    query = f"""
    WITH base_data AS (
        SELECT 
            Year,
            Month,
            DayOfWeek,
            CAST(FLOOR(CRSDepTime / 100) AS INTEGER) as dep_hour,
            Origin,
            Dest
        FROM read_parquet('{parquet_path}')
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
        AND a.Dest = d.Dest
    """
    
    return conn.execute(query).fetchdf()


def create_split_assignments(
    df: pd.DataFrame,
    train_months: List[int],
    test_months: List[int]
) -> pd.DataFrame:
    """
    Create train/test split assignments based on months.
    
    Args:
        df: DataFrame with Month column
        train_months: List of months for training
        test_months: List of months for testing
        
    Returns:
        DataFrame with split column added
    """
    df = df.copy()
    df['split'] = df['Month'].apply(
        lambda x: 'train' if x in train_months else ('test' if x in test_months else 'other')
    )
    return df


def fit_target_encoders(
    train_df: pd.DataFrame,
    categorical_cols: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Fit target encoders (mean encoding) on training data.
    
    This computes the mean of the target variable (ontime15) for each category,
    which is target encoding (not frequency encoding).
    
    Args:
        train_df: Training DataFrame
        categorical_cols: List of categorical column names to encode
        
    Returns:
        Dictionary mapping column names to value->target_rate mappings
    """
    encoders = {}
    
    for col in categorical_cols:
        if col not in train_df.columns:
            continue
            
        # Compute target encoding (mean of ontime15 for each category)
        target_map = train_df.groupby(col)['ontime15'].mean().to_dict()
        encoders[col] = target_map
    
    return encoders


def apply_target_encoders(
    df: pd.DataFrame,
    encoders: Dict[str, Dict[str, float]],
    suffix: str = '_freq'
) -> pd.DataFrame:
    """
    Apply target encoders to DataFrame.
    
    Args:
        df: DataFrame to encode
        encoders: Dictionary of encoders (from fit_target_encoders)
        suffix: Suffix to add to encoded column names
        
    Returns:
        DataFrame with encoded columns added
    """
    df = df.copy()
    
    for col, target_map in encoders.items():
        if col not in df.columns:
            continue
            
        encoded_col = f"{col}{suffix}"
        df[encoded_col] = df[col].map(target_map).fillna(0.5)  # Default to 0.5 for unseen values
    
    return df


def save_encoders(encoders: Dict[str, Dict[str, float]], filepath: str) -> None:
    """Save target encoders to JSON file."""
    # Convert numpy types to native Python types for JSON serialization
    encoders_serializable = {}
    for col, freq_map in encoders.items():
        encoders_serializable[col] = {
            str(k): float(v) for k, v in freq_map.items()
        }
    
    with open(filepath, 'w') as f:
        json.dump(encoders_serializable, f, indent=2)


def load_encoders(filepath: str) -> Dict[str, Dict[str, float]]:
    """Load target encoders from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)
