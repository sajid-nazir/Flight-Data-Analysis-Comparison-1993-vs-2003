"""
Comparison utilities for 1993 vs 2003
"""
import pandas as pd
import duckdb
from typing import Dict, Tuple


def compute_overall_weighted_delta(
    kpi_df: pd.DataFrame,
    metric_col: str = 'ontime_rate_pct'
) -> Dict[str, float]:
    """
    Compute flight-weighted overall delta between 1993 and 2003.
    
    Args:
        kpi_df: DataFrame with Year, total_flights, and metric columns
        metric_col: Column name for the metric to compare
        
    Returns:
        Dictionary with delta statistics
    """
    df_1993 = kpi_df[kpi_df['Year'] == 1993].iloc[0]
    df_2003 = kpi_df[kpi_df['Year'] == 2003].iloc[0]
    
    total_flights_1993 = df_1993['total_flights']
    total_flights_2003 = df_2003['total_flights']
    total_flights = total_flights_1993 + total_flights_2003
    
    metric_1993 = df_1993[metric_col]
    metric_2003 = df_2003[metric_col]
    
    # Weighted average
    weighted_avg = (metric_1993 * total_flights_1993 + metric_2003 * total_flights_2003) / total_flights
    
    # Simple delta
    delta = metric_2003 - metric_1993
    delta_pct = (delta / metric_1993 * 100) if metric_1993 != 0 else 0
    
    return {
        'metric': metric_col,
        'value_1993': metric_1993,
        'value_2003': metric_2003,
        'delta_absolute': delta,
        'delta_percent': delta_pct,
        'weighted_avg': weighted_avg,
        'total_flights_1993': total_flights_1993,
        'total_flights_2003': total_flights_2003
    }


def compute_delta_by_dimension(
    summary_df: pd.DataFrame,
    dimension_col: str,
    metric_col: str = 'ontime_rate_pct',
    min_volume: int = 10000
) -> pd.DataFrame:
    """
    Compute delta by dimension (carrier, airport, route, etc.).
    
    Args:
        summary_df: DataFrame with Year, dimension_col, metric_col, total_flights
        dimension_col: Column name for the dimension (e.g., 'UniqueCarrier', 'Origin')
        metric_col: Column name for the metric to compare
        min_volume: Minimum total flights across both years to include
        
    Returns:
        DataFrame with delta calculations
    """
    # Pivot to get 1993 and 2003 side by side
    pivot_df = summary_df.pivot_table(
        index=dimension_col,
        columns='Year',
        values=[metric_col, 'total_flights'],
        aggfunc='first'
    )
    
    # Flatten column names
    pivot_df.columns = [f'{col[0]}_{col[1]}' for col in pivot_df.columns]
    
    # Compute totals across both years
    total_col_1993 = f'total_flights_1993'
    total_col_2003 = f'total_flights_2003'
    metric_col_1993 = f'{metric_col}_1993'
    metric_col_2003 = f'{metric_col}_2003'
    
    if total_col_1993 in pivot_df.columns and total_col_2003 in pivot_df.columns:
        pivot_df['total_flights_both'] = (
            pivot_df[total_col_1993].fillna(0) + 
            pivot_df[total_col_2003].fillna(0)
        )
    else:
        # If one year is missing, use available year
        pivot_df['total_flights_both'] = pivot_df[[c for c in pivot_df.columns if 'total_flights' in c]].sum(axis=1)
    
    # Filter by minimum volume
    pivot_df = pivot_df[pivot_df['total_flights_both'] >= min_volume]
    
    # Compute deltas
    if metric_col_1993 in pivot_df.columns and metric_col_2003 in pivot_df.columns:
        pivot_df['delta_absolute'] = (
            pivot_df[metric_col_2003].fillna(0) - 
            pivot_df[metric_col_1993].fillna(0)
        )
        pivot_df['delta_percent'] = (
            (pivot_df['delta_absolute'] / pivot_df[metric_col_1993] * 100)
            .replace([float('inf'), float('-inf')], 0)
            .fillna(0)
        )
    else:
        pivot_df['delta_absolute'] = 0
        pivot_df['delta_percent'] = 0
    
    # Reset index to make dimension_col a regular column
    pivot_df = pivot_df.reset_index()
    
    # Rename columns for clarity
    rename_dict = {}
    if metric_col_1993 in pivot_df.columns:
        rename_dict[metric_col_1993] = f'{metric_col}_1993'
    if metric_col_2003 in pivot_df.columns:
        rename_dict[metric_col_2003] = f'{metric_col}_2003'
    if total_col_1993 in pivot_df.columns:
        rename_dict[total_col_1993] = 'total_flights_1993'
    if total_col_2003 in pivot_df.columns:
        rename_dict[total_col_2003] = 'total_flights_2003'
    
    pivot_df = pivot_df.rename(columns=rename_dict)
    
    return pivot_df


def compute_delta_by_month(
    monthly_df: pd.DataFrame,
    metric_col: str = 'ontime_rate_pct'
) -> pd.DataFrame:
    """Compute delta by month."""
    return compute_delta_by_dimension(monthly_df, 'Month', metric_col, min_volume=0)


def compute_delta_by_dep_hour(
    hourly_df: pd.DataFrame,
    metric_col: str = 'ontime_rate_pct'
) -> pd.DataFrame:
    """Compute delta by departure hour."""
    return compute_delta_by_dimension(hourly_df, 'dep_hour', metric_col, min_volume=0)
