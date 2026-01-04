#!/usr/bin/env python3
"""
Stage 12: Build drilldown tables (hybrid: aggregates + raw queries)

This script:
1. Creates aggregated drilldown tables for carrier, origin airport, and route
2. Precomputes monthly, hourly, and top routes/destinations for each dimension
3. Saves drilldown tables for instant web UI panels
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.flight_delay.config import load_config
from src.flight_delay.io_duckdb import get_duckdb_connection

def create_carrier_monthly_drilldown(conn, parquet_path: str, year: int) -> pd.DataFrame:
    """Create carrier monthly drilldown table."""
    query = f"""
    SELECT 
        UniqueCarrier,
        Month,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
        AVG(ArrDelay) as avg_arr_delay,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ArrDelay) as median_arr_delay
    FROM read_parquet('{parquet_path}')
    GROUP BY UniqueCarrier, Month
    ORDER BY UniqueCarrier, Month
    """
    df = conn.execute(query).fetchdf()
    df['year'] = year
    return df

def create_carrier_dep_hour_drilldown(conn, parquet_path: str, year: int) -> pd.DataFrame:
    """Create carrier departure hour drilldown table."""
    query = f"""
    SELECT 
        UniqueCarrier,
        CAST(FLOOR(CRSDepTime / 100) AS INTEGER) as dep_hour,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
        AVG(ArrDelay) as avg_arr_delay
    FROM read_parquet('{parquet_path}')
    WHERE CRSDepTime IS NOT NULL
    GROUP BY UniqueCarrier, dep_hour
    ORDER BY UniqueCarrier, dep_hour
    """
    df = conn.execute(query).fetchdf()
    df['year'] = year
    return df

def create_carrier_top_routes_drilldown(conn, parquet_path: str, year: int, top_n: int = 20) -> pd.DataFrame:
    """Create carrier top routes drilldown table."""
    query = f"""
    SELECT 
        UniqueCarrier,
        Origin || '_' || Dest as route,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
        AVG(ArrDelay) as avg_arr_delay,
        AVG(Distance) as avg_distance
    FROM read_parquet('{parquet_path}')
    GROUP BY UniqueCarrier, route
    HAVING COUNT(*) >= 10
    ORDER BY UniqueCarrier, total_flights DESC
    """
    df = conn.execute(query).fetchdf()
    # Keep top N routes per carrier
    df = df.groupby('UniqueCarrier').head(top_n).reset_index(drop=True)
    df['year'] = year
    return df

def create_origin_monthly_drilldown(conn, parquet_path: str, year: int) -> pd.DataFrame:
    """Create origin airport monthly drilldown table."""
    query = f"""
    SELECT 
        Origin,
        Month,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
        AVG(ArrDelay) as avg_arr_delay,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ArrDelay) as median_arr_delay
    FROM read_parquet('{parquet_path}')
    GROUP BY Origin, Month
    ORDER BY Origin, Month
    """
    df = conn.execute(query).fetchdf()
    df['year'] = year
    return df

def create_origin_dep_hour_drilldown(conn, parquet_path: str, year: int) -> pd.DataFrame:
    """Create origin airport departure hour drilldown table."""
    query = f"""
    SELECT 
        Origin,
        CAST(FLOOR(CRSDepTime / 100) AS INTEGER) as dep_hour,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
        AVG(ArrDelay) as avg_arr_delay
    FROM read_parquet('{parquet_path}')
    WHERE CRSDepTime IS NOT NULL
    GROUP BY Origin, dep_hour
    ORDER BY Origin, dep_hour
    """
    df = conn.execute(query).fetchdf()
    df['year'] = year
    return df

def create_origin_top_dests_drilldown(conn, parquet_path: str, year: int, top_n: int = 20) -> pd.DataFrame:
    """Create origin airport top destinations drilldown table."""
    query = f"""
    SELECT 
        Origin,
        Dest,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
        AVG(ArrDelay) as avg_arr_delay,
        AVG(Distance) as avg_distance
    FROM read_parquet('{parquet_path}')
    GROUP BY Origin, Dest
    HAVING COUNT(*) >= 10
    ORDER BY Origin, total_flights DESC
    """
    df = conn.execute(query).fetchdf()
    # Keep top N destinations per origin
    df = df.groupby('Origin').head(top_n).reset_index(drop=True)
    df['year'] = year
    return df

def create_route_monthly_drilldown(conn, parquet_path: str, year: int) -> pd.DataFrame:
    """Create route monthly drilldown table."""
    query = f"""
    SELECT 
        Origin || '_' || Dest as route,
        Month,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
        AVG(ArrDelay) as avg_arr_delay,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ArrDelay) as median_arr_delay
    FROM read_parquet('{parquet_path}')
    GROUP BY route, Month
    ORDER BY route, Month
    """
    df = conn.execute(query).fetchdf()
    df['year'] = year
    return df

def create_route_dep_hour_drilldown(conn, parquet_path: str, year: int) -> pd.DataFrame:
    """Create route departure hour drilldown table."""
    query = f"""
    SELECT 
        Origin || '_' || Dest as route,
        CAST(FLOOR(CRSDepTime / 100) AS INTEGER) as dep_hour,
        COUNT(*) as total_flights,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate_pct,
        AVG(ArrDelay) as avg_arr_delay
    FROM read_parquet('{parquet_path}')
    WHERE CRSDepTime IS NOT NULL
    GROUP BY route, dep_hour
    ORDER BY route, dep_hour
    """
    df = conn.execute(query).fetchdf()
    df['year'] = year
    return df

def main():
    """Main execution function for Stage 12."""
    print("=" * 60)
    print("Stage 12: Build Drilldown Tables")
    print("=" * 60)
    
    # Load configuration
    print("\n[1/4] Loading configuration...")
    try:
        config = load_config("config/params.yaml")
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        sys.exit(1)
    
    # Connect to DuckDB
    print("\n[2/4] Connecting to DuckDB...")
    db_path = project_root / "db" / "flights.duckdb"
    conn = get_duckdb_connection(str(db_path))
    print("✓ Connected to database")
    
    # Create output directory
    tables_dir = project_root / "outputs" / "tables" / "drilldown"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Build drilldown tables
    print("\n[3/4] Building drilldown tables...")
    
    for year in [1993, 2003]:
        print(f"\n  Processing {year}...")
        parquet_path = f"parquet/clean/common/year={year}/**/*.parquet"
        
        # Carrier drilldowns
        print(f"    Creating carrier drilldowns...", end=" ")
        carrier_monthly = create_carrier_monthly_drilldown(conn, parquet_path, year)
        carrier_monthly.to_parquet(tables_dir / f"tbl_dd_{'01' if year == 1993 else '02'}_carrier_monthly_{year}.parquet", index=False)
        
        carrier_hour = create_carrier_dep_hour_drilldown(conn, parquet_path, year)
        carrier_hour.to_parquet(tables_dir / f"tbl_dd_{'03' if year == 1993 else '04'}_carrier_dep_hour_{year}.parquet", index=False)
        
        carrier_routes = create_carrier_top_routes_drilldown(conn, parquet_path, year)
        carrier_routes.to_parquet(tables_dir / f"tbl_dd_{'05' if year == 1993 else '06'}_carrier_top_routes_{year}.parquet", index=False)
        print("✓")
        
        # Origin airport drilldowns
        print(f"    Creating origin airport drilldowns...", end=" ")
        origin_monthly = create_origin_monthly_drilldown(conn, parquet_path, year)
        origin_monthly.to_parquet(tables_dir / f"tbl_dd_{'07' if year == 1993 else '08'}_origin_monthly_{year}.parquet", index=False)
        
        origin_hour = create_origin_dep_hour_drilldown(conn, parquet_path, year)
        origin_hour.to_parquet(tables_dir / f"tbl_dd_{'09' if year == 1993 else '10'}_origin_dep_hour_{year}.parquet", index=False)
        
        origin_dests = create_origin_top_dests_drilldown(conn, parquet_path, year)
        origin_dests.to_parquet(tables_dir / f"tbl_dd_{'11' if year == 1993 else '12'}_origin_top_dests_{year}.parquet", index=False)
        print("✓")
        
        # Route drilldowns
        print(f"    Creating route drilldowns...", end=" ")
        route_monthly = create_route_monthly_drilldown(conn, parquet_path, year)
        route_monthly.to_parquet(tables_dir / f"tbl_dd_{'13' if year == 1993 else '14'}_route_monthly_{year}.parquet", index=False)
        
        route_hour = create_route_dep_hour_drilldown(conn, parquet_path, year)
        route_hour.to_parquet(tables_dir / f"tbl_dd_{'15' if year == 1993 else '16'}_route_dep_hour_{year}.parquet", index=False)
        print("✓")
    
    # Summary
    print("\n[4/4] Drilldown Summary:")
    print("=" * 60)
    print("\nCarrier drilldowns:")
    print("  - Monthly breakdowns (2 files)")
    print("  - Departure hour breakdowns (2 files)")
    print("  - Top routes per carrier (2 files)")
    print("\nOrigin airport drilldowns:")
    print("  - Monthly breakdowns (2 files)")
    print("  - Departure hour breakdowns (2 files)")
    print("  - Top destinations per origin (2 files)")
    print("\nRoute drilldowns:")
    print("  - Monthly breakdowns (2 files)")
    print("  - Departure hour breakdowns (2 files)")
    print(f"\nTotal: 16 drilldown tables created")
    print("\n✓ Stage 12 completed successfully!")
    print("=" * 60)
    
    conn.close()

if __name__ == "__main__":
    main()
