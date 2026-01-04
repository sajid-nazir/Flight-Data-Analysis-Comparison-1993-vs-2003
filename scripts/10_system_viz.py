#!/usr/bin/env python3
"""
Stage 10: System visuals (optional)

This script:
1. Creates route network edges tables
2. Creates delay by departure hour tables
3. Generates route network and delay visualizations
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.flight_delay.config import load_config
from src.flight_delay.io_duckdb import get_duckdb_connection
from src.flight_delay.viz_specs import save_dual, save_plotly_json


def create_route_network_edges(conn, parquet_path: str, year: int) -> pd.DataFrame:
    """Create route network edges table."""
    query = f"""
    SELECT 
        Origin,
        Dest,
        COUNT(*) as flight_count,
        AVG(ArrDelay) as avg_delay,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate
    FROM read_parquet('{parquet_path}')
    GROUP BY Origin, Dest
    HAVING COUNT(*) >= 100
    ORDER BY flight_count DESC
    """
    
    edges = conn.execute(query).fetchdf()
    edges['year'] = year
    return edges


def create_delay_by_hour_table(conn, parquet_path: str, year: int) -> pd.DataFrame:
    """Create delay by departure hour table."""
    query = f"""
    SELECT 
        CAST(FLOOR(CRSDepTime / 100) AS INTEGER) as dep_hour,
        COUNT(*) as flight_count,
        AVG(ArrDelay) as avg_delay,
        MEDIAN(ArrDelay) as median_delay,
        SUM(CASE WHEN ArrDelay <= 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime_rate
    FROM read_parquet('{parquet_path}')
    WHERE CRSDepTime IS NOT NULL
    GROUP BY dep_hour
    ORDER BY dep_hour
    """
    
    hourly = conn.execute(query).fetchdf()
    hourly['year'] = year
    return hourly


def create_route_network_viz(edges_df: pd.DataFrame, year: int) -> go.Figure:
    """Create route network visualization (simplified - top routes only)."""
    # Get top 50 routes by volume
    top_routes = edges_df.nlargest(50, 'flight_count')
    
    # Create scatter plot (simplified network view)
    fig = go.Figure()
    
    # Group by origin to show connections
    for origin in top_routes['Origin'].unique()[:10]:  # Top 10 origins
        origin_routes = top_routes[top_routes['Origin'] == origin]
        fig.add_trace(go.Scatter(
            x=origin_routes['flight_count'],
            y=origin_routes['ontime_rate'],
            mode='markers',
            name=origin,
            marker=dict(size=8, opacity=0.7),
            hovertemplate='<b>%{text}</b><br>Flights: %{x:,}<br>On-Time: %{y:.1f}%<extra></extra>',
            text=[f"{row['Origin']}-{row['Dest']}" for _, row in origin_routes.iterrows()]
        ))
    
    fig.update_layout(
        title=dict(
            text=f"Route Network Overview - {year}<br><sub>Top Routes by Volume and On-Time Performance</sub>",
            x=0.5,
            font=dict(size=20, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="Flight Count (Log Scale)", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            type='log',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title=dict(text="On-Time Rate (%)", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        template="plotly_white",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12, family='Arial Black')
        ),
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=70, r=40, t=100, b=60)
    )
    
    return fig


def create_delay_by_hour_viz(hourly_1993: pd.DataFrame, hourly_2003: pd.DataFrame) -> go.Figure:
    """Create delay by departure hour comparison."""
    fig = go.Figure()
    
    for year, hourly_df, color in [(1993, hourly_1993, '#2E86AB'), (2003, hourly_2003, '#A23B72')]:
        hourly_df = hourly_df.sort_values('dep_hour')
        fig.add_trace(go.Scatter(
            x=hourly_df['dep_hour'],
            y=hourly_df['avg_delay'],
            mode='lines+markers',
            name=str(year),
            line=dict(color=color, width=3),
            marker=dict(size=6, color=color),
            hovertemplate=f'<b>{year}</b><br>Hour: %{{x}}<br>Avg Delay: %{{y:.1f}} min<br>Flights: %{{customdata:,}}<extra></extra>',
            customdata=hourly_df['flight_count']
        ))
    
    fig.update_layout(
        title=dict(
            text="Average Delay by Departure Hour: 1993 vs 2003<br><sub>Hourly Pattern Comparison</sub>",
            x=0.5,
            font=dict(size=20, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="Departure Hour (24-hour format)", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            dtick=2,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title=dict(text="Average Arrival Delay (minutes)", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        template="plotly_white",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12, family='Arial Black')
        ),
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=70, r=40, t=100, b=60)
    )
    
    return fig


def main():
    """Main execution function for Stage 10."""
    print("=" * 60)
    print("Stage 10: System Visuals (Optional)")
    print("=" * 60)
    
    # Load configuration
    print("\n[1/5] Loading configuration...")
    try:
        config = load_config("config/params.yaml")
        export_png = config.get("export_png", True)
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        sys.exit(1)
    
    # Connect to DuckDB
    print("\n[2/5] Connecting to DuckDB...")
    db_path = project_root / "db" / "flights.duckdb"
    conn = get_duckdb_connection(str(db_path))
    print(f"✓ Connected to database")
    
    # Create output directories
    tables_dir = project_root / "outputs" / "tables" / "system"
    viz_dir = project_root / "outputs" / "viz" / "system"
    fig_dir = project_root / "outputs" / "figures" / "system"
    
    tables_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Create route network edges and delay by hour tables
    print("\n[3/5] Creating system tables...")
    all_edges = []
    all_hourly = []
    
    for year in [1993, 2003]:
        print(f"  Processing {year}...")
        parquet_path = str(project_root / "parquet" / "clean" / "common" / f"year={year}" / "**" / "*.parquet")
        
        # Route network edges
        print(f"    Creating route network edges...", end=" ")
        edges = create_route_network_edges(conn, parquet_path, year)
        all_edges.append(edges)
        edges.to_parquet(tables_dir / f"tbl_50_route_graph_edges_{year}.parquet", index=False)
        print(f"✓ ({len(edges)} routes)")
        
        # Delay by hour
        print(f"    Creating delay by hour table...", end=" ")
        hourly = create_delay_by_hour_table(conn, parquet_path, year)
        all_hourly.append(hourly)
        hourly.to_parquet(tables_dir / f"tbl_52_delay_by_dep_hour_{year}.parquet", index=False)
        print(f"✓ ({len(hourly)} hours)")
    
    # Generate visualizations
    print("\n[4/5] Generating visualizations...")
    
    # Route network visualizations
    for year in [1993, 2003]:
        print(f"  Creating route network viz for {year}...", end=" ")
        edges_df = pd.concat([e for e in all_edges if e['year'].iloc[0] == year])
        fig = create_route_network_viz(edges_df, year)
        
        viz_num = 58 if year == 1993 else 59
        save_dual(
            fig,
            str(viz_dir / f"viz_{viz_num:02d}_route_network_{year}.plotly.json"),
            str(fig_dir / f"fig_{viz_num:02d}_route_network_{year}.png"),
            export_png=export_png
        )
        print("✓")
    
    # Delay by hour comparison
    print("  Creating delay by hour comparison...", end=" ")
    hourly_1993 = pd.concat([h for h in all_hourly if h['year'].iloc[0] == 1993])
    hourly_2003 = pd.concat([h for h in all_hourly if h['year'].iloc[0] == 2003])
    fig = create_delay_by_hour_viz(hourly_1993, hourly_2003)
    save_dual(
        fig,
        str(viz_dir / "viz_60_delay_by_dep_hour_1993_vs_2003.plotly.json"),
        str(fig_dir / "fig_60_delay_by_dep_hour_1993_vs_2003.png"),
        export_png=export_png
    )
    print("✓")
    
    # Summary
    print("\n[5/5] System Visuals Summary:")
    print("=" * 60)
    for year in [1993, 2003]:
        edges_df = pd.concat([e for e in all_edges if e['year'].iloc[0] == year])
        hourly_df = pd.concat([h for h in all_hourly if h['year'].iloc[0] == year])
        print(f"\n{year}:")
        print(f"  Routes: {len(edges_df):,}")
        print(f"  Hourly delay data: {len(hourly_df)} hours")
    
    print("\n✓ Stage 10 completed successfully!")
    print("=" * 60)
    
    # Close connection
    conn.close()


if __name__ == "__main__":
    main()
