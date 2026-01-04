#!/usr/bin/env python3
"""
Stage 04: Exploratory Data Analysis

This script:
1. Computes core KPIs for both years
2. Analyzes on-time performance across multiple dimensions (month, day-of-week, hour, carrier, airport, route)
3. Performs route-matched and airport-matched analysis for fair comparison
4. Generates descriptive visualizations comparing 1993 vs 2003
"""
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.flight_delay.config import load_config
from src.flight_delay.io_duckdb import get_duckdb_connection
from src.flight_delay.io_artifacts import save_json
from src.flight_delay.eda import (
    compute_core_kpis,
    compute_ontime_by_month,
    compute_ontime_by_dow,
    compute_ontime_by_dep_hour,
    compute_carrier_summary,
    compute_airport_summary,
    compute_route_summary,
    compute_route_matched_summary,
    compute_airport_matched_summary
)
from src.flight_delay.viz_specs import save_dual, save_plotly_json


def create_kpi_panel(kpi_df: pd.DataFrame) -> go.Figure:
    """Create multi-panel KPI dashboard."""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Total Flights', 'On-Time Rate (%)', 'Mean Arrival Delay (min)',
            'Median Arrival Delay (min)', 'Mean Distance (miles)', 'Mean Departure Delay (min)'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    years = kpi_df['Year'].astype(str)
    colors = ['#2E86AB', '#A23B72']
    
    # Total flights
    fig.add_trace(
        go.Bar(x=years, y=kpi_df['total_flights'], marker_color=colors, showlegend=False,
               text=[f"{val:,.0f}" for val in kpi_df['total_flights']], textposition='outside',
               textfont=dict(size=12, family='Arial Black')),
        row=1, col=1
    )
    
    # On-time rate
    fig.add_trace(
        go.Bar(x=years, y=kpi_df['ontime_rate_pct'], marker_color=colors, showlegend=False,
               text=[f"{val:.1f}%" for val in kpi_df['ontime_rate_pct']], textposition='outside',
               textfont=dict(size=12, family='Arial Black')),
        row=1, col=2
    )
    
    # Mean arrival delay
    fig.add_trace(
        go.Bar(x=years, y=kpi_df['mean_arr_delay'], marker_color=colors, showlegend=False,
               text=[f"{val:.1f}" for val in kpi_df['mean_arr_delay']], textposition='outside',
               textfont=dict(size=12, family='Arial Black')),
        row=1, col=3
    )
    
    # Median arrival delay
    fig.add_trace(
        go.Bar(x=years, y=kpi_df['median_arr_delay'], marker_color=colors, showlegend=False,
               text=[f"{val:.1f}" for val in kpi_df['median_arr_delay']], textposition='outside',
               textfont=dict(size=12, family='Arial Black')),
        row=2, col=1
    )
    
    # Mean distance
    fig.add_trace(
        go.Bar(x=years, y=kpi_df['mean_distance'], marker_color=colors, showlegend=False,
               text=[f"{val:.0f}" for val in kpi_df['mean_distance']], textposition='outside',
               textfont=dict(size=12, family='Arial Black')),
        row=2, col=2
    )
    
    # Mean departure delay
    fig.add_trace(
        go.Bar(x=years, y=kpi_df['mean_dep_delay'], marker_color=colors, showlegend=False,
               text=[f"{val:.1f}" for val in kpi_df['mean_dep_delay']], textposition='outside',
               textfont=dict(size=12, family='Arial Black')),
        row=2, col=3
    )
    
    # Update axis labels for each subplot
    fig.update_xaxes(title_text="Year", row=1, col=1, tickfont=dict(size=11, family='Arial'))
    fig.update_xaxes(title_text="Year", row=1, col=2, tickfont=dict(size=11, family='Arial'))
    fig.update_xaxes(title_text="Year", row=1, col=3, tickfont=dict(size=11, family='Arial'))
    fig.update_xaxes(title_text="Year", row=2, col=1, tickfont=dict(size=11, family='Arial'))
    fig.update_xaxes(title_text="Year", row=2, col=2, tickfont=dict(size=11, family='Arial'))
    fig.update_xaxes(title_text="Year", row=2, col=3, tickfont=dict(size=11, family='Arial'))
    
    fig.update_yaxes(title_text="Count", row=1, col=1, tickfont=dict(size=11, family='Arial'))
    fig.update_yaxes(title_text="Percentage", row=1, col=2, tickfont=dict(size=11, family='Arial'))
    fig.update_yaxes(title_text="Minutes", row=1, col=3, tickfont=dict(size=11, family='Arial'))
    fig.update_yaxes(title_text="Minutes", row=2, col=1, tickfont=dict(size=11, family='Arial'))
    fig.update_yaxes(title_text="Miles", row=2, col=2, tickfont=dict(size=11, family='Arial'))
    fig.update_yaxes(title_text="Minutes", row=2, col=3, tickfont=dict(size=11, family='Arial'))
    
    fig.update_layout(
        title=dict(
            text="Core KPIs: 1993 vs 2003<br><sub>Key Performance Indicators Comparison</sub>",
            x=0.5, font=dict(size=22, family='Arial Black', color='#1a1a1a'), y=0.98
        ),
        template="plotly_white",
        height=800,
        showlegend=False,
        font=dict(family="Arial", size=11),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=120, b=60)
    )
    
    return fig


def create_monthly_comparison(monthly_df: pd.DataFrame) -> go.Figure:
    """Create monthly on-time rate comparison."""
    fig = go.Figure()
    
    for year in [1993, 2003]:
        year_data = monthly_df[monthly_df['Year'] == year].sort_values('Month')
        color = '#2E86AB' if year == 1993 else '#A23B72'
        
        fig.add_trace(go.Scatter(
            x=year_data['Month'],
            y=year_data['ontime_rate_pct'],
            mode='lines+markers',
            name=str(year),
            line=dict(color=color, width=3),
            marker=dict(size=8, color=color, symbol='circle' if year == 1993 else 'diamond'),
            hovertemplate=f'<b>{year}</b><br>Month: %{{x}}<br>On-Time Rate: %{{y:.1f}}%<br>Flights: %{{customdata:,}}<extra></extra>',
            customdata=year_data['total_flights']
        ))
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig.update_xaxes(
        title=dict(text="Month", font=dict(size=14, family='Arial Black')),
        tickfont=dict(size=12, family='Arial'),
        tickmode='linear',
        tick0=1,
        dtick=1,
        ticktext=month_names,
        tickvals=list(range(1, 13)),
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)'
    )
    fig.update_yaxes(
        title=dict(text="On-Time Rate (%)", font=dict(size=14, family='Arial Black')),
        tickfont=dict(size=12, family='Arial'),
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)'
    )
    
    fig.update_layout(
        title=dict(
            text="On-Time Performance by Month: 1993 vs 2003<br><sub>Monthly On-Time Rate Comparison</sub>",
            x=0.5, font=dict(size=22, family='Arial Black', color='#1a1a1a'), y=0.98
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


def create_dow_comparison(dow_df: pd.DataFrame) -> go.Figure:
    """Create day-of-week comparison."""
    fig = go.Figure()
    
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    for year in [1993, 2003]:
        year_data = dow_df[dow_df['Year'] == year].sort_values('DayOfWeek')
        color = '#2E86AB' if year == 1993 else '#A23B72'
        
        fig.add_trace(go.Bar(
            x=[dow_names[int(d)-1] for d in year_data['DayOfWeek']],
            y=year_data['ontime_rate_pct'],
            name=str(year),
            marker_color=color,
            marker_line=dict(color='white', width=2),
            hovertemplate=f'<b>{year}</b><br>Day: %{{x}}<br>On-Time Rate: %{{y:.1f}}%<br>Flights: %{{customdata:,}}<extra></extra>',
            customdata=year_data['total_flights']
        ))
    
    fig.update_xaxes(
        title=dict(text="Day of Week", font=dict(size=14, family='Arial Black')),
        tickfont=dict(size=12, family='Arial'),
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)'
    )
    fig.update_yaxes(
        title=dict(text="On-Time Rate (%)", font=dict(size=14, family='Arial Black')),
        tickfont=dict(size=12, family='Arial'),
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)'
    )
    
    fig.update_layout(
        title=dict(
            text="On-Time Performance by Day of Week: 1993 vs 2003<br><sub>Weekly Pattern Comparison</sub>",
            x=0.5, font=dict(size=22, family='Arial Black', color='#1a1a1a'), y=0.98
        ),
        template="plotly_white",
        height=600,
        barmode='group',
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


def create_hourly_comparison(hourly_df: pd.DataFrame) -> go.Figure:
    """Create departure hour comparison."""
    fig = go.Figure()
    
    for year in [1993, 2003]:
        year_data = hourly_df[hourly_df['Year'] == year].sort_values('dep_hour')
        color = '#2E86AB' if year == 1993 else '#A23B72'
        
        fig.add_trace(go.Scatter(
            x=year_data['dep_hour'],
            y=year_data['ontime_rate_pct'],
            mode='lines+markers',
            name=str(year),
            line=dict(color=color, width=3),
            marker=dict(size=6, color=color, symbol='circle' if year == 1993 else 'diamond'),
            hovertemplate=f'<b>{year}</b><br>Hour: %{{x}}<br>On-Time Rate: %{{y:.1f}}%<br>Flights: %{{customdata:,}}<extra></extra>',
            customdata=year_data['total_flights']
        ))
    
    fig.update_xaxes(
        title=dict(text="Departure Hour (24-hour format)", font=dict(size=14, family='Arial Black')),
        tickfont=dict(size=12, family='Arial'),
        dtick=2,
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)'
    )
    fig.update_yaxes(
        title=dict(text="On-Time Rate (%)", font=dict(size=14, family='Arial Black')),
        tickfont=dict(size=12, family='Arial'),
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)'
    )
    
    fig.update_layout(
        title=dict(
            text="On-Time Performance by Departure Hour: 1993 vs 2003<br><sub>Hourly Pattern Comparison</sub>",
            x=0.5, font=dict(size=22, family='Arial Black', color='#1a1a1a'), y=0.98
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


def create_carrier_chart(carrier_df: pd.DataFrame, year: int) -> go.Figure:
    """Create top 10 carriers chart for a specific year."""
    year_data = carrier_df[carrier_df['Year'] == year].head(10).sort_values('ontime_rate_pct', ascending=True)
    color = '#2E86AB' if year == 1993 else '#A23B72'
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=year_data['UniqueCarrier'],
        x=year_data['ontime_rate_pct'],
        orientation='h',
        marker_color=color,
        marker_line=dict(color='white', width=2),
        text=[f"{val:.1f}%" for val in year_data['ontime_rate_pct']],
        textposition='outside',
        textfont=dict(size=11, family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>On-Time Rate: %{x:.1f}%<br>Flights: %{customdata:,}<extra></extra>',
        customdata=year_data['total_flights']
    ))
    
    fig.update_xaxes(
        title=dict(text="On-Time Rate (%)", font=dict(size=14, family='Arial Black')),
        tickfont=dict(size=12, family='Arial'),
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)'
    )
    fig.update_yaxes(
        title=dict(text="Carrier", font=dict(size=14, family='Arial Black')),
        tickfont=dict(size=12, family='Arial'),
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)'
    )
    
    fig.update_layout(
        title=dict(
            text=f"Top 10 Carriers by Volume - {year}<br><sub>On-Time Performance</sub>",
            x=0.5, font=dict(size=20, family='Arial Black', color='#1a1a1a'), y=0.98
        ),
        template="plotly_white",
        height=500,
        showlegend=False,
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=80, r=40, t=100, b=60)
    )
    
    return fig


def create_route_matched_chart(route_matched_df: pd.DataFrame) -> go.Figure:
    """Create route-matched comparison scatter plot."""
    # Pivot to get 1993 and 2003 rates side by side
    route_pivot = route_matched_df.pivot_table(
        index='route', columns='Year', values='ontime_rate_pct', aggfunc='first'
    ).reset_index()
    
    # Filter to routes with data in both years
    route_pivot = route_pivot.dropna()
    
    if len(route_pivot) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.add_annotation(text="No common routes found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Get top 20 by total volume
    route_volumes = route_matched_df.groupby('route')['total_flights'].sum().sort_values(ascending=False)
    top_routes = route_volumes.head(20).index.tolist()
    route_pivot = route_pivot[route_pivot['route'].isin(top_routes)]
    
    if len(route_pivot) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.add_annotation(text="No top routes found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure()
    
    # Diagonal reference line (y=x) - add first so it's behind
    max_val = max(route_pivot[1993].max(), route_pivot[2003].max())
    min_val = min(route_pivot[1993].min(), route_pivot[2003].min())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='gray', dash='dash', width=2),
        name='Equal Performance',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Scatter plot - show only markers, not text labels (too cluttered)
    fig.add_trace(go.Scatter(
        x=route_pivot[1993],
        y=route_pivot[2003],
        mode='markers',
        marker=dict(size=12, color='#A23B72', opacity=0.7, line=dict(color='white', width=1)),
        text=route_pivot['route'],
        hovertemplate='<b>%{text}</b><br>1993: %{x:.1f}%<br>2003: %{y:.1f}%<br>Delta: %{customdata:+.1f}%<extra></extra>',
        customdata=route_pivot[2003] - route_pivot[1993],
        name='Routes',
        showlegend=True
    ))
    
    fig.update_xaxes(
        title=dict(text="1993 On-Time Rate (%)", font=dict(size=14, family='Arial Black')),
        tickfont=dict(size=12, family='Arial')
    )
    fig.update_yaxes(
        title=dict(text="2003 On-Time Rate (%)", font=dict(size=14, family='Arial Black')),
        tickfont=dict(size=12, family='Arial')
    )
    
    fig.update_layout(
        title=dict(
            text="Route-Matched Comparison: 1993 vs 2003<br><sub>Top 20 Routes by Volume (Common Routes Only)</sub>",
            x=0.5, font=dict(size=20, family='Arial Black', color='#1a1a1a'), y=0.98
        ),
        template="plotly_white",
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=12, family='Arial Black')),
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=70, r=40, t=100, b=60)
    )
    
    return fig


def create_airport_matched_chart(airport_matched_df: pd.DataFrame) -> go.Figure:
    """Create airport-matched comparison chart."""
    if len(airport_matched_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Get top 20 airports by total volume
    airport_volumes = airport_matched_df.groupby('airport')['total_flights'].sum().sort_values(ascending=False)
    top_airports = airport_volumes.head(20).index.tolist()
    
    if len(top_airports) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No airports found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Filter and pivot
    top_data = airport_matched_df[airport_matched_df['airport'].isin(top_airports)].copy()
    airport_pivot = top_data.pivot_table(
        index=['airport', 'airport_role'], columns='Year', values='ontime_rate_pct', aggfunc='first'
    ).reset_index()
    airport_pivot = airport_pivot.dropna()
    
    if len(airport_pivot) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No matching data found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Separate origin and destination
    origin_data = airport_pivot[airport_pivot['airport_role'] == 'origin'].head(10)
    dest_data = airport_pivot[airport_pivot['airport_role'] == 'destination'].head(10)
    
    if len(origin_data) == 0 and len(dest_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No origin/destination data found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Sort by 1993 values for better visualization
    if len(origin_data) > 0:
        origin_data = origin_data.sort_values(1993, ascending=True)
    if len(dest_data) > 0:
        dest_data = dest_data.sort_values(1993, ascending=True)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Top 10 Origin Airports', 'Top 10 Destination Airports'),
        horizontal_spacing=0.15
    )
    
    # Origin airports - 1993
    if len(origin_data) > 0:
        fig.add_trace(go.Bar(
            x=origin_data[1993],
            y=origin_data['airport'],
            orientation='h',
            name='1993',
            marker_color='#2E86AB',
            marker_line=dict(color='white', width=1),
            showlegend=True,
            text=[f"{val:.1f}%" for val in origin_data[1993]],
            textposition='outside',
            textfont=dict(size=9, family='Arial Black'),
            hovertemplate='<b>%{y}</b><br>1993: %{x:.1f}%<extra></extra>'
        ), row=1, col=1)
        
        # Origin airports - 2003
        fig.add_trace(go.Bar(
            x=origin_data[2003],
            y=origin_data['airport'],
            orientation='h',
            name='2003',
            marker_color='#A23B72',
            marker_line=dict(color='white', width=1),
            showlegend=True,
            text=[f"{val:.1f}%" for val in origin_data[2003]],
            textposition='outside',
            textfont=dict(size=9, family='Arial Black'),
            hovertemplate='<b>%{y}</b><br>2003: %{x:.1f}%<extra></extra>'
        ), row=1, col=1)
    
    # Destination airports - 1993
    if len(dest_data) > 0:
        fig.add_trace(go.Bar(
            x=dest_data[1993],
            y=dest_data['airport'],
            orientation='h',
            name='1993',
            marker_color='#2E86AB',
            marker_line=dict(color='white', width=1),
            showlegend=False,
            text=[f"{val:.1f}%" for val in dest_data[1993]],
            textposition='outside',
            textfont=dict(size=9, family='Arial Black'),
            hovertemplate='<b>%{y}</b><br>1993: %{x:.1f}%<extra></extra>'
        ), row=1, col=2)
        
        # Destination airports - 2003
        fig.add_trace(go.Bar(
            x=dest_data[2003],
            y=dest_data['airport'],
            orientation='h',
            name='2003',
            marker_color='#A23B72',
            marker_line=dict(color='white', width=1),
            showlegend=False,
            text=[f"{val:.1f}%" for val in dest_data[2003]],
            textposition='outside',
            textfont=dict(size=9, family='Arial Black'),
            hovertemplate='<b>%{y}</b><br>2003: %{x:.1f}%<extra></extra>'
        ), row=1, col=2)
    
    fig.update_xaxes(
        title=dict(text="On-Time Rate (%)", font=dict(size=14, family='Arial Black')),
        tickfont=dict(size=12, family='Arial'),
        row=1, col=1
    )
    fig.update_xaxes(
        title=dict(text="On-Time Rate (%)", font=dict(size=14, family='Arial Black')),
        tickfont=dict(size=12, family='Arial'),
        row=1, col=2
    )
    fig.update_yaxes(
        title=dict(text="Airport", font=dict(size=14, family='Arial Black')),
        tickfont=dict(size=11, family='Arial'),
        row=1, col=1
    )
    fig.update_yaxes(
        title=dict(text="Airport", font=dict(size=14, family='Arial Black')),
        tickfont=dict(size=11, family='Arial'),
        row=1, col=2
    )
    
    fig.update_layout(
        title=dict(
            text="Airport-Matched Comparison: 1993 vs 2003<br><sub>Top 10 Common Airports by Volume</sub>",
            x=0.5, font=dict(size=20, family='Arial Black', color='#1a1a1a'), y=0.98
        ),
        template="plotly_white",
        height=600,
        barmode='group',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=12, family='Arial Black')),
        font=dict(family="Arial", size=11),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=100, r=40, t=120, b=60)
    )
    
    return fig


def main():
    """Main execution function for Stage 04."""
    print("=" * 60)
    print("Stage 04: Exploratory Data Analysis")
    print("=" * 60)
    
    # Load configuration
    print("\n[1/8] Loading configuration...")
    try:
        config = load_config("config/params.yaml")
        export_png = config.get("export_png", True)
        on_time_threshold = config.get("on_time_threshold_min", 15)
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        sys.exit(1)
    
    # Connect to DuckDB
    print("\n[2/8] Connecting to DuckDB...")
    db_path = project_root / "db" / "flights.duckdb"
    conn = get_duckdb_connection(str(db_path))
    print(f"✓ Connected to database")
    
    # Check if clean_common Parquet files exist
    print("\n[3/8] Checking clean_common Parquet files...")
    parquet_1993 = project_root / "parquet" / "clean" / "common" / "year=1993"
    parquet_2003 = project_root / "parquet" / "clean" / "common" / "year=2003"
    
    if not parquet_1993.exists() or not list(parquet_1993.glob("**/*.parquet")):
        print(f"✗ Error: Clean common Parquet files for 1993 not found. Please run Stage 03 first.")
        sys.exit(1)
    if not parquet_2003.exists() or not list(parquet_2003.glob("**/*.parquet")):
        print(f"✗ Error: Clean common Parquet files for 2003 not found. Please run Stage 03 first.")
        sys.exit(1)
    
    # Build Parquet path patterns
    parquet_path_1993 = str(parquet_1993 / "**" / "*.parquet")
    parquet_path_2003 = str(parquet_2003 / "**" / "*.parquet")
    print(f"✓ Found Parquet files")
    
    # Create output directories
    eda_dir = project_root / "outputs" / "tables" / "eda"
    viz_dir = project_root / "outputs" / "viz" / "eda"
    fig_dir = project_root / "outputs" / "figures" / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute all aggregate tables
    print("\n[4/8] Computing aggregate tables...")
    
    print("  Computing core KPIs...", end=" ")
    kpi_df = compute_core_kpis(conn, parquet_path_1993, parquet_path_2003, on_time_threshold)
    kpi_path = eda_dir / "tbl_10_core_kpis_by_year.csv"
    kpi_df.to_csv(kpi_path, index=False)
    print(f"✓ Saved: {kpi_path}")
    
    print("  Computing on-time by month...", end=" ")
    monthly_df = compute_ontime_by_month(conn, parquet_path_1993, parquet_path_2003, on_time_threshold)
    monthly_path = eda_dir / "tbl_11_ontime_by_month.parquet"
    monthly_df.to_parquet(monthly_path, index=False)
    print(f"✓ Saved: {monthly_path}")
    
    print("  Computing on-time by day of week...", end=" ")
    dow_df = compute_ontime_by_dow(conn, parquet_path_1993, parquet_path_2003, on_time_threshold)
    dow_path = eda_dir / "tbl_12_ontime_by_dow.parquet"
    dow_df.to_parquet(dow_path, index=False)
    print(f"✓ Saved: {dow_path}")
    
    print("  Computing on-time by departure hour...", end=" ")
    hourly_df = compute_ontime_by_dep_hour(conn, parquet_path_1993, parquet_path_2003, on_time_threshold)
    hourly_path = eda_dir / "tbl_13_ontime_by_dep_hour.parquet"
    hourly_df.to_parquet(hourly_path, index=False)
    print(f"✓ Saved: {hourly_path}")
    
    print("  Computing carrier summary...", end=" ")
    carrier_df = compute_carrier_summary(conn, parquet_path_1993, parquet_path_2003, on_time_threshold)
    carrier_path = eda_dir / "tbl_14_carrier_summary.parquet"
    carrier_df.to_parquet(carrier_path, index=False)
    print(f"✓ Saved: {carrier_path}")
    
    print("  Computing origin airport summary...", end=" ")
    origin_df = compute_airport_summary(conn, parquet_path_1993, parquet_path_2003, 'origin', on_time_threshold)
    origin_path = eda_dir / "tbl_15_origin_airport_summary.parquet"
    origin_df.to_parquet(origin_path, index=False)
    print(f"✓ Saved: {origin_path}")
    
    print("  Computing destination airport summary...", end=" ")
    dest_df = compute_airport_summary(conn, parquet_path_1993, parquet_path_2003, 'dest', on_time_threshold)
    dest_path = eda_dir / "tbl_16_dest_airport_summary.parquet"
    dest_df.to_parquet(dest_path, index=False)
    print(f"✓ Saved: {dest_path}")
    
    print("  Computing route summary...", end=" ")
    route_df = compute_route_summary(conn, parquet_path_1993, parquet_path_2003, on_time_threshold)
    route_path = eda_dir / "tbl_17_route_summary.parquet"
    route_df.to_parquet(route_path, index=False)
    print(f"✓ Saved: {route_path}")
    
    print("  Computing route-matched summary...", end=" ")
    route_matched_df = compute_route_matched_summary(conn, parquet_path_1993, parquet_path_2003, on_time_threshold)
    route_matched_path = eda_dir / "tbl_18_route_matched_summary.parquet"
    route_matched_df.to_parquet(route_matched_path, index=False)
    print(f"✓ Saved: {route_matched_path}")
    
    print("  Computing airport-matched summary...", end=" ")
    airport_matched_df = compute_airport_matched_summary(conn, parquet_path_1993, parquet_path_2003, on_time_threshold)
    airport_matched_path = eda_dir / "tbl_19_airport_matched_summary.parquet"
    airport_matched_df.to_parquet(airport_matched_path, index=False)
    print(f"✓ Saved: {airport_matched_path}")
    
    # Generate visualizations
    print("\n[5/8] Generating visualizations...")
    
    print("  Creating KPI panel...", end=" ")
    fig_kpi = create_kpi_panel(kpi_df)
    save_dual(
        fig_kpi,
        str(viz_dir / "viz_11_kpi_panel_1993_vs_2003.plotly.json"),
        str(fig_dir / "fig_11_kpi_panel_1993_vs_2003.png"),
        export_png=export_png
    )
    print("✓")
    
    print("  Creating monthly comparison...", end=" ")
    fig_monthly = create_monthly_comparison(monthly_df)
    save_dual(
        fig_monthly,
        str(viz_dir / "viz_12_ontime_by_month_1993_vs_2003.plotly.json"),
        str(fig_dir / "fig_12_ontime_by_month_1993_vs_2003.png"),
        export_png=export_png
    )
    print("✓")
    
    print("  Creating day-of-week comparison...", end=" ")
    fig_dow = create_dow_comparison(dow_df)
    save_dual(
        fig_dow,
        str(viz_dir / "viz_13_ontime_by_dow_1993_vs_2003.plotly.json"),
        str(fig_dir / "fig_13_ontime_by_dow_1993_vs_2003.png"),
        export_png=export_png
    )
    print("✓")
    
    print("  Creating hourly comparison...", end=" ")
    fig_hourly = create_hourly_comparison(hourly_df)
    save_dual(
        fig_hourly,
        str(viz_dir / "viz_14_ontime_by_dep_hour_1993_vs_2003.plotly.json"),
        str(fig_dir / "fig_14_ontime_by_dep_hour_1993_vs_2003.png"),
        export_png=export_png
    )
    print("✓")
    
    print("  Creating top 10 carriers 1993...", end=" ")
    fig_carrier_1993 = create_carrier_chart(carrier_df, 1993)
    save_dual(
        fig_carrier_1993,
        str(viz_dir / "viz_15_top10_carriers_1993.plotly.json"),
        str(fig_dir / "fig_15_top10_carriers_1993.png"),
        export_png=export_png
    )
    print("✓")
    
    print("  Creating top 10 carriers 2003...", end=" ")
    fig_carrier_2003 = create_carrier_chart(carrier_df, 2003)
    save_dual(
        fig_carrier_2003,
        str(viz_dir / "viz_16_top10_carriers_2003.plotly.json"),
        str(fig_dir / "fig_16_top10_carriers_2003.png"),
        export_png=export_png
    )
    print("✓")
    
    print("  Creating route-matched comparison...", end=" ")
    fig_route_matched = create_route_matched_chart(route_matched_df)
    save_dual(
        fig_route_matched,
        str(viz_dir / "viz_17_route_matched_comparison.plotly.json"),
        str(fig_dir / "fig_17_route_matched_comparison.png"),
        export_png=export_png
    )
    print("✓")
    
    print("  Creating airport-matched comparison...", end=" ")
    fig_airport_matched = create_airport_matched_chart(airport_matched_df)
    save_dual(
        fig_airport_matched,
        str(viz_dir / "viz_18_airport_matched_comparison.plotly.json"),
        str(fig_dir / "fig_18_airport_matched_comparison.png"),
        export_png=export_png
    )
    print("✓")
    
    # Summary
    print("\n" + "=" * 60)
    print("EDA Summary:")
    print("=" * 60)
    print(f"Core KPIs: {len(kpi_df)} years")
    print(f"Monthly analysis: {len(monthly_df)} records")
    print(f"Day-of-week analysis: {len(dow_df)} records")
    print(f"Hourly analysis: {len(hourly_df)} records")
    print(f"Carriers: {len(carrier_df)} records")
    print(f"Routes: {len(route_df)} records")
    print(f"Route-matched: {len(route_matched_df)} records")
    print(f"Airport-matched: {len(airport_matched_df)} records")
    print(f"\n✓ Stage 04 completed successfully!")
    print("=" * 60)
    
    conn.close()


if __name__ == "__main__":
    main()
