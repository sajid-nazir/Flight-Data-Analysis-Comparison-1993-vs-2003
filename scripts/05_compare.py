#!/usr/bin/env python3
"""
Stage 05: Compare 1993 vs 2003 with standardized deltas

This script:
1. Computes weighted overall delta (flight-weighted)
2. Computes per-dimension delta tables (carrier, origin, destination, route, month, dep hour)
3. Creates "delta vs volume" tables for safe interpretation
4. Generates visualizations showing improvements/declines
"""
import sys
from pathlib import Path
from typing import Dict
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
from src.flight_delay.compare import (
    compute_overall_weighted_delta,
    compute_delta_by_dimension,
    compute_delta_by_month,
    compute_delta_by_dep_hour
)
from src.flight_delay.viz_specs import save_dual, save_plotly_json


def create_overall_delta_summary(delta_dict: Dict) -> go.Figure:
    """Create overall delta summary visualization."""
    fig = go.Figure()
    
    years = ['1993', '2003']
    values = [delta_dict['value_1993'], delta_dict['value_2003']]
    colors = ['#2E86AB', '#A23B72']
    
    fig.add_trace(go.Bar(
        x=years,
        y=values,
        marker_color=colors,
        marker_line=dict(color='white', width=2),
        text=[f"{val:.2f}%" for val in values],
        textposition='outside',
        textfont=dict(size=14, family='Arial Black', color='#1a1a1a'),
        hovertemplate='<b>%{x}</b><br>On-Time Rate: %{y:.2f}%<br>Total Flights: %{customdata:,}<extra></extra>',
        customdata=[delta_dict['total_flights_1993'], delta_dict['total_flights_2003']],
        name='On-Time Rate'
    ))
    
    # Add delta annotation - better positioned and clearer
    delta_val = delta_dict['delta_absolute']
    delta_pct = delta_dict['delta_percent']
    # Clarify: delta_val is absolute change in percentage points, delta_pct is relative percentage change
    annotation_text = f"Change: {delta_val:+.2f} pp<br>({delta_pct:+.1f}% relative)"
    
    # Position annotation above bars, centered
    fig.add_annotation(
        x=0.5,  # Center between bars
        y=max(values) * 1.15,  # Above bars
        xref='paper',
        yref='y',
        text=annotation_text,
        showarrow=False,
        font=dict(size=16, family='Arial Black', color='#F18F01'),
        bgcolor='rgba(255,255,255,0.95)',
        bordercolor='#F18F01',
        borderwidth=2,
        borderpad=8
    )
    
    fig.update_layout(
        title=dict(
            text="Overall On-Time Performance Comparison<br><sub>1993 vs 2003 (Flight-Weighted)</sub>",
            x=0.5,
            font=dict(size=22, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="Year", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=13, family='Arial'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title=dict(text="On-Time Rate (%)", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            range=[min(values) * 0.95, max(values) * 1.25],  # Make room for annotation
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        template="plotly_white",
        height=600,
        showlegend=False,
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=120, b=60)
    )
    
    return fig


def create_delta_vs_volume_chart(delta_df: pd.DataFrame, dimension_col: str, title: str) -> go.Figure:
    """Create delta vs volume scatter chart."""
    if len(delta_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure()
    
    # Separate improvements and declines for better visualization
    improvements = delta_df[delta_df['delta_absolute'] > 0].copy()
    declines = delta_df[delta_df['delta_absolute'] < 0].copy()
    neutral = delta_df[delta_df['delta_absolute'] == 0].copy()
    
    # Improvements (positive delta)
    if len(improvements) > 0:
        fig.add_trace(go.Scatter(
            x=improvements['total_flights_both'],
            y=improvements['delta_absolute'],
            mode='markers',
            marker=dict(
                size=10,
                color='#2E86AB',
                line=dict(color='white', width=1.5),
                opacity=0.8,
                symbol='circle'
            ),
            text=improvements[dimension_col],
            hovertemplate='<b>%{text}</b><br>Total Flights: %{x:,}<br>Improvement: +%{y:.2f}%<extra></extra>',
            name='Improvement',
            showlegend=True
        ))
    
    # Declines (negative delta)
    if len(declines) > 0:
        fig.add_trace(go.Scatter(
            x=declines['total_flights_both'],
            y=declines['delta_absolute'],
            mode='markers',
            marker=dict(
                size=10,
                color='#E74C3C',
                line=dict(color='white', width=1.5),
                opacity=0.8,
                symbol='diamond'
            ),
            text=declines[dimension_col],
            hovertemplate='<b>%{text}</b><br>Total Flights: %{x:,}<br>Decline: %{y:.2f}%<extra></extra>',
            name='Decline',
            showlegend=True
        ))
    
    # Neutral (zero delta)
    if len(neutral) > 0:
        fig.add_trace(go.Scatter(
            x=neutral['total_flights_both'],
            y=neutral['delta_absolute'],
            mode='markers',
            marker=dict(
                size=8,
                color='#95A5A6',
                line=dict(color='white', width=1),
                opacity=0.6,
                symbol='square'
            ),
            text=neutral[dimension_col],
            hovertemplate='<b>%{text}</b><br>Total Flights: %{x:,}<br>No Change<extra></extra>',
            name='No Change',
            showlegend=True
        ))
    
    # Add prominent zero line
    max_delta = abs(delta_df['delta_absolute']).max() if len(delta_df) > 0 else 5
    fig.add_hline(
        y=0, 
        line_dash="dash", 
        line_color="#666666", 
        line_width=2,
        opacity=0.8,
        annotation_text="No Change",
        annotation_position="right",
        annotation_font=dict(size=11, family='Arial', color='#666666')
    )
    
    # Calculate proper x-axis range for log scale to show all data
    if len(delta_df) > 0:
        min_flights = delta_df['total_flights_both'].min()
        max_flights = delta_df['total_flights_both'].max()
        # For log scale, set range to cover all data with some padding
        # Use log10 values for range
        log_min = np.log10(max(1, min_flights * 0.5))  # Add padding below
        log_max = np.log10(max_flights * 2.0)  # Add padding above
    else:
        log_min = 0
        log_max = 6  # Default to 1M
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="Total Flights (Both Years, Log Scale)", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            type='log',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            range=[log_min, log_max]  # Explicit range to show all data
        ),
        yaxis=dict(
            title=dict(text="Delta (2003 - 1993) %", font=dict(size=14, family='Arial Black')),
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


def create_dumbbell_chart(delta_df: pd.DataFrame, dimension_col: str, title: str, top_n: int = 10) -> go.Figure:
    """Create dumbbell chart showing 1993 vs 2003 for top N items."""
    if len(delta_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Sort by total volume and take top N
    top_df = delta_df.nlargest(top_n, 'total_flights_both').copy()
    if len(top_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    top_df = top_df.sort_values('delta_absolute', ascending=True)
    
    # Ensure we have the required columns
    if 'ontime_rate_pct_1993' not in top_df.columns or 'ontime_rate_pct_2003' not in top_df.columns:
        # Try to compute from available columns
        if 'value_1993' in top_df.columns and 'value_2003' in top_df.columns:
            top_df['ontime_rate_pct_1993'] = top_df['value_1993']
            top_df['ontime_rate_pct_2003'] = top_df['value_2003']
        else:
            fig = go.Figure()
            fig.add_annotation(text="Missing required data columns", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
    
    fig = go.Figure()
    
    # Add connecting lines first (so they're behind markers)
    for idx, row in top_df.iterrows():
        val_1993 = row.get('ontime_rate_pct_1993', 0)
        val_2003 = row.get('ontime_rate_pct_2003', 0)
        if pd.notna(val_1993) and pd.notna(val_2003):
            fig.add_trace(go.Scatter(
                x=[val_1993, val_2003],
                y=[row[dimension_col], row[dimension_col]],
                mode='lines',
                line=dict(color='gray', width=2, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add 1993 values
    fig.add_trace(go.Scatter(
        x=top_df['ontime_rate_pct_1993'],
        y=top_df[dimension_col],
        mode='markers',
        marker=dict(size=14, color='#2E86AB', symbol='circle', line=dict(color='white', width=2)),
        name='1993',
        hovertemplate='<b>%{y}</b><br>1993: %{x:.1f}%<br>Flights: %{customdata:,}<extra></extra>',
        customdata=top_df.get('total_flights_1993', top_df.get('total_flights_both', [0] * len(top_df)))
    ))
    
    # Add 2003 values
    fig.add_trace(go.Scatter(
        x=top_df['ontime_rate_pct_2003'],
        y=top_df[dimension_col],
        mode='markers',
        marker=dict(size=14, color='#A23B72', symbol='diamond', line=dict(color='white', width=2)),
        name='2003',
        hovertemplate='<b>%{y}</b><br>2003: %{x:.1f}%<br>Flights: %{customdata:,}<br>Delta: %{text:+.1f}%<extra></extra>',
        customdata=top_df.get('total_flights_2003', top_df.get('total_flights_both', [0] * len(top_df))),
        text=top_df.get('delta_absolute', [0] * len(top_df))
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="On-Time Rate (%)", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title=dict(text=dimension_col, font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=11, family='Arial'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        template="plotly_white",
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=12, family='Arial Black')),
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=120, r=40, t=100, b=60)
    )
    
    return fig


def create_improvements_declines_chart(delta_df: pd.DataFrame, dimension_col: str, title: str, top_n: int = 15) -> go.Figure:
    """Create chart showing top improvements and declines."""
    if len(delta_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Separate improvements and declines
    improvements = delta_df.nlargest(top_n, 'delta_absolute').copy()
    declines = delta_df.nsmallest(top_n, 'delta_absolute').copy()
    
    # Sort for better visualization
    improvements = improvements.sort_values('delta_absolute', ascending=True)
    declines = declines.sort_values('delta_absolute', ascending=True)
    
    fig = go.Figure()
    
    # Improvements (positive delta) - only if we have improvements
    if len(improvements) > 0 and improvements['delta_absolute'].max() > 0:
        improvements_positive = improvements[improvements['delta_absolute'] > 0]
        if len(improvements_positive) > 0:
            fig.add_trace(go.Bar(
                y=improvements_positive[dimension_col],
                x=improvements_positive['delta_absolute'],
                orientation='h',
                marker_color='#2E86AB',
                marker_line=dict(color='white', width=1),
                name='Improvement',
                text=[f"+{val:.1f}%" for val in improvements_positive['delta_absolute']],
                textposition='outside',
                textfont=dict(size=10, family='Arial Black'),
                hovertemplate='<b>%{y}</b><br>Improvement: +%{x:.2f}%<br>Flights: %{customdata:,}<extra></extra>',
                customdata=improvements_positive.get('total_flights_both', [0] * len(improvements_positive))
            ))
    
    # Declines (negative delta) - only if we have declines
    if len(declines) > 0 and declines['delta_absolute'].min() < 0:
        declines_negative = declines[declines['delta_absolute'] < 0]
        if len(declines_negative) > 0:
            fig.add_trace(go.Bar(
                y=declines_negative[dimension_col],
                x=declines_negative['delta_absolute'],
                orientation='h',
                marker_color='#E74C3C',
                marker_line=dict(color='white', width=1),
                name='Decline',
                text=[f"{val:.1f}%" for val in declines_negative['delta_absolute']],
                textposition='outside',
                textfont=dict(size=10, family='Arial Black'),
                hovertemplate='<b>%{y}</b><br>Decline: %{x:.2f}%<br>Flights: %{customdata:,}<extra></extra>',
                customdata=declines_negative.get('total_flights_both', [0] * len(declines_negative))
            ))
    
    # Add zero line
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5, line_width=1)
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="Delta (2003 - 1993) %", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title=dict(text=dimension_col, font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=11, family='Arial'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        template="plotly_white",
        height=800,
        barmode='group',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=12, family='Arial Black')),
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=120, r=40, t=100, b=60)
    )
    
    return fig


def main():
    """Main execution function for Stage 05."""
    print("=" * 60)
    print("Stage 05: Compare 1993 vs 2003 with Standardized Deltas")
    print("=" * 60)
    
    # Load configuration
    print("\n[1/7] Loading configuration...")
    try:
        config = load_config("config/params.yaml")
        export_png = config.get("export_png", True)
        min_group_volume = config.get("min_group_volume", 10000)
        on_time_threshold = config.get("on_time_threshold_min", 15)
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        sys.exit(1)
    
    # Connect to DuckDB
    print("\n[2/7] Connecting to DuckDB...")
    db_path = project_root / "db" / "flights.duckdb"
    conn = get_duckdb_connection(str(db_path))
    print(f"✓ Connected to database")
    
    # Load EDA summary tables
    print("\n[3/7] Loading EDA summary tables...")
    eda_dir = project_root / "outputs" / "tables" / "eda"
    
    try:
        kpi_df = pd.read_csv(eda_dir / "tbl_10_core_kpis_by_year.csv")
        monthly_df = pd.read_parquet(eda_dir / "tbl_11_ontime_by_month.parquet")
        dow_df = pd.read_parquet(eda_dir / "tbl_12_ontime_by_dow.parquet")
        hourly_df = pd.read_parquet(eda_dir / "tbl_13_ontime_by_dep_hour.parquet")
        carrier_df = pd.read_parquet(eda_dir / "tbl_14_carrier_summary.parquet")
        origin_df = pd.read_parquet(eda_dir / "tbl_15_origin_airport_summary.parquet")
        dest_df = pd.read_parquet(eda_dir / "tbl_16_dest_airport_summary.parquet")
        route_df = pd.read_parquet(eda_dir / "tbl_17_route_summary.parquet")
        print("✓ All EDA tables loaded")
    except Exception as e:
        print(f"✗ Error loading EDA tables: {e}")
        sys.exit(1)
    
    # Create output directories
    compare_dir = project_root / "outputs" / "tables" / "compare"
    viz_dir = project_root / "outputs" / "viz" / "compare"
    fig_dir = project_root / "outputs" / "figures" / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute overall weighted delta
    print("\n[4/7] Computing deltas...")
    overall_delta = compute_overall_weighted_delta(kpi_df, 'ontime_rate_pct')
    
    overall_delta_df = pd.DataFrame([overall_delta])
    overall_delta_df.to_csv(compare_dir / "tbl_18_overall_weighted_delta.csv", index=False)
    print(f"  ✓ Overall weighted delta computed")
    
    # Compute per-dimension deltas
    delta_carrier = compute_delta_by_dimension(carrier_df, 'UniqueCarrier', 'ontime_rate_pct', min_group_volume)
    delta_origin = compute_delta_by_dimension(origin_df, 'airport', 'ontime_rate_pct', min_group_volume)
    delta_dest = compute_delta_by_dimension(dest_df, 'airport', 'ontime_rate_pct', min_group_volume)
    delta_route = compute_delta_by_dimension(route_df, 'route', 'ontime_rate_pct', min_group_volume)
    delta_month = compute_delta_by_month(monthly_df, 'ontime_rate_pct')
    delta_hour = compute_delta_by_dep_hour(hourly_df, 'ontime_rate_pct')
    
    # Save delta tables
    delta_carrier.to_parquet(compare_dir / "tbl_19_delta_by_carrier.parquet", index=False)
    delta_origin.to_parquet(compare_dir / "tbl_20_delta_by_origin_airport.parquet", index=False)
    delta_dest.to_parquet(compare_dir / "tbl_21_delta_by_dest_airport.parquet", index=False)
    delta_route.to_parquet(compare_dir / "tbl_22_delta_by_route.parquet", index=False)
    delta_month.to_parquet(compare_dir / "tbl_23_delta_by_month.parquet", index=False)
    delta_hour.to_parquet(compare_dir / "tbl_24_delta_by_dep_hour.parquet", index=False)
    
    print(f"  ✓ Per-dimension deltas computed")
    print(f"    - Carriers: {len(delta_carrier)}")
    print(f"    - Origin airports: {len(delta_origin)}")
    print(f"    - Destination airports: {len(delta_dest)}")
    print(f"    - Routes: {len(delta_route)}")
    print(f"    - Months: {len(delta_month)}")
    print(f"    - Departure hours: {len(delta_hour)}")
    
    # Generate visualizations
    print("\n[5/7] Generating visualizations...")
    
    # Overall delta summary
    print("  Creating overall delta summary...", end=" ")
    fig = create_overall_delta_summary(overall_delta)
    save_dual(
        fig,
        str(viz_dir / "viz_21_overall_delta_summary.plotly.json"),
        str(fig_dir / "fig_21_overall_delta_summary.png"),
        export_png=export_png
    )
    print("✓")
    
    # Delta vs volume - carriers
    print("  Creating delta vs volume (carriers)...", end=" ")
    fig = create_delta_vs_volume_chart(
        delta_carrier,
        'UniqueCarrier',
        "Carrier Performance Delta vs Volume<br><sub>On-Time Rate Change (2003 - 1993)</sub>"
    )
    save_dual(
        fig,
        str(viz_dir / "viz_22_delta_vs_volume_carrier.plotly.json"),
        str(fig_dir / "fig_22_delta_vs_volume_carrier.png"),
        export_png=export_png
    )
    print("✓")
    
    # Dumbbell chart - top 10 carriers
    print("  Creating dumbbell chart (top 10 carriers)...", end=" ")
    fig = create_dumbbell_chart(
        delta_carrier,
        'UniqueCarrier',
        "Top 10 Carriers: 1993 vs 2003 On-Time Performance<br><sub>Dumbbell Comparison</sub>",
        top_n=10
    )
    save_dual(
        fig,
        str(viz_dir / "viz_23_dumbbell_top10_carriers.plotly.json"),
        str(fig_dir / "fig_23_dumbbell_top10_carriers.png"),
        export_png=export_png
    )
    print("✓")
    
    # Top improvements and declines - carriers
    print("  Creating improvements/declines chart (carriers)...", end=" ")
    fig = create_improvements_declines_chart(
        delta_carrier,
        'UniqueCarrier',
        "Top 15 Carrier Improvements and Declines<br><sub>On-Time Rate Change (2003 - 1993)</sub>",
        top_n=15
    )
    save_dual(
        fig,
        str(viz_dir / "viz_27_top15_improvements_declines_carriers.plotly.json"),
        str(fig_dir / "fig_27_top15_improvements_declines_carriers.png"),
        export_png=export_png
    )
    print("✓")
    
    # Summary
    print("\n[6/7] Comparison Summary:")
    print("=" * 60)
    print(f"Overall On-Time Rate Delta: {overall_delta['delta_absolute']:+.2f}% ({overall_delta['delta_percent']:+.1f}%)")
    print(f"  - 1993: {overall_delta['value_1993']:.2f}%")
    print(f"  - 2003: {overall_delta['value_2003']:.2f}%")
    print(f"\nDimensions analyzed:")
    print(f"  - Carriers: {len(delta_carrier)} (min volume: {min_group_volume:,})")
    print(f"  - Origin airports: {len(delta_origin)}")
    print(f"  - Destination airports: {len(delta_dest)}")
    print(f"  - Routes: {len(delta_route)}")
    print(f"  - Months: {len(delta_month)}")
    print(f"  - Departure hours: {len(delta_hour)}")
    
    print("\n✓ Stage 05 completed successfully!")
    print("=" * 60)
    
    # Close connection
    conn.close()


if __name__ == "__main__":
    main()
