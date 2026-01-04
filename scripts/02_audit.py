#!/usr/bin/env python3
"""
Stage 02: Audit missingness, validity, comparability

This script performs comprehensive data quality audits:
1. Missingness analysis by column and year
2. Range checks (invalid times, distances, dates)
3. Cancellation and diversion rates
4. Feature availability matrix
5. Generates visualizations for all audit results
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
from src.flight_delay.audit import (
    calculate_missingness,
    perform_range_checks,
    calculate_cancel_divert_rates,
    create_availability_matrix
)
from src.flight_delay.viz_specs import save_dual, save_plotly_json


def main():
    """Main execution function for Stage 02."""
    print("=" * 60)
    print("Stage 02: Audit - Missingness, Validity, Comparability")
    print("=" * 60)
    
    # Load configuration
    print("\n[1/6] Loading configuration...")
    try:
        config = load_config("config/params.yaml")
        export_png = config.get("export_png", True)
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        sys.exit(1)
    
    # Connect to DuckDB
    print("\n[2/6] Connecting to DuckDB...")
    db_path = project_root / "db" / "flights.duckdb"
    conn = get_duckdb_connection(str(db_path))
    print(f"✓ Connected to database")
    
    # Check if raw tables exist
    try:
        conn.execute("SELECT COUNT(*) FROM raw_1993").fetchone()
        conn.execute("SELECT COUNT(*) FROM raw_2003").fetchone()
    except Exception as e:
        print(f"✗ Error: Raw tables not found. Please run Stage 01 first.")
        sys.exit(1)
    
    # 1. Missingness Analysis
    print("\n[3/6] Calculating missingness by column and year...")
    missingness_1993 = calculate_missingness(conn, "raw_1993", 1993)
    missingness_2003 = calculate_missingness(conn, "raw_2003", 2003)
    missingness_df = pd.concat([missingness_1993, missingness_2003], ignore_index=True)
    
    # Save missingness table
    audit_dir = project_root / "outputs" / "tables" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    missingness_path = audit_dir / "tbl_04_missingness_by_column_year.parquet"
    missingness_df.to_parquet(missingness_path, index=False)
    print(f"✓ Saved: {missingness_path}")
    
    # 2. Range Checks
    print("\n[4/6] Performing range checks...")
    range_checks_1993 = perform_range_checks(conn, "raw_1993", 1993)
    range_checks_2003 = perform_range_checks(conn, "raw_2003", 2003)
    range_checks_df = pd.concat([range_checks_1993, range_checks_2003], ignore_index=True)
    
    # Save range checks table
    range_checks_path = audit_dir / "tbl_05_range_checks_summary.csv"
    range_checks_df.to_csv(range_checks_path, index=False)
    print(f"✓ Saved: {range_checks_path}")
    
    # 3. Cancellation/Diversion Rates
    print("\n[5/6] Calculating cancellation and diversion rates...")
    cancel_divert_1993 = calculate_cancel_divert_rates(conn, "raw_1993", 1993)
    cancel_divert_2003 = calculate_cancel_divert_rates(conn, "raw_2003", 2003)
    
    # Convert to DataFrame
    cancel_divert_df = pd.DataFrame([cancel_divert_1993, cancel_divert_2003])
    # Flatten cancel_codes dict
    cancel_divert_df['cancel_codes'] = cancel_divert_df['cancel_codes'].apply(str)
    cancel_divert_path = audit_dir / "tbl_06_cancel_divert_rates.csv"
    cancel_divert_df.to_csv(cancel_divert_path, index=False)
    print(f"✓ Saved: {cancel_divert_path}")
    
    # 4. Feature Availability Matrix
    print("\n[6/6] Creating feature availability matrix...")
    availability_df = create_availability_matrix(conn, "raw_1993", "raw_2003")
    availability_path = audit_dir / "tbl_07_feature_availability_matrix.parquet"
    availability_df.to_parquet(availability_path, index=False)
    print(f"✓ Saved: {availability_path}")
    
    # Generate Visualizations
    print("\n[7/7] Generating visualizations...")
    viz_dir = project_root / "outputs" / "viz" / "audit"
    fig_dir = project_root / "outputs" / "figures" / "audit"
    viz_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualization 1: Missingness Heatmap - Enhanced aesthetics
    print("  Creating missingness heatmap...", end=" ")
    missingness_pivot = missingness_df.pivot(
        index='column_name',
        columns='year',
        values='missing_pct'
    ).fillna(0)
    
    # Create text matrix and prepare scatter overlay for high values with white text
    text_matrix = []
    scatter_x = []
    scatter_y = []
    scatter_text = []
    
    year_cols = missingness_pivot.columns.astype(str).tolist()
    
    for col_name in missingness_pivot.index:
        text_row = []
        for year in missingness_pivot.columns:
            val = missingness_pivot.loc[col_name, year]
            year_str = str(year)
            if val > 50:  # High missingness - hide default text, add to scatter
                text_row.append('')  # Empty string
                scatter_x.append(year_str)
                scatter_y.append(col_name)
                scatter_text.append(f'{val:.1f}%')
            else:  # Low missingness - show black text
                text_row.append(f'{val:.1f}%')
        text_matrix.append(text_row)
    
    fig1 = go.Figure()
    
    # Add heatmap
    fig1.add_trace(go.Heatmap(
        z=missingness_pivot.values,
        x=missingness_pivot.columns.astype(str),
        y=missingness_pivot.index,
        colorscale=[
            [0, '#E3F2FD'],      # Very light blue (0%)
            [0.2, '#BBDEFB'],    # Light blue (20%)
            [0.4, '#90CAF9'],    # Medium blue (40%)
            [0.5, '#64B5F6'],    # Blue (50%)
            [0.7, '#42A5F5'],    # Darker blue (70%)
            [1, '#1976D2']       # Dark blue (100%)
        ],
        text=text_matrix,
        texttemplate='%{text}',
        textfont=dict(size=10, family='Arial Black', color='#1a1a1a'),  # Black text
        colorbar=dict(
            title=dict(text="Missing %", font=dict(size=13, family='Arial Black')),
            title_font=dict(size=13, family='Arial Black'),
            tickfont=dict(size=11, family='Arial'),
            thickness=20,
            len=0.7,
            x=1.02
        ),
        hovertemplate='<b>%{y}</b><br>Year: %{x}<br>Missing: %{z:.2f}%<extra></extra>',
        showscale=True
    ))
    
    # Add scatter overlay for white text on dark cells
    if scatter_x:
        fig1.add_trace(go.Scatter(
            x=scatter_x,
            y=scatter_y,
            mode='text',
            text=scatter_text,
            textfont=dict(size=10, family='Arial Black', color='white'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig1.update_layout(
        title=dict(
            text="Data Completeness Analysis<br><sub>Missing Values by Column and Year</sub>",
            x=0.5,
            font=dict(size=22, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="Year", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            showgrid=False,
            side='top'
        ),
        yaxis=dict(
            title=dict(text="Column Name", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=10, family='Arial'),
            showgrid=False
        ),
        template="plotly_white",
        height=max(700, len(missingness_pivot) * 22),
        font=dict(family="Arial", size=11),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=150, r=100, t=100, b=50)
    )
    
    save_dual(
        fig1,
        str(viz_dir / "viz_03_missingness_heatmap.plotly.json"),
        str(fig_dir / "fig_03_missingness_heatmap.png"),
        export_png=export_png,
        height=max(600, len(missingness_pivot) * 20)
    )
    
    # Save individual component
    viz1_dir = viz_dir / "viz_03_missingness_heatmap"
    viz1_dir.mkdir(exist_ok=True)
    save_plotly_json(fig1, str(viz1_dir / "missingness_heatmap.plotly.json"))
    print("✓")
    
    # Visualization 2: Cancel/Divert Rates - Enhanced aesthetics
    print("  Creating cancel/divert rates chart...", end=" ")
    fig2 = go.Figure()
    
    years = cancel_divert_df['year'].astype(str)
    cancel_pcts = cancel_divert_df['cancelled_pct']
    divert_pcts = cancel_divert_df['diverted_pct']
    
    # Calculate max for y-axis
    max_rate = max(cancel_pcts.max(), divert_pcts.max())
    y_padding = max_rate * 0.25
    
    # Use consistent colors: same color for same metric across years
    cancel_color = '#2E86AB'  # Blue for cancellations
    divert_color = '#F18F01'  # Orange for diversions
    
    fig2.add_trace(go.Bar(
        name='Cancellation Rate',
        x=years,
        y=cancel_pcts,
        marker=dict(
            color=cancel_color,  # Single color for all cancellation bars
            line=dict(color='white', width=2.5)
        ),
        text=[f"{val:.2f}%" for val in cancel_pcts],
        textposition="outside",
        textfont=dict(size=15, color='#1a1a1a', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>Cancellation Rate: %{y:.2f}%<br>Count: %{customdata}<extra></extra>',
        customdata=cancel_divert_df['cancelled_count'],
        width=0.4
    ))
    
    fig2.add_trace(go.Bar(
        name='Diversion Rate',
        x=years,
        y=divert_pcts,
        marker=dict(
            color=divert_color,  # Single color for all diversion bars
            line=dict(color='white', width=2.5)
        ),
        text=[f"{val:.2f}%" for val in divert_pcts],
        textposition="outside",
        textfont=dict(size=15, color='#1a1a1a', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>Diversion Rate: %{y:.2f}%<br>Count: %{customdata}<extra></extra>',
        customdata=cancel_divert_df['diverted_count'],
        width=0.4
    ))
    
    fig2.update_layout(
        title=dict(
            text="Flight Disruption Analysis<br><sub>Cancellation and Diversion Rates by Year</sub>",
            x=0.5,
            font=dict(size=22, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="Year", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=13, family='Arial'),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text="Rate (%)", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            range=[0, max_rate + y_padding],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.08)',
            gridwidth=1
        ),
        template="plotly_white",
        height=600,
        barmode='group',
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13, family='Arial Black'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.15)',
            borderwidth=1.5
        ),
        margin=dict(l=60, r=40, t=100, b=60)
    )
    
    save_dual(
        fig2,
        str(viz_dir / "viz_04_cancel_divert_rates_by_year.plotly.json"),
        str(fig_dir / "fig_04_cancel_divert_rates_by_year.png"),
        export_png=export_png
    )
    
    # Save individual component
    viz2_dir = viz_dir / "viz_04_cancel_divert_rates_by_year"
    viz2_dir.mkdir(exist_ok=True)
    save_plotly_json(fig2, str(viz2_dir / "cancel_divert_rates.plotly.json"))
    print("✓")
    
    # Visualization 3: ArrDelay Distribution - Full dataset using aggregation
    print("  Creating ArrDelay distribution chart...", end=" ")
    
    # First, find the actual min and max delays in the data
    min_max_1993 = conn.execute(
        "SELECT MIN(ArrDelay), MAX(ArrDelay) FROM raw_1993 WHERE ArrDelay IS NOT NULL"
    ).fetchone()
    min_max_2003 = conn.execute(
        "SELECT MIN(ArrDelay), MAX(ArrDelay) FROM raw_2003 WHERE ArrDelay IS NOT NULL"
    ).fetchone()
    
    # Get the overall min and max across both years
    min_delay = min(min_max_1993[0] if min_max_1993[0] else 0, min_max_2003[0] if min_max_2003[0] else 0)
    max_delay = max(min_max_1993[1] if min_max_1993[1] else 0, min_max_2003[1] if min_max_2003[1] else 0)
    
    # Round to nice bin boundaries
    min_delay = int(min_delay // 10) * 10 - 10  # Round down to nearest 10, then subtract 10
    max_delay = int(max_delay // 10) * 10 + 10  # Round up to nearest 10, then add 10
    
    bin_width = 5
    bins = list(range(min_delay, max_delay + bin_width, bin_width))
    
    # Get histogram data using aggregation for 1993
    bin_centers_1993 = []
    counts_1993 = []
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        count = conn.execute(
            f"""
            SELECT COUNT(*) 
            FROM raw_1993 
            WHERE ArrDelay IS NOT NULL 
            AND ArrDelay >= {bin_start} 
            AND ArrDelay < {bin_end}
            """
        ).fetchone()[0]
        if count > 0:
            bin_centers_1993.append((bin_start + bin_end) / 2)
            counts_1993.append(count)
    
    # Get histogram data using aggregation for 2003
    bin_centers_2003 = []
    counts_2003 = []
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        count = conn.execute(
            f"""
            SELECT COUNT(*) 
            FROM raw_2003 
            WHERE ArrDelay IS NOT NULL 
            AND ArrDelay >= {bin_start} 
            AND ArrDelay < {bin_end}
            """
        ).fetchone()[0]
        if count > 0:
            bin_centers_2003.append((bin_start + bin_end) / 2)
            counts_2003.append(count)
    
    # Calculate statistics for full dataset
    stats_1993 = conn.execute(
        "SELECT AVG(ArrDelay), PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ArrDelay) FROM raw_1993 WHERE ArrDelay IS NOT NULL"
    ).fetchone()
    mean_1993 = stats_1993[0] if stats_1993[0] else 0
    median_1993 = stats_1993[1] if stats_1993[1] else 0
    
    stats_2003 = conn.execute(
        "SELECT AVG(ArrDelay), PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ArrDelay) FROM raw_2003 WHERE ArrDelay IS NOT NULL"
    ).fetchone()
    mean_2003 = stats_2003[0] if stats_2003[0] else 0
    median_2003 = stats_2003[1] if stats_2003[1] else 0
    
    # Convert to probability density
    total_1993 = sum(counts_1993)
    total_2003 = sum(counts_2003)
    density_1993 = [c / (total_1993 * bin_width) for c in counts_1993]
    density_2003 = [c / (total_2003 * bin_width) for c in counts_2003]
    
    fig3 = go.Figure()
    
    # Create histogram for 1993
    fig3.add_trace(go.Bar(
        x=bin_centers_1993,
        y=density_1993,
        name='1993',
        opacity=0.75,
        marker=dict(
            color='#2E86AB',
            line=dict(color='#1a5f7a', width=1.5)
        ),
        hovertemplate='<b>1993</b><br>Delay: %{x:.0f} min<br>Density: %{y:.4f}<br>Count: %{customdata}<extra></extra>',
        customdata=counts_1993,
        width=bin_width * 0.9
    ))
    
    # Create histogram for 2003
    fig3.add_trace(go.Bar(
        x=bin_centers_2003,
        y=density_2003,
        name='2003',
        opacity=0.75,
        marker=dict(
            color='#A23B72',
            line=dict(color='#7a2a54', width=1.5)
        ),
        hovertemplate='<b>2003</b><br>Delay: %{x:.0f} min<br>Density: %{y:.4f}<br>Count: %{customdata}<extra></extra>',
        customdata=counts_2003,
        width=bin_width * 0.9
    ))
    
    # Add vertical lines for means
    fig3.add_vline(
        x=mean_1993,
        line_dash="dash",
        line_color="#2E86AB",
        opacity=0.8,
        line_width=2,
        annotation_text=f"1993 Mean: {mean_1993:.1f} min",
        annotation_position="top right",
        annotation_font=dict(size=11, color='#2E86AB', family='Arial Black'),
        annotation_bgcolor='rgba(46, 134, 171, 0.1)',
        annotation_bordercolor='#2E86AB'
    )
    fig3.add_vline(
        x=mean_2003,
        line_dash="dash",
        line_color="#A23B72",
        opacity=0.8,
        line_width=2,
        annotation_text=f"2003 Mean: {mean_2003:.1f} min",
        annotation_position="top left",
        annotation_font=dict(size=11, color='#A23B72', family='Arial Black'),
        annotation_bgcolor='rgba(162, 59, 114, 0.1)',
        annotation_bordercolor='#A23B72'
    )
    
    fig3.update_layout(
        title=dict(
            text="Arrival Delay Distribution Comparison<br><sub>Full Dataset: 1993 vs 2003 (All Flights)</sub>",
            x=0.5,
            font=dict(size=22, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="Arrival Delay (minutes)", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.08)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.3)',
            zerolinewidth=2
        ),
        yaxis=dict(
            title=dict(text="Probability Density", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.08)'
        ),
        template="plotly_white",
        height=650,
        barmode='overlay',
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13, family='Arial Black'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.15)',
            borderwidth=1.5
        ),
        margin=dict(l=70, r=40, t=120, b=60)
    )
    
    save_dual(
        fig3,
        str(viz_dir / "viz_05_arrdelay_raw_distribution_1993_vs_2003.plotly.json"),
        str(fig_dir / "fig_05_arrdelay_raw_distribution_1993_vs_2003.png"),
        export_png=export_png
    )
    
    # Save individual component
    viz3_dir = viz_dir / "viz_05_arrdelay_raw_distribution_1993_vs_2003"
    viz3_dir.mkdir(exist_ok=True)
    save_plotly_json(fig3, str(viz3_dir / "arrdelay_distribution.plotly.json"))
    print("✓")
    
    # Summary
    print("\n" + "=" * 60)
    print("Audit Summary:")
    print("=" * 60)
    
    # Missingness summary
    high_missing = missingness_df[missingness_df['missing_pct'] > 50]
    if len(high_missing) > 0:
        print(f"\n⚠ High missingness (>50%): {len(high_missing)} columns")
        for _, row in high_missing.iterrows():
            print(f"  - {row['column_name']} ({row['year']}): {row['missing_pct']:.1f}%")
    
    # Range checks summary
    invalid_checks = range_checks_df[range_checks_df['invalid_pct'] > 0]
    if len(invalid_checks) > 0:
        print(f"\n⚠ Invalid data found: {len(invalid_checks)} checks")
        for _, row in invalid_checks.head(10).iterrows():
            print(f"  - {row['check_name']} ({row['column']}, {row['year']}): {row['invalid_pct']:.2f}%")
    
    # Cancel/divert summary
    print(f"\nCancellation Rates:")
    for _, row in cancel_divert_df.iterrows():
        print(f"  {row['year']}: {row['cancelled_pct']:.2f}% cancelled, {row['diverted_pct']:.2f}% diverted")
    
    print("\n✓ Stage 02 completed successfully!")
    print("=" * 60)
    
    # Close connection
    conn.close()

if __name__ == "__main__":
    main()
