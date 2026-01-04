#!/usr/bin/env python3
"""
Stage 06: Build feature tables for modeling

This script:
1. Builds target variable (ontime15)
2. Builds features from clean_common data
3. Creates train/test splits by month
4. Fits target encoders (mean encoding) on train only
5. Applies encoders and writes feature tables

Note: Uses target encoding (mean of ontime15 per category), not frequency encoding.
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
from src.flight_delay.io_duckdb import get_duckdb_connection, write_partitioned_parquet
from src.flight_delay.io_artifacts import save_json
from src.flight_delay.features import (
    build_base_features,
    compute_congestion_features,
    create_split_assignments,
    fit_target_encoders,
    apply_target_encoders,
    save_encoders
)
from src.flight_delay.viz_specs import save_dual, save_plotly_json


def create_congestion_distribution_chart(features_1993: pd.DataFrame, features_2003: pd.DataFrame) -> go.Figure:
    """Create congestion feature distribution comparison."""
    fig = go.Figure()
    
    for year, df, color in [(1993, features_1993, '#2E86AB'), (2003, features_2003, '#A23B72')]:
        if 'origin_hourly_volume' in df.columns:
            fig.add_trace(go.Histogram(
                x=df['origin_hourly_volume'],
                name=f'{year} Origin',
                opacity=0.6,
                marker_color=color,
                histnorm='probability density',
                nbinsx=50
            ))
    
    fig.update_layout(
        title=dict(
            text="Congestion Feature Distributions: 1993 vs 2003<br><sub>Origin Hourly Volume</sub>",
            x=0.5,
            font=dict(size=20, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="Origin Hourly Volume", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial')
        ),
        yaxis=dict(
            title=dict(text="Probability Density", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial')
        ),
        template="plotly_white",
        height=600,
        barmode='overlay',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=70, r=40, t=100, b=60)
    )
    
    return fig


def create_distance_ontime_heatmap(features_df: pd.DataFrame, year: int) -> go.Figure:
    """Create distance bins vs on-time rate heatmap."""
    # Check required columns
    required_cols = ['distance_bin', 'dep_hour_bin', 'ontime15']
    if not all(col in features_df.columns for col in required_cols):
        fig = go.Figure()
        fig.add_annotation(text="Missing required columns", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Aggregate by distance bin
    heatmap_data = features_df.groupby(['distance_bin', 'dep_hour_bin']).agg({
        'ontime15': 'mean',
        'Year': 'count'
    }).reset_index()
    heatmap_data.columns = ['distance_bin', 'dep_hour_bin', 'ontime_rate', 'count']
    
    if len(heatmap_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Pivot for heatmap
    pivot = heatmap_data.pivot(index='distance_bin', columns='dep_hour_bin', values='ontime_rate')
    count_pivot = heatmap_data.pivot(index='distance_bin', columns='dep_hour_bin', values='count')
    
    # Fill NaN with 0 for display
    pivot = pivot.fillna(0)
    count_pivot = count_pivot.fillna(0)
    
    # Format text for display (show percentage)
    text_data = pivot.values * 100  # Convert to percentage
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale='RdYlGn',
        text=text_data,
        texttemplate='%{text:.1f}%',
        textfont=dict(size=9, family='Arial Black', color='white'),
        hovertemplate='<b>Distance: %{y}</b><br>Hour: %{x}<br>On-Time Rate: %{z:.1%}<br>Flights: %{customdata:,.0f}<extra></extra>',
        customdata=count_pivot.values,
        colorbar=dict(
            title=dict(text="On-Time Rate", font=dict(size=12, family='Arial Black')),
            tickfont=dict(size=11, family='Arial'),
            tickformat='.1%'
        )
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Distance Bins vs Departure Hour: On-Time Rate - {year}<br><sub>Heatmap (darker = higher on-time rate)</sub>",
            x=0.5,
            font=dict(size=20, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="Departure Hour Bin", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=11, family='Arial')
        ),
        yaxis=dict(
            title=dict(text="Distance Bin", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=11, family='Arial')
        ),
        template="plotly_white",
        height=600,
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=100, r=40, t=100, b=60)
    )
    
    return fig


def main():
    """Main execution function for Stage 06."""
    print("=" * 60)
    print("Stage 06: Build Feature Tables for Modeling")
    print("=" * 60)
    
    # Load configuration
    print("\n[1/8] Loading configuration...")
    try:
        config = load_config("config/params.yaml")
        export_png = config.get("export_png", True)
        on_time_threshold = config.get("on_time_threshold_min", 15)
        train_months = config.get("train_months", [1, 2, 3, 4, 5, 6, 7, 8, 9])
        test_months = config.get("test_months", [10, 11, 12])
        prediction_moment = config.get("prediction_moment", "scheduled")
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        sys.exit(1)
    
    # Connect to DuckDB
    print("\n[2/8] Connecting to DuckDB...")
    db_path = project_root / "db" / "flights.duckdb"
    conn = get_duckdb_connection(str(db_path))
    print(f"✓ Connected to database")
    
    # Create output directories
    features_dir = project_root / "parquet" / "features"
    encoders_dir = project_root / "encoders"
    tables_dir = project_root / "outputs" / "tables" / "features"
    viz_dir = project_root / "outputs" / "viz" / "features"
    fig_dir = project_root / "outputs" / "figures" / "features"
    
    features_dir.mkdir(parents=True, exist_ok=True)
    encoders_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each year
    print("\n[3/8] Building base features...")
    all_features = {}
    
    for year in [1993, 2003]:
        print(f"\n  Processing {year}...")
        # Parquet files are in nested directories: year=YYYY/Year=YYYY/Month=X/data_*.parquet
        # Use recursive pattern that DuckDB can handle
        parquet_path = str(project_root / "parquet" / "clean" / "common" / f"year={year}" / "**" / "*.parquet")
        
        # Build base features
        print(f"    Building base features...", end=" ")
        base_features = build_base_features(conn, parquet_path, year, on_time_threshold)
        print(f"✓ ({len(base_features):,} rows)")
        
        # Compute congestion features
        print(f"    Computing congestion features...", end=" ")
        congestion = compute_congestion_features(conn, parquet_path)
        
        # Merge congestion features (congestion has 'dep_hour', base_features has 'dep_hour_raw')
        congestion = congestion.rename(columns={'dep_hour': 'dep_hour_raw'})
        base_features = base_features.merge(
            congestion,
            on=['Year', 'Month', 'DayOfWeek', 'dep_hour_raw', 'Origin', 'Dest'],
            how='left'
        )
        base_features['origin_hourly_volume'] = base_features['origin_hourly_volume'].fillna(0)
        base_features['dest_hourly_volume'] = base_features['dest_hourly_volume'].fillna(0)
        print("✓")
        
        # Create split assignments
        print(f"    Creating train/test splits...", end=" ")
        base_features = create_split_assignments(base_features, train_months, test_months)
        print("✓")
        
        all_features[year] = base_features
        
        # Save base features
        output_path = features_dir / f"year={year}" / "features_base.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        base_features.to_parquet(output_path, index=False)
        print(f"    ✓ Saved base features: {output_path}")
        
        # Save split assignments
        split_df = base_features[['Year', 'Month', 'split']].drop_duplicates()
        split_path = tables_dir / f"tbl_25_split_assignments_{year}.parquet"
        split_df.to_parquet(split_path, index=False)
        print(f"    ✓ Saved split assignments: {split_path}")
    
    # Fit target encoders on training data only
    print("\n[4/8] Fitting target encoders (mean encoding)...")
    categorical_cols = ['UniqueCarrier', 'Origin', 'Dest', 'route', 'dep_hour_bin', 'distance_bin']
    all_encoders = {}
    
    for year in [1993, 2003]:
        print(f"  Fitting encoders for {year}...", end=" ")
        train_data = all_features[year][all_features[year]['split'] == 'train']
        encoders = fit_target_encoders(train_data, categorical_cols)
        all_encoders[year] = encoders
        
        # Save encoders
        encoder_path = encoders_dir / f"target_encoders_train_{year}.json"
        save_encoders(encoders, str(encoder_path))
        print(f"✓ Saved: {encoder_path}")
    
    # Apply encoders and create model-ready features
    print("\n[5/8] Applying encoders and creating model-ready features...")
    for year in [1993, 2003]:
        print(f"  Processing {year}...", end=" ")
        features_model = apply_target_encoders(all_features[year], all_encoders[year])
        
        # Select final feature columns (exclude raw categoricals, keep encoded versions)
        # Note: DepTime is excluded because we're predicting at "scheduled" moment
        # CRSDepTime is included as it's known at scheduled time
        feature_cols = [
            'Year', 'Month', 'DayOfWeek', 'split', 'ontime15',
            'CRSDepTime', 'dep_hour_raw', 'Distance', 'CRSElapsedTime',
            'origin_hourly_volume', 'dest_hourly_volume'
        ]
        
        # Add encoded columns
        for col in categorical_cols:
            encoded_col = f"{col}_freq"
            if encoded_col in features_model.columns:
                feature_cols.append(encoded_col)
        
        features_model = features_model[feature_cols]
        
        # Save model-ready features
        output_path = features_dir / f"year={year}" / "features_model.parquet"
        features_model.to_parquet(output_path, index=False)
        print(f"✓ Saved: {output_path}")
    
    # Generate feature summary statistics
    print("\n[6/8] Generating feature summary statistics...")
    for year in [1993, 2003]:
        print(f"  Computing stats for {year}...", end=" ")
        features_model = pd.read_parquet(features_dir / f"year={year}" / "features_model.parquet")
        
        summary = features_model.describe().T
        summary['missing_count'] = features_model.isnull().sum()
        summary['missing_pct'] = (summary['missing_count'] / len(features_model) * 100)
        
        summary_path = tables_dir / f"tbl_27_feature_summary_stats_{year}.csv"
        summary.to_csv(summary_path)
        print(f"✓ Saved: {summary_path}")
    
    # Generate visualizations
    print("\n[7/8] Generating visualizations...")
    
    # Congestion distribution
    print("  Creating congestion distribution chart...", end=" ")
    features_1993 = pd.read_parquet(features_dir / "year=1993" / "features_base.parquet")
    features_2003 = pd.read_parquet(features_dir / "year=2003" / "features_base.parquet")
    
    fig = create_congestion_distribution_chart(features_1993, features_2003)
    save_dual(
        fig,
        str(viz_dir / "viz_29_congestion_feature_distributions_1993_vs_2003.plotly.json"),
        str(fig_dir / "fig_29_congestion_feature_distributions_1993_vs_2003.png"),
        export_png=export_png
    )
    print("✓")
    
    # Distance vs on-time heatmaps
    for year in [1993, 2003]:
        print(f"  Creating distance/ontime heatmap for {year}...", end=" ")
        features_base = pd.read_parquet(features_dir / f"year={year}" / "features_base.parquet")
        fig = create_distance_ontime_heatmap(features_base, year)
        
        viz_num = 30 if year == 1993 else 31
        save_dual(
            fig,
            str(viz_dir / f"viz_{viz_num:02d}_distance_bins_vs_ontime_heatmap_{year}.plotly.json"),
            str(fig_dir / f"fig_{viz_num:02d}_distance_bins_vs_ontime_heatmap_{year}.png"),
            export_png=export_png
        )
        print("✓")
    
    # Summary
    print("\n[8/8] Feature Engineering Summary:")
    print("=" * 60)
    for year in [1993, 2003]:
        features_model = pd.read_parquet(features_dir / f"year={year}" / "features_model.parquet")
        train_count = len(features_model[features_model['split'] == 'train'])
        test_count = len(features_model[features_model['split'] == 'test'])
        ontime_rate = features_model['ontime15'].mean() * 100
        
        # Count only model features (exclude metadata and target)
        # Include Month as it's a useful feature for seasonal patterns
        model_feature_cols = [col for col in features_model.columns 
                             if col not in ['Year', 'split', 'ontime15']]
        
        print(f"\n{year}:")
        print(f"  Total rows: {len(features_model):,}")
        print(f"  Train: {train_count:,} ({train_count/len(features_model)*100:.1f}%)")
        print(f"  Test: {test_count:,} ({test_count/len(features_model)*100:.1f}%)")
        print(f"  On-time rate: {ontime_rate:.2f}%")
        print(f"  Features: {len(model_feature_cols)}")
    
    print("\n✓ Stage 06 completed successfully!")
    print("=" * 60)
    
    # Close connection
    conn.close()


if __name__ == "__main__":
    main()
