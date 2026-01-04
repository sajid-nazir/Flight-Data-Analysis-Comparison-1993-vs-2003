#!/usr/bin/env python3
"""
Stage 09: Interpretability - feature drivers and driver shift

This script:
1. Computes feature importance (gain) from LightGBM models
2. Computes permutation importance on test data
3. Compares feature ranks between 1993 and 2003
4. Computes partial dependence for top features
5. Generates visualizations
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.flight_delay.config import load_config
from src.flight_delay.interpret import (
    get_lgbm_feature_importance,
    compute_permutation_importance,
    compute_partial_dependence,
    compute_shap_values,
    create_shap_summary_data,
    get_shap_dependence_data
)
from src.flight_delay.viz_specs import save_dual, save_plotly_json


def predict_proba_lgbm(model, X: pd.DataFrame) -> np.ndarray:
    """Get predictions from LightGBM model."""
    return model.predict(X.values, num_iteration=model.best_iteration)


def create_shap_summary_plot(shap_summary: pd.DataFrame, year: int) -> go.Figure:
    """Create SHAP summary plot."""
    fig = go.Figure()
    
    # Sort by mean absolute SHAP
    shap_sorted = shap_summary.sort_values('mean_abs_shap', ascending=True)
    
    # Color bars by mean SHAP (positive = red, negative = blue)
    colors = ['#E74C3C' if x > 0 else '#2E86AB' for x in shap_sorted['mean_shap']]
    
    fig.add_trace(go.Bar(
        y=shap_sorted['feature'],
        x=shap_sorted['mean_abs_shap'],
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=1)),
        text=[f"{val:.4f}" for val in shap_sorted['mean_shap']],
        textposition='outside',
        textfont=dict(size=10, family='Arial'),
        hovertemplate='<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<br>Mean SHAP: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"SHAP Summary Plot - {year}<br><sub>Feature Impact on On-Time Prediction</sub>",
            x=0.5,
            font=dict(size=22, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="Mean |SHAP Value|", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial')
        ),
        yaxis=dict(
            title=dict(text="Feature", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=11, family='Arial')
        ),
        template="plotly_white",
        height=600,
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=150, r=40, t=100, b=60)
    )
    
    return fig


def create_importance_chart(importance_df: pd.DataFrame, year: int, top_n: int = 15) -> go.Figure:
    """Create feature importance bar chart."""
    top_features = importance_df.head(top_n).sort_values('importance_gain', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=top_features['feature'],
        x=top_features['importance_gain'],
        orientation='h',
        marker_color='#2E86AB',
        marker_line=dict(color='white', width=2),
        text=[f"{val:.0f}" for val in top_features['importance_gain']],
        textposition='outside',
        textfont=dict(size=11, family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.0f}<br>Rank: %{customdata}<extra></extra>',
        customdata=top_features['rank']
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Top {top_n} Feature Importance (Gain) - {year}<br><sub>LightGBM Feature Importance</sub>",
            x=0.5,
            font=dict(size=20, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="Importance (Gain)", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title=dict(text="Feature", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=11, family='Arial'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        template="plotly_white",
        height=600,
        showlegend=False,
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=150, r=40, t=100, b=60)
    )
    
    return fig


def create_rank_shift_chart(rank_shift_df: pd.DataFrame) -> go.Figure:
    """Create slope chart showing feature rank shifts."""
    if len(rank_shift_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Calculate rank change
    rank_shift_df = rank_shift_df.copy()
    rank_shift_df['rank_change'] = rank_shift_df['rank_2003'] - rank_shift_df['rank_1993']
    rank_shift_df['avg_rank'] = (rank_shift_df['rank_1993'] + rank_shift_df['rank_2003']) / 2
    
    # Get top 15 features by average rank (reduced from 20 to reduce clutter)
    top_features = rank_shift_df.nsmallest(15, 'avg_rank').copy()
    
    # Sort by rank change for better visualization
    top_features = top_features.sort_values('rank_change', ascending=False)
    
    fig = go.Figure()
    
    # Add grid lines for better readability
    max_rank = max(top_features[['rank_1993', 'rank_2003']].max())
    for i in range(1, int(max_rank) + 1, 2):
        fig.add_hline(y=i, line_dash="dot", line_color="rgba(0,0,0,0.1)", line_width=1)
    
    # Color code by direction of change
    for _, row in top_features.iterrows():
        rank_change = row['rank_change']
        
        # Determine color based on change direction
        if rank_change > 0:  # Rank increased (worse) - red
            color = '#E74C3C'
            line_style = 'solid'
            line_width = 3
        elif rank_change < 0:  # Rank decreased (better) - green
            color = '#27AE60'
            line_style = 'solid'
            line_width = 3
        else:  # No change - gray
            color = '#95A5A6'
            line_style = 'dash'
            line_width = 2
        
        # Add line
        fig.add_trace(go.Scatter(
            x=[1993, 2003],
            y=[row['rank_1993'], row['rank_2003']],
            mode='lines+markers',
            name=row['feature'],
            line=dict(color=color, width=line_width, dash=line_style),
            marker=dict(size=12, color=color, line=dict(color='white', width=1.5)),
            showlegend=False,
            hovertemplate=f'<b>{row["feature"]}</b><br>1993: Rank {row["rank_1993"]:.0f}<br>2003: Rank {row["rank_2003"]:.0f}<br>Change: {rank_change:+.0f}<extra></extra>'
        ))
        
        # Add feature label on right side (2003)
        fig.add_annotation(
            x=2003,
            y=row['rank_2003'],
            xref='x',
            yref='y',
            text=row['feature'],
            showarrow=False,
            xanchor='left',
            xshift=10,
            font=dict(size=10, family='Arial', color=color),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=color,
            borderwidth=1,
            borderpad=3
        )
    
    # Add legend for change direction
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines+markers',
        name='Improved (↓)',
        line=dict(color='#27AE60', width=3),
        marker=dict(size=12, color='#27AE60')
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines+markers',
        name='Declined (↑)',
        line=dict(color='#E74C3C', width=3),
        marker=dict(size=12, color='#E74C3C')
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines+markers',
        name='No Change',
        line=dict(color='#95A5A6', width=2, dash='dash'),
        marker=dict(size=12, color='#95A5A6')
    ))
    
    fig.update_layout(
        title=dict(
            text="Feature Rank Shift: 1993 vs 2003<br><sub>Top 15 Features by Average Rank (Lower Rank = More Important)</sub>",
            x=0.5,
            font=dict(size=20, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="Year", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            tickmode='array',
            tickvals=[1993, 2003],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title=dict(text="Feature Rank (Lower = Better)", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            autorange='reversed',  # Lower rank (1) at top
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            dtick=1
        ),
        template="plotly_white",
        height=700,
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
        margin=dict(l=80, r=250, t=100, b=60)
    )
    
    return fig


def main():
    """Main execution function for Stage 09."""
    print("=" * 60)
    print("Stage 09: Interpretability - Feature Drivers and Driver Shift")
    print("=" * 60)
    
    # Load configuration
    print("\n[1/7] Loading configuration...")
    try:
        config = load_config("config/params.yaml")
        export_png = config.get("export_png", True)
        seed = config.get("seed", 42)
        compute_shap = config.get("compute_shap", False)
        shap_sample_size = config.get("shap_sample_size", 1000)
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        sys.exit(1)
    
    # Create output directories
    tables_dir = project_root / "outputs" / "tables" / "interpret"
    viz_dir = project_root / "outputs" / "viz" / "interpret"
    fig_dir = project_root / "outputs" / "figures" / "interpret"
    
    tables_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Load feature data and models
    print("\n[2/7] Loading models and feature data...")
    all_importance = {}
    all_perm_importance = {}
    all_pd_data = []
    
    for year in [1993, 2003]:
        print(f"\n  Processing {year}...")
        
        # Load features
        features_path = project_root / "parquet" / "features" / f"year={year}" / "features_model.parquet"
        features = pd.read_parquet(features_path)
        
        # Split data
        train_data = features[features['split'] == 'train'].copy()
        test_data = features[features['split'] == 'test'].copy()
        
        # Load LightGBM model first to get its feature names
        model_path = project_root / "models" / "within_year" / str(year) / "lgbm.txt"
        print(f"    Loading LightGBM model...", end=" ")
        lgbm_model = lgb.Booster(model_file=str(model_path))
        model_feature_names = lgbm_model.feature_name()  # Get features model was trained with
        print("✓")
        
        # Use only features that model was trained with (filter to match model)
        feature_cols = [col for col in model_feature_names 
                       if col in features.columns and col not in ['Year', 'split', 'ontime15']]
        
        X_train = train_data[feature_cols]
        y_train = train_data['ontime15']
        X_test = test_data[feature_cols]
        y_test = test_data['ontime15']
        
        # Feature importance (gain) - use model's actual feature names
        print(f"    Computing feature importance (gain)...", end=" ")
        importance_df = get_lgbm_feature_importance(lgbm_model)  # Use model's feature names
        importance_df['year'] = year
        all_importance[year] = importance_df
        
        importance_path = tables_dir / f"tbl_43_feature_importance_gain_{year}.parquet"
        importance_df.to_parquet(importance_path, index=False)
        print(f"✓ Saved: {importance_path}")
        
        # Permutation importance
        print(f"    Computing permutation importance...", end=" ")
        perm_importance_df = compute_permutation_importance(
            lgbm_model, X_test, y_test, model_type='lgbm', n_repeats=3, random_state=seed
        )
        perm_importance_df['year'] = year
        all_perm_importance[year] = perm_importance_df
        
        perm_path = tables_dir / f"tbl_45_permutation_importance_{year}.parquet"
        perm_importance_df.to_parquet(perm_path, index=False)
        print(f"✓ Saved: {perm_path}")
        
        # Partial dependence for top 5 features
        print(f"    Computing partial dependence (top 5 features)...", end=" ")
        top_features = importance_df.head(5)['feature'].tolist()
        
        for feat in top_features:
            if feat in X_test.columns:
                try:
                    feat_values, pred_values = compute_partial_dependence(
                        lgbm_model, X_test, feat, model_type='lgbm', grid_points=30
                    )
                    
                    pd_df = pd.DataFrame({
                        'year': year,
                        'feature': feat,
                        'feature_value': feat_values,
                        'predicted_value': pred_values
                    })
                    all_pd_data.append(pd_df)
                except Exception as e:
                    print(f"\n      Warning: Could not compute PD for {feat}: {e}")
        
        print("✓")
        
        # SHAP values if enabled
        if compute_shap:
            print(f"    Computing SHAP values (sample size: {shap_sample_size})...", end=" ")
            try:
                shap_values, expected_value = compute_shap_values(
                    lgbm_model, X_test, sample_size=shap_sample_size, random_state=seed
                )
                
                # Save SHAP values
                shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
                shap_df['expected_value'] = expected_value
                shap_path = tables_dir / f"tbl_shap_values_{year}.parquet"
                shap_df.to_parquet(shap_path, index=False)
                
                # Create SHAP summary (use sampled X_test that was used for SHAP)
                X_test_sample = X_test.sample(n=min(shap_sample_size, len(X_test)), random_state=seed)
                shap_summary = create_shap_summary_data(
                    shap_values, X_test_sample, X_test.columns.tolist(), top_n=15
                )
                shap_summary['year'] = year
                shap_summary_path = tables_dir / f"tbl_shap_summary_{year}.csv"
                shap_summary.to_csv(shap_summary_path, index=False)
                
                print(f"✓ Saved: {shap_path}")
            except Exception as e:
                print(f"✗ Error computing SHAP: {e}")
    
    # Feature rank shift analysis
    print("\n[3/7] Computing feature rank shift...")
    importance_1993 = all_importance[1993].set_index('feature')
    importance_2003 = all_importance[2003].set_index('feature')
    
    # Merge ranks
    all_features = importance_1993.index.union(importance_2003.index)
    rank_shift = pd.DataFrame({
        'feature': all_features,
        'rank_1993': [importance_1993.loc[feat, 'rank'] if feat in importance_1993.index else 999 
                     for feat in all_features],
        'rank_2003': [importance_2003.loc[feat, 'rank'] if feat in importance_2003.index else 999 
                     for feat in all_features]
    })
    
    rank_shift['rank_shift'] = rank_shift['rank_2003'] - rank_shift['rank_1993']
    rank_shift = rank_shift.fillna(999)  # Features not in one year get high rank
    
    rank_shift_path = tables_dir / "tbl_47_feature_rank_shift_1993_vs_2003.csv"
    rank_shift.to_csv(rank_shift_path, index=False)
    print(f"  ✓ Saved: {rank_shift_path}")
    
    # Save importance for 2003 (was missing in loop)
    importance_2003_df = all_importance[2003]
    importance_2003_path = tables_dir / f"tbl_44_feature_importance_gain_2003.parquet"
    importance_2003_df.to_parquet(importance_2003_path, index=False)
    
    # Save permutation importance for 2003
    perm_2003_df = all_perm_importance[2003]
    perm_2003_path = tables_dir / f"tbl_46_permutation_importance_2003.parquet"
    perm_2003_df.to_parquet(perm_2003_path, index=False)
    
    # Save partial dependence data
    if all_pd_data:
        pd_df_all = pd.concat(all_pd_data, ignore_index=True)
        pd_1993 = pd_df_all[pd_df_all['year'] == 1993]
        pd_2003 = pd_df_all[pd_df_all['year'] == 2003]
        
        pd_1993.to_parquet(tables_dir / "tbl_48_partial_dependence_points_1993.parquet", index=False)
        pd_2003.to_parquet(tables_dir / "tbl_49_partial_dependence_points_2003.parquet", index=False)
        print(f"  ✓ Saved partial dependence data")
    
    # Generate visualizations
    print("\n[4/7] Generating visualizations...")
    
    # Feature importance charts
    for year in [1993, 2003]:
        print(f"  Creating importance chart for {year}...", end=" ")
        fig = create_importance_chart(all_importance[year], year, top_n=15)
        
        viz_num = 51 if year == 1993 else 52
        save_dual(
            fig,
            str(viz_dir / f"viz_{viz_num:02d}_top15_importance_gain_{year}.plotly.json"),
            str(fig_dir / f"fig_{viz_num:02d}_top15_importance_gain_{year}.png"),
            export_png=export_png
        )
        print("✓")
    
    # Rank shift chart
    print("  Creating rank shift chart...", end=" ")
    fig = create_rank_shift_chart(rank_shift)
    save_dual(
        fig,
        str(viz_dir / "viz_53_rank_shift_slope_chart.plotly.json"),
        str(fig_dir / "fig_53_rank_shift_slope_chart.png"),
        export_png=export_png
    )
    print("✓")
    
    # SHAP visualizations if computed
    if compute_shap:
        print("\n[4.5/7] Generating SHAP visualizations...")
        for year in [1993, 2003]:
            try:
                shap_summary_path = tables_dir / f"tbl_shap_summary_{year}.csv"
                
                if shap_summary_path.exists():
                    shap_summary = pd.read_csv(shap_summary_path)
                    
                    # SHAP summary plot
                    print(f"  Creating SHAP summary for {year}...", end=" ")
                    fig = create_shap_summary_plot(shap_summary, year)
                    viz_num = 54 if year == 1993 else 55
                    save_dual(
                        fig,
                        str(viz_dir / f"viz_{viz_num:02d}_shap_summary_{year}.plotly.json"),
                        str(fig_dir / f"fig_{viz_num:02d}_shap_summary_{year}.png"),
                        export_png=export_png
                    )
                    print("✓")
            except Exception as e:
                print(f"  ✗ Error creating SHAP plots for {year}: {e}")
    
    # Summary
    print("\n[5/7] Interpretability Summary:")
    print("=" * 60)
    for year in [1993, 2003]:
        print(f"\n{year} - Top 5 Features:")
        top5 = all_importance[year].head(5)
        for idx, row in top5.iterrows():
            print(f"  {int(row['rank'])}. {row['feature']}: {row['importance_gain']:.0f} ({row['importance_pct']:.1f}%)")
    
    print(f"\nFeature Rank Shifts:")
    top_shifts = rank_shift.nlargest(5, 'rank_shift')  # Biggest increases
    print("  Top 5 Rank Increases (2003 vs 1993):")
    for _, row in top_shifts.iterrows():
        print(f"    {row['feature']}: {row['rank_1993']:.0f} → {row['rank_2003']:.0f} (+{row['rank_shift']:.0f})")
    
    bottom_shifts = rank_shift.nsmallest(5, 'rank_shift')  # Biggest decreases
    print("  Top 5 Rank Decreases (2003 vs 1993):")
    for _, row in bottom_shifts.iterrows():
        print(f"    {row['feature']}: {row['rank_1993']:.0f} → {row['rank_2003']:.0f} ({row['rank_shift']:.0f})")
    
    print("\n✓ Stage 09 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
