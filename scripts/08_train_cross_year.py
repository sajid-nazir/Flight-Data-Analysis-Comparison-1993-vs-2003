#!/usr/bin/env python3
"""
Stage 08: Cross-year generalization and drift diagnostics

This script:
1. Trains on 1993, tests on 2003 (LR + LGBM) - using 1993 encoders on 2003 data
2. Trains on 2003, tests on 1993 (LR + LGBM) - using 2003 encoders on 1993 data
3. Computes generalization loss vs within-year metrics
4. Performs drift diagnostics:
   - PSI for numeric features
   - Category frequency shifts (pure frequency: count/N per year)
   - Target rate shifts (mean ontime15 per category per year) - descriptive only
5. Generates visualizations

CRITICAL: Uses train-year encoders on test-year data to avoid target leakage.
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
from src.flight_delay.model_train import train_logistic_regression, train_lightgbm
from src.flight_delay.model_eval import (
    evaluate_model, get_calibration_data, calculate_psi
)
from src.flight_delay.features import load_encoders, apply_target_encoders
from src.flight_delay.viz_specs import save_dual, save_plotly_json


def predict_proba_lgbm(model, X: pd.DataFrame) -> np.ndarray:
    """Get predictions from LightGBM model."""
    return model.predict(X.values, num_iteration=model.best_iteration)


def create_auc_heatmap(metrics_df: pd.DataFrame) -> go.Figure:
    """Create AUC heatmap for train/test year combinations."""
    # Pivot table: train_year x test_year x model
    # Only use 1993 and 2003
    years = [1993, 2003]
    heatmap_data = []
    for train_year in years:
        for test_year in years:
            for model in ['logreg', 'lightgbm']:
                row = metrics_df[
                    (metrics_df['train_year'] == train_year) &
                    (metrics_df['test_year'] == test_year) &
                    (metrics_df['model'] == model)
                ]
                if len(row) > 0:
                    heatmap_data.append({
                        'train_year': train_year,
                        'test_year': test_year,
                        'model': model,
                        'roc_auc': row.iloc[0]['roc_auc']
                    })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Create separate heatmaps for each model
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Logistic Regression', 'LightGBM'),
        horizontal_spacing=0.15
    )
    
    for idx, model in enumerate(['logreg', 'lightgbm'], 1):
        model_data = heatmap_df[heatmap_df['model'] == model].copy()
        
        # Explicitly set index and columns to only 1993 and 2003
        pivot = model_data.pivot(index='train_year', columns='test_year', values='roc_auc')
        
        # Reindex to ensure only 1993 and 2003 are shown, in correct order
        pivot = pivot.reindex(index=years, columns=years)
        
        # Convert to numpy array for heatmap
        z_values = pivot.values
        x_labels = [str(year) for year in pivot.columns.tolist()]
        y_labels = [str(year) for year in pivot.index.tolist()]
        
        fig.add_trace(
            go.Heatmap(
                z=z_values,
                x=x_labels,
                y=y_labels,
                colorscale='RdYlGn',
                text=z_values,
                texttemplate='%{text:.3f}',
                textfont=dict(size=14, family='Arial Black', color='white'),
                colorbar=dict(title=dict(text="ROC-AUC", font=dict(size=12, family='Arial Black')), x=1.0 if idx == 2 else 0.45),
                showscale=(idx == 2),
                hovertemplate='Train: %{y}<br>Test: %{x}<br>AUC: %{z:.3f}<extra></extra>'
            ),
            row=1, col=idx
        )
    
    # Set explicit tick values to only show 1993 and 2003
    fig.update_xaxes(
        title=dict(text="Test Year", font=dict(size=14, family='Arial Black')),
        tickfont=dict(size=12, family='Arial'),
        tickmode='array',
        tickvals=[0, 1],
        ticktext=['1993', '2003'],
        row=1, col=1
    )
    fig.update_xaxes(
        title=dict(text="Test Year", font=dict(size=14, family='Arial Black')),
        tickfont=dict(size=12, family='Arial'),
        tickmode='array',
        tickvals=[0, 1],
        ticktext=['1993', '2003'],
        row=1, col=2
    )
    fig.update_yaxes(
        title=dict(text="Train Year", font=dict(size=14, family='Arial Black')),
        tickfont=dict(size=12, family='Arial'),
        tickmode='array',
        tickvals=[0, 1],
        ticktext=['1993', '2003'],
        row=1, col=1
    )
    fig.update_yaxes(
        title=dict(text="Train Year", font=dict(size=14, family='Arial Black')),
        tickfont=dict(size=12, family='Arial'),
        tickmode='array',
        tickvals=[0, 1],
        ticktext=['1993', '2003'],
        row=1, col=2
    )
    
    fig.update_layout(
        title=dict(
            text="Cross-Year Model Performance Heatmap<br><sub>ROC-AUC by Train/Test Year Combination</sub>",
            x=0.5,
            font=dict(size=20, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        template="plotly_white",
        height=500,
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=80, r=40, t=100, b=60)
    )
    
    return fig


def create_generalization_loss_chart(loss_df: pd.DataFrame) -> go.Figure:
    """Create generalization loss bar chart."""
    fig = go.Figure()
    
    for model in ['logreg', 'lightgbm']:
        model_data = loss_df[loss_df['model'] == model]
        color = '#2E86AB' if model == 'logreg' else '#A23B72'
        
        fig.add_trace(go.Bar(
            x=[f"{row['train_year']}→{row['test_year']}" for _, row in model_data.iterrows()],
            y=model_data['generalization_loss_pct'],
            name=model.upper(),
            marker_color=color,
            marker_line=dict(color='white', width=2),
            text=[f"{val:.1f}%" for val in model_data['generalization_loss_pct']],
            textposition='outside',
            textfont=dict(size=12, family='Arial Black')
        ))
    
    fig.update_layout(
        title=dict(
            text="Generalization Loss: Cross-Year vs Within-Year<br><sub>Performance Degradation</sub>",
            x=0.5,
            font=dict(size=20, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="Train→Test", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title=dict(text="Generalization Loss (%)", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
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


def create_cross_year_calibration_plot(calibration_data: pd.DataFrame) -> go.Figure:
    """Create cross-year calibration curves."""
    fig = go.Figure()
    
    # Updated colors to include model distinction
    colors = {
        '1993→2003_logreg': '#2E86AB', 
        '1993→2003_lightgbm': '#1a5f7a',
        '2003→1993_logreg': '#E74C3C', 
        '2003→1993_lightgbm': '#C0392B'
    }
    
    # Group by both train_test AND model to get separate curves for each model
    for (combo, model_name) in calibration_data[['train_test', 'model']].drop_duplicates().values:
        combo_data = calibration_data[
            (calibration_data['train_test'] == combo) & 
            (calibration_data['model'] == model_name)
        ]
        
        # Create unique identifier for legend
        legend_name = f"{combo} ({model_name.upper()})"
        color_key = f"{combo}_{model_name}"
        color = colors.get(color_key, '#95A5A6')
        
        fig.add_trace(go.Scatter(
            x=combo_data['mean_predicted_value'],
            y=combo_data['fraction_of_positives'],
            mode='lines+markers',
            name=legend_name,
            line=dict(color=color, width=3),
            marker=dict(size=6, color=color),
            hovertemplate=f'<b>{legend_name}</b><br>Predicted: %{{x:.3f}}<br>Actual: %{{y:.3f}}<extra></extra>'
        ))
    
    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=True
    ))
    
    fig.update_layout(
        title=dict(
            text="Cross-Year Calibration Curves<br><sub>Predicted vs Actual Probabilities</sub>",
            x=0.5,
            font=dict(size=20, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="Mean Predicted Probability", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            range=[0, 1],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            dtick=0.2
        ),
        yaxis=dict(
            title=dict(text="Fraction of Positives", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            range=[0, 1],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            dtick=0.2
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


def create_psi_chart(psi_df: pd.DataFrame) -> go.Figure:
    """Create PSI (Population Stability Index) chart."""
    # Sort by PSI value
    psi_df = psi_df.sort_values('psi', ascending=True)
    
    # Color by PSI severity
    colors = psi_df['psi'].apply(
        lambda x: '#2E86AB' if x < 0.1 else '#F18F01' if x < 0.25 else '#E74C3C'
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=psi_df['feature'],
        x=psi_df['psi'],
        orientation='h',
        marker_color=colors,
        marker_line=dict(color='white', width=1),
        text=[f"{val:.3f}" for val in psi_df['psi']],
        textposition='outside',
        textfont=dict(size=11, family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>PSI: %{x:.3f}<extra></extra>'
    ))
    
    # Add PSI threshold lines - more prominent
    fig.add_vline(
        x=0.1,
        line_dash="dash",
        line_color="#27AE60",
        line_width=2,
        opacity=0.8,
        annotation_text="Low Drift (<0.1)",
        annotation_position="top",
        annotation_font=dict(size=11, family='Arial', color='#27AE60')
    )
    fig.add_vline(
        x=0.25,
        line_dash="dash",
        line_color="#F18F01",
        line_width=2,
        opacity=0.8,
        annotation_text="Medium Drift (0.25)",
        annotation_position="top",
        annotation_font=dict(size=11, family='Arial', color='#F18F01')
    )
    
    fig.update_layout(
        title=dict(
            text="Feature Drift: Population Stability Index (PSI)<br><sub>1993 vs 2003 Feature Distributions</sub>",
            x=0.5,
            font=dict(size=20, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="PSI Value", font=dict(size=14, family='Arial Black')),
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
        margin=dict(l=120, r=40, t=100, b=60)
    )
    
    return fig


def main():
    """Main execution function for Stage 08."""
    print("=" * 60)
    print("Stage 08: Cross-Year Generalization and Drift Diagnostics")
    print("=" * 60)
    
    # Load configuration
    print("\n[1/9] Loading configuration...")
    try:
        config = load_config("config/params.yaml")
        export_png = config.get("export_png", True)
        seed = config.get("seed", 42)
        models_to_train = config.get("models", ["logreg", "lightgbm"])
        force_retrain = config.get("force_retrain", False)
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        sys.exit(1)
    
    # Create output directories
    models_dir = project_root / "models" / "cross_year"
    tables_dir = project_root / "outputs" / "tables" / "model"
    viz_dir = project_root / "outputs" / "viz" / "model"
    fig_dir = project_root / "outputs" / "figures" / "model"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base feature data (not pre-encoded, to avoid leakage)
    print("\n[2/9] Loading base feature data...")
    base_1993 = pd.read_parquet(project_root / "parquet" / "features" / "year=1993" / "features_base.parquet")
    base_2003 = pd.read_parquet(project_root / "parquet" / "features" / "year=2003" / "features_base.parquet")
    
    print(f"  1993: {len(base_1993):,} rows")
    print(f"  2003: {len(base_2003):,} rows")
    
    # Load encoders (will be applied on-the-fly using train-year encoders)
    print("  Loading encoders...")
    encoders_dir = project_root / "encoders"
    encoders_1993 = load_encoders(str(encoders_dir / f"target_encoders_train_1993.json"))
    encoders_2003 = load_encoders(str(encoders_dir / f"target_encoders_train_2003.json"))
    print("  ✓ Encoders loaded")
    
    # Load within-year metrics for comparison
    print("\n[3/9] Loading within-year metrics...")
    within_year_metrics = pd.read_csv(tables_dir / "tbl_29_within_year_metrics.csv")
    print("✓ Loaded")
    
    # Cross-year training and evaluation
    print("\n[4/9] Training cross-year models...")
    print("  NOTE: Using train-year encoders to avoid target leakage")
    cross_year_metrics = []
    calibration_points = []
    
    for train_year in [1993, 2003]:
        for test_year in [1993, 2003]:
            if train_year == test_year:
                continue  # Skip same-year (already done in Stage 07)
            
            print(f"\n  Training on {train_year}, testing on {test_year}...")
            
            # Get base train and test data
            train_base = base_1993 if train_year == 1993 else base_2003
            test_base = base_1993 if test_year == 1993 else base_2003
            
            # Load train-year encoders (CRITICAL: use train-year encoders, not test-year)
            train_encoders = encoders_1993 if train_year == 1993 else encoders_2003
            
            # Apply train-year encoders to both train and test data
            print(f"    Applying {train_year} encoders to train and test data...", end=" ")
            train_encoded = apply_target_encoders(train_base, train_encoders)
            test_encoded = apply_target_encoders(test_base, train_encoders)
            print("✓")
            
            # Get feature columns (exclude metadata and target)
            # Include Month as it's a useful feature for seasonal patterns
            feature_cols = [col for col in train_encoded.columns 
                           if col not in ['Year', 'split', 'ontime15', 'ArrDelay',
                                         'UniqueCarrier', 'Origin', 'Dest', 'route', 
                                         'dep_hour_bin', 'distance_bin', 'DepTime']]
            
            # Use all training data (no split for cross-year)
            X_train = train_encoded[feature_cols]
            y_train = train_encoded['ontime15']
            X_test = test_encoded[feature_cols]
            y_test = test_encoded['ontime15']
            
            print(f"    Train: {len(X_train):,} rows, Test: {len(X_test):,} rows")
            print(f"    Features: {len(feature_cols)}")
            
            # Train and evaluate models
            for model_name in models_to_train:
                model_dir = models_dir / f"train{train_year}_test{test_year}"
                model_dir.mkdir(parents=True, exist_ok=True)
                
                if model_name == 'logreg':
                    model_path = model_dir / "lr.joblib"
                    if model_path.exists() and not force_retrain:
                        print(f"    Loading existing {model_name} model...", end=" ")
                        model = joblib.load(model_path)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        print("✓")
                    else:
                        print(f"    Training {model_name}...", end=" ")
                        model = train_logistic_regression(X_train, y_train, random_state=seed)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        joblib.dump(model, model_path)
                        print("✓")
                    
                elif model_name == 'lightgbm':
                    model_path = model_dir / "lgbm.txt"
                    if model_path.exists() and not force_retrain:
                        print(f"    Loading existing {model_name} model...", end=" ")
                        model = lgb.Booster(model_file=str(model_path))
                        y_pred_proba = predict_proba_lgbm(model, X_test)
                        print("✓")
                    else:
                        print(f"    Training {model_name}...", end=" ")
                        model = train_lightgbm(X_train, y_train, random_state=seed)
                        y_pred_proba = predict_proba_lgbm(model, X_test)
                        model.save_model(str(model_path))
                        print("✓")
                
                # Evaluate
                print(f"      Evaluating...", end=" ")
                metrics = evaluate_model(y_test.values, y_pred_proba, threshold=0.5)
                metrics['train_year'] = train_year
                metrics['test_year'] = test_year
                metrics['model'] = model_name
                cross_year_metrics.append(metrics)
                
                # Calibration data
                frac_pos, mean_pred, counts = get_calibration_data(y_test.values, y_pred_proba)
                cal_df = pd.DataFrame({
                    'train_year': train_year,
                    'test_year': test_year,
                    'model': model_name,
                    'mean_predicted_value': mean_pred,
                    'fraction_of_positives': frac_pos,
                    'count': counts,
                    'train_test': f"{train_year}→{test_year}"
                })
                calibration_points.append(cal_df)
                
                print("✓")
    
    # Save cross-year metrics
    print("\n[5/9] Saving cross-year metrics...")
    cross_year_df = pd.DataFrame(cross_year_metrics)
    cross_year_df.to_csv(tables_dir / "tbl_37_cross_year_metrics.csv", index=False)
    print(f"  ✓ Saved: {tables_dir / 'tbl_37_cross_year_metrics.csv'}")
    
    # Compute generalization loss
    print("\n[6/9] Computing generalization loss...")
    generalization_loss = []
    
    for train_year in [1993, 2003]:
        for test_year in [1993, 2003]:
            if train_year == test_year:
                continue
            
            for model_name in models_to_train:
                # Within-year performance
                within = within_year_metrics[
                    (within_year_metrics['year'] == train_year) &
                    (within_year_metrics['model'] == model_name)
                ]
                within_auc = within.iloc[0]['roc_auc'] if len(within) > 0 else 0
                
                # Cross-year performance
                cross = cross_year_df[
                    (cross_year_df['train_year'] == train_year) &
                    (cross_year_df['test_year'] == test_year) &
                    (cross_year_df['model'] == model_name)
                ]
                cross_auc = cross.iloc[0]['roc_auc'] if len(cross) > 0 else 0
                
                # Generalization loss
                loss_absolute = within_auc - cross_auc
                loss_pct = (loss_absolute / within_auc * 100) if within_auc > 0 else 0
                
                generalization_loss.append({
                    'train_year': train_year,
                    'test_year': test_year,
                    'model': model_name,
                    'within_year_auc': within_auc,
                    'cross_year_auc': cross_auc,
                    'generalization_loss_absolute': loss_absolute,
                    'generalization_loss_pct': loss_pct
                })
    
    loss_df = pd.DataFrame(generalization_loss)
    loss_df.to_csv(tables_dir / "tbl_38_generalization_loss.csv", index=False)
    print(f"  ✓ Saved: {tables_dir / 'tbl_38_generalization_loss.csv'}")
    
    # Save calibration points
    calibration_df = pd.concat(calibration_points, ignore_index=True)
    calibration_df.to_parquet(tables_dir / "tbl_39_cross_year_calibration_points.parquet", index=False)
    print(f"  ✓ Saved: {tables_dir / 'tbl_39_cross_year_calibration_points.parquet'}")
    
    # Drift diagnostics: PSI for numeric features
    print("\n[7/9] Computing feature drift (PSI)...")
    # Load model features for PSI (using within-year encodings for comparison)
    model_1993 = pd.read_parquet(project_root / "parquet" / "features" / "year=1993" / "features_model.parquet")
    model_2003 = pd.read_parquet(project_root / "parquet" / "features" / "year=2003" / "features_model.parquet")
    
    # PSI on numeric features only (exclude target-encoded features to avoid circularity)
    numeric_features = [col for col in model_1993.columns if col not in 
                       ['Year', 'Month', 'split', 'ontime15',
                        'UniqueCarrier_freq', 'Origin_freq', 'Dest_freq', 'route_freq', 
                        'dep_hour_bin_freq', 'distance_bin_freq']]
    
    psi_results = []
    for feat in numeric_features:
        if feat in model_1993.columns and feat in model_2003.columns:
            psi = calculate_psi(model_1993[feat].values, model_2003[feat].values)
            psi_results.append({'feature': feat, 'psi': psi})
    
    psi_df = pd.DataFrame(psi_results)
    psi_df.to_csv(tables_dir / "tbl_40_feature_shift_psi.csv", index=False)
    print(f"  ✓ Saved: {tables_dir / 'tbl_40_feature_shift_psi.csv'}")
    
    # Category frequency shifts (pure frequency, no target involved)
    print("\n[8/9] Computing category frequency shifts (pure frequency)...")
    
    # Airport frequency shifts (count/N per year)
    if 'Origin' in base_1993.columns and 'Origin' in base_2003.columns:
        origin_freq_1993 = base_1993['Origin'].value_counts(normalize=True)
        origin_freq_2003 = base_2003['Origin'].value_counts(normalize=True)
        
        airport_shift = pd.DataFrame({
            'airport': origin_freq_1993.index.union(origin_freq_2003.index),
            'freq_1993': [origin_freq_1993.get(airport, 0) for airport in origin_freq_1993.index.union(origin_freq_2003.index)],
            'freq_2003': [origin_freq_2003.get(airport, 0) for airport in origin_freq_1993.index.union(origin_freq_2003.index)]
        })
        airport_shift['shift_absolute'] = airport_shift['freq_2003'] - airport_shift['freq_1993']
        airport_shift['shift_pct'] = (airport_shift['shift_absolute'] / airport_shift['freq_1993'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
        airport_shift.to_csv(tables_dir / "tbl_41_category_frequency_shift_airports.csv", index=False)
        print(f"  ✓ Saved airport frequency shifts")
    
    # Carrier frequency shifts (count/N per year)
    if 'UniqueCarrier' in base_1993.columns and 'UniqueCarrier' in base_2003.columns:
        carrier_freq_1993 = base_1993['UniqueCarrier'].value_counts(normalize=True)
        carrier_freq_2003 = base_2003['UniqueCarrier'].value_counts(normalize=True)
        
        carrier_shift = pd.DataFrame({
            'carrier': carrier_freq_1993.index.union(carrier_freq_2003.index),
            'freq_1993': [carrier_freq_1993.get(carrier, 0) for carrier in carrier_freq_1993.index.union(carrier_freq_2003.index)],
            'freq_2003': [carrier_freq_2003.get(carrier, 0) for carrier in carrier_freq_1993.index.union(carrier_freq_2003.index)]
        })
        carrier_shift['shift_absolute'] = carrier_shift['freq_2003'] - carrier_shift['freq_1993']
        carrier_shift['shift_pct'] = (carrier_shift['shift_absolute'] / carrier_shift['freq_1993'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
        carrier_shift.to_csv(tables_dir / "tbl_42_category_frequency_shift_carriers.csv", index=False)
        print(f"  ✓ Saved carrier frequency shifts")
    
    # Target rate shifts (mean ontime15 per category per year) - SEPARATE from frequency
    print("\n[8b/9] Computing target rate shifts (mean ontime15 per category)...")
    print("  NOTE: This is descriptive only, NOT used as model features")
    
    # Airport target rate shifts
    if 'Origin' in base_1993.columns and 'Origin' in base_2003.columns:
        origin_target_1993 = base_1993.groupby('Origin')['ontime15'].mean()
        origin_target_2003 = base_2003.groupby('Origin')['ontime15'].mean()
        
        airport_target_shift = pd.DataFrame({
            'airport': origin_target_1993.index.union(origin_target_2003.index),
            'target_rate_1993': [origin_target_1993.get(airport, np.nan) for airport in origin_target_1993.index.union(origin_target_2003.index)],
            'target_rate_2003': [origin_target_2003.get(airport, np.nan) for airport in origin_target_1993.index.union(origin_target_2003.index)]
        })
        airport_target_shift['shift_absolute'] = airport_target_shift['target_rate_2003'] - airport_target_shift['target_rate_1993']
        airport_target_shift['shift_pct'] = (airport_target_shift['shift_absolute'] / airport_target_shift['target_rate_1993'] * 100).replace([np.inf, -np.inf], np.nan)
        airport_target_shift.to_csv(tables_dir / "tbl_43_category_target_rate_shift_airports.csv", index=False)
        print(f"  ✓ Saved airport target rate shifts")
    
    # Carrier target rate shifts
    if 'UniqueCarrier' in base_1993.columns and 'UniqueCarrier' in base_2003.columns:
        carrier_target_1993 = base_1993.groupby('UniqueCarrier')['ontime15'].mean()
        carrier_target_2003 = base_2003.groupby('UniqueCarrier')['ontime15'].mean()
        
        carrier_target_shift = pd.DataFrame({
            'carrier': carrier_target_1993.index.union(carrier_target_2003.index),
            'target_rate_1993': [carrier_target_1993.get(carrier, np.nan) for carrier in carrier_target_1993.index.union(carrier_target_2003.index)],
            'target_rate_2003': [carrier_target_2003.get(carrier, np.nan) for carrier in carrier_target_1993.index.union(carrier_target_2003.index)]
        })
        carrier_target_shift['shift_absolute'] = carrier_target_shift['target_rate_2003'] - carrier_target_shift['target_rate_1993']
        carrier_target_shift['shift_pct'] = (carrier_target_shift['shift_absolute'] / carrier_target_shift['target_rate_1993'] * 100).replace([np.inf, -np.inf], np.nan)
        carrier_target_shift.to_csv(tables_dir / "tbl_44_category_target_rate_shift_carriers.csv", index=False)
        print(f"  ✓ Saved carrier target rate shifts")
    
    # Generate visualizations
    print("\n[9/9] Generating visualizations...")
    
    # AUC heatmap
    print("  Creating AUC heatmap...", end=" ")
    # Combine within-year and cross-year metrics
    all_metrics = pd.concat([
        within_year_metrics.rename(columns={'year': 'test_year'}).assign(train_year=within_year_metrics['year']),
        cross_year_df
    ], ignore_index=True)
    fig = create_auc_heatmap(all_metrics)
    save_dual(
        fig,
        str(viz_dir / "viz_43_auc_heatmap_train_vs_test_year.plotly.json"),
        str(fig_dir / "fig_43_auc_heatmap_train_vs_test_year.png"),
        export_png=export_png
    )
    print("✓")
    
    # Generalization loss
    print("  Creating generalization loss chart...", end=" ")
    fig = create_generalization_loss_chart(loss_df)
    save_dual(
        fig,
        str(viz_dir / "viz_44_generalization_loss_bars.plotly.json"),
        str(fig_dir / "fig_44_generalization_loss_bars.png"),
        export_png=export_png
    )
    print("✓")
    
    # Cross-year calibration
    print("  Creating cross-year calibration curves...", end=" ")
    fig = create_cross_year_calibration_plot(calibration_df)
    save_dual(
        fig,
        str(viz_dir / "viz_47_cross_year_calibration_curves.plotly.json"),
        str(fig_dir / "fig_47_cross_year_calibration_curves.png"),
        export_png=export_png
    )
    print("✓")
    
    # PSI chart
    print("  Creating PSI chart...", end=" ")
    fig = create_psi_chart(psi_df)
    save_dual(
        fig,
        str(viz_dir / "viz_48_feature_shift_psi.plotly.json"),
        str(fig_dir / "fig_48_feature_shift_psi.png"),
        export_png=export_png
    )
    print("✓")
    
    # Summary
    print("\n[10/10] Cross-Year Generalization Summary:")
    print("=" * 60)
    print("\nGeneralization Loss:")
    for _, row in loss_df.iterrows():
        print(f"  {row['model'].upper()} {row['train_year']}→{row['test_year']}: "
              f"{row['generalization_loss_pct']:.2f}% "
              f"({row['within_year_auc']:.4f} → {row['cross_year_auc']:.4f})")
    
    print(f"\nFeature Drift (PSI):")
    print(f"  Features analyzed: {len(psi_df)}")
    print(f"  High drift (PSI > 0.25): {len(psi_df[psi_df['psi'] > 0.25])}")
    print(f"  Medium drift (0.1 < PSI <= 0.25): {len(psi_df[(psi_df['psi'] > 0.1) & (psi_df['psi'] <= 0.25)])}")
    print(f"  Low drift (PSI <= 0.1): {len(psi_df[psi_df['psi'] <= 0.1])}")
    
    print("\n✓ Stage 08 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
