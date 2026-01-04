#!/usr/bin/env python3
"""
Stage 07: Within-year modeling (LR + LGBM + calibration)

This script:
1. Trains Logistic Regression baseline
2. Trains LightGBM model
3. Evaluates models (ROC-AUC, PR-AUC, Brier, confusion matrix)
4. Generates evaluation visualizations
5. Saves models and metrics
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import joblib
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.flight_delay.config import load_config
from src.flight_delay.model_train import (
    train_logistic_regression, train_lightgbm, tune_lightgbm_hyperparameters
)
from src.flight_delay.model_eval import (
    evaluate_model, get_roc_curve, get_pr_curve, get_calibration_data,
    compute_naive_baselines
)
from src.flight_delay.viz_specs import save_dual, save_plotly_json
from src.flight_delay.io_artifacts import save_json


def create_roc_curve_plot(fpr: np.ndarray, tpr: np.ndarray, auc: float, year: int, model_name: str) -> go.Figure:
    """Create ROC curve visualization."""
    fig = go.Figure()
    
    # Main ROC curve with better styling
    model_color = '#2E86AB' if model_name.lower() == 'logreg' else '#A23B72'
    # Convert hex to rgba for fill
    if model_color == '#2E86AB':
        fill_color = 'rgba(46, 134, 171, 0.2)'
    else:
        fill_color = 'rgba(162, 59, 114, 0.2)'
    
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'{model_name.upper()} (AUC = {auc:.3f})',
        line=dict(color=model_color, width=4),
        fill='tozeroy',
        fillcolor=fill_color,
        hovertemplate='<b>ROC Curve</b><br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<br>AUC: ' + f'{auc:.3f}<extra></extra>'
    ))
    
    # Diagonal line (random classifier) - more prominent
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier (AUC = 0.500)',
        line=dict(color='#666666', width=2.5, dash='dash'),
        showlegend=True,
        hovertemplate='Random Classifier<br>AUC = 0.500<extra></extra>'
    ))
    
    # Add AUC annotation
    fig.add_annotation(
        x=0.6,
        y=0.2,
        text=f'AUC = {auc:.3f}',
        showarrow=False,
        font=dict(size=16, family='Arial Black', color=model_color),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor=model_color,
        borderwidth=2,
        borderpad=6
    )
    
    fig.update_layout(
        title=dict(
            text=f"ROC Curve - {year} ({model_name.upper()})<br><sub>Receiver Operating Characteristic</sub>",
            x=0.5,
            font=dict(size=20, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="False Positive Rate", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            range=[0, 1],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            dtick=0.2
        ),
        yaxis=dict(
            title=dict(text="True Positive Rate", font=dict(size=14, family='Arial Black')),
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


def create_pr_curve_plot(precision: np.ndarray, recall: np.ndarray, auc: float, year: int, model_name: str) -> go.Figure:
    """Create Precision-Recall curve visualization."""
    fig = go.Figure()
    
    # Main PR curve with better styling
    model_color = '#2E86AB' if model_name.lower() == 'logreg' else '#A23B72'
    # Convert hex to rgba for fill
    if model_color == '#2E86AB':
        fill_color = 'rgba(46, 134, 171, 0.2)'
    else:
        fill_color = 'rgba(162, 59, 114, 0.2)'
    
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f'{model_name.upper()} (PR-AUC = {auc:.3f})',
        line=dict(color=model_color, width=4),
        fill='tozeroy',
        fillcolor=fill_color,
        hovertemplate='<b>PR Curve</b><br>Recall: %{x:.3f}<br>Precision: %{y:.3f}<br>PR-AUC: ' + f'{auc:.3f}<extra></extra>'
    ))
    
    # Add PR-AUC annotation
    fig.add_annotation(
        x=0.6,
        y=0.2,
        text=f'PR-AUC = {auc:.3f}',
        showarrow=False,
        font=dict(size=16, family='Arial Black', color=model_color),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor=model_color,
        borderwidth=2,
        borderpad=6
    )
    
    fig.update_layout(
        title=dict(
            text=f"Precision-Recall Curve - {year} ({model_name.upper()})<br><sub>Precision vs Recall</sub>",
            x=0.5,
            font=dict(size=20, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="Recall", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            range=[0, 1],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            dtick=0.2
        ),
        yaxis=dict(
            title=dict(text="Precision", font=dict(size=14, family='Arial Black')),
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


def create_baseline_comparison_chart(metrics_df: pd.DataFrame) -> go.Figure:
    """Create baseline comparison chart."""
    # Filter to get model metrics and baseline metrics
    model_metrics = metrics_df[~metrics_df['model'].isin(['always_ontime', 'majority_class', 'random'])].copy()
    baseline_metrics = metrics_df[metrics_df['model'].isin(['always_ontime', 'majority_class', 'random'])].copy()
    
    fig = go.Figure()
    
    # Add baseline bars
    baseline_colors = {
        'always_ontime': '#E74C3C',
        'majority_class': '#F39C12',
        'random': '#95A5A6'
    }
    
    for baseline_type in ['always_ontime', 'majority_class', 'random']:
        baseline_data = baseline_metrics[baseline_metrics['model'] == baseline_type]
        if len(baseline_data) > 0:
            fig.add_trace(go.Bar(
                name=baseline_type.replace('_', ' ').title(),
                x=baseline_data['year'].astype(str),
                y=baseline_data['roc_auc'],
                marker=dict(color=baseline_colors[baseline_type], line=dict(color='white', width=1)),
                opacity=0.7,
                legendgroup='baseline'
            ))
    
    # Add model bars
    for year in [1993, 2003]:
        year_models = model_metrics[model_metrics['year'] == year]
        for model_name in ['logreg', 'lightgbm']:
            model_data = year_models[year_models['model'] == model_name]
            if len(model_data) > 0:
                fig.add_trace(go.Bar(
                    name=f'{model_name.upper()} {year}',
                    x=[str(year)],
                    y=[model_data['roc_auc'].iloc[0]],
                    marker=dict(
                        color='#2E86AB' if model_name == 'logreg' else '#A23B72',
                        line=dict(color='white', width=2)
                    ),
                    legendgroup='model',
                    showlegend=True
                ))
    
    fig.update_layout(
        title=dict(
            text="Model Performance vs. Naive Baselines<br><sub>ROC-AUC Comparison</sub>",
            x=0.5,
            font=dict(size=22, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="Year", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial')
        ),
        yaxis=dict(
            title=dict(text="ROC-AUC", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            range=[0, 1]
        ),
        barmode='group',
        template="plotly_white",
        height=600,
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
            font=dict(size=12, family='Arial Black')
        ),
        margin=dict(l=60, r=40, t=100, b=60)
    )
    
    return fig


def create_calibration_plot(
    fraction_of_positives: np.ndarray,
    mean_predicted_value: np.ndarray,
    year: int,
    model_name: str
) -> go.Figure:
    """Create calibration curve visualization."""
    fig = go.Figure()
    
    # Perfect calibration line - add first so it's behind
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='#666666', width=2.5, dash='dash'),
        showlegend=True,
        hovertemplate='Perfect Calibration<br>Predicted = Actual<extra></extra>'
    ))
    
    # Calibration curve with better styling
    model_color = '#2E86AB' if model_name.lower() == 'logreg' else '#A23B72'
    fig.add_trace(go.Scatter(
        x=mean_predicted_value,
        y=fraction_of_positives,
        mode='lines+markers',
        name=f'{model_name.upper()} Calibration',
        line=dict(color=model_color, width=4),
        marker=dict(size=10, color=model_color, line=dict(color='white', width=1.5)),
        hovertemplate='<b>Calibration Point</b><br>Predicted: %{x:.3f}<br>Actual: %{y:.3f}<br>Difference: %{customdata:+.3f}<extra></extra>',
        customdata=fraction_of_positives - mean_predicted_value
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Calibration Curve - {year} ({model_name.upper()})<br><sub>Predicted vs Actual Probabilities</sub>",
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


def predict_proba_lgbm(model, X: pd.DataFrame) -> np.ndarray:
    """Get predictions from LightGBM model."""
    return model.predict(X.values, num_iteration=model.best_iteration)


def main():
    """Main execution function for Stage 07."""
    print("=" * 60)
    print("Stage 07: Within-Year Modeling (LR + LGBM)")
    print("=" * 60)
    
    # Load configuration
    print("\n[1/8] Loading configuration...")
    try:
        config = load_config("config/params.yaml")
        export_png = config.get("export_png", True)
        seed = config.get("seed", 42)
        models_to_train = config.get("models", ["logreg", "lightgbm"])
        hyperparameter_tuning = config.get("hyperparameter_tuning", False)
        tuning_trials = config.get("tuning_trials", 50)
        use_validation_set = config.get("use_validation_set", True)
        force_retrain = config.get("force_retrain", False)
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        sys.exit(1)
    
    # Create output directories
    models_dir = project_root / "models" / "within_year"
    tables_dir = project_root / "outputs" / "tables" / "model"
    viz_dir = project_root / "outputs" / "viz" / "model"
    fig_dir = project_root / "outputs" / "figures" / "model"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each year
    print("\n[2/8] Loading feature data...")
    all_metrics = []
    all_confusion_matrices = []
    
    for year in [1993, 2003]:
        print(f"\n  Processing {year}...")
        year_models_dir = models_dir / str(year)
        year_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load features
        features_path = project_root / "parquet" / "features" / f"year={year}" / "features_model.parquet"
        print(f"    Loading features from {features_path}...", end=" ")
        features = pd.read_parquet(features_path)
        print(f"✓ ({len(features):,} rows)")
        
        # Split into train and test
        train_data = features[features['split'] == 'train'].copy()
        test_data = features[features['split'] == 'test'].copy()
        
        print(f"    Train: {len(train_data):,} rows, Test: {len(test_data):,} rows")
        
        # Prepare features and target
        # Exclude: Year (constant within year), split (data split indicator), ontime15 (target)
        # Include: Month (seasonal patterns), DayOfWeek, and all other features
        feature_cols = [col for col in features.columns 
                       if col not in ['Year', 'split', 'ontime15']]
        
        X_train = train_data[feature_cols]
        y_train = train_data['ontime15']
        X_test = test_data[feature_cols]
        y_test = test_data['ontime15']
        
        print(f"    Features: {len(feature_cols)}")
        
        # Compute naive baselines
        print(f"\n[2.5/8] Computing naive baselines for {year}...")
        baselines = compute_naive_baselines(y_test.values)
        baseline_results = []
        for baseline_name, baseline_metrics in baselines.items():
            baseline_results.append({
                'year': year,
                'baseline_type': baseline_name,
                'roc_auc': baseline_metrics['roc_auc'],
                'pr_auc': baseline_metrics['pr_auc'],
                'accuracy': baseline_metrics['accuracy'],
                'description': baseline_metrics['description']
            })
        all_metrics.extend(baseline_results)
        print("✓ Baselines computed")
        
        # Train and evaluate models
        print(f"\n[3/8] Training models for {year}...")
        
        for model_name in models_to_train:
            if model_name == 'logreg':
                model_path = year_models_dir / "lr.joblib"
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
                model_path = year_models_dir / "lgbm.txt"
                if model_path.exists() and not force_retrain:
                    print(f"    Loading existing {model_name} model...", end=" ")
                    model = lgb.Booster(model_file=str(model_path))
                    y_pred_proba = predict_proba_lgbm(model, X_test)
                    print("✓")
                else:
                    print(f"    Training {model_name}...", end=" ")
                    best_params = None
                    tuning_results = None
                    
                    # Hyperparameter tuning if enabled
                    if hyperparameter_tuning and use_validation_set:
                        print(f"\n      Tuning hyperparameters ({tuning_trials} trials)...", end=" ")
                        # Split training data into train/validation (80/20)
                        from sklearn.model_selection import train_test_split
                        X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(
                            X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
                        )
                        
                        best_params, best_score = tune_lightgbm_hyperparameters(
                            X_train_tune, y_train_tune, X_val_tune, y_val_tune,
                            n_trials=tuning_trials, random_state=seed
                        )
                        
                        tuning_results = {
                            'year': year,
                            'best_params': best_params,
                            'best_validation_auc': float(best_score),
                            'n_trials': tuning_trials
                        }
                        
                        # Save tuning results
                        tuning_path = tables_dir / f"tbl_tuning_results_{year}.json"
                        save_json(tuning_results, str(tuning_path))
                        print(f"✓ Best AUC: {best_score:.4f}")
                        
                        # Use full training set with best params
                        model = train_lightgbm(X_train, y_train, random_state=seed, best_params=best_params)
                    else:
                        model = train_lightgbm(X_train, y_train, random_state=seed)
                    
                    y_pred_proba = predict_proba_lgbm(model, X_test)
                    
                    # Save model
                    model.save_model(str(model_path))
                    print("✓")
            
            # Evaluate model
            print(f"      Evaluating {model_name}...", end=" ")
            metrics = evaluate_model(y_test.values, y_pred_proba, threshold=0.5)
            metrics['year'] = year
            metrics['model'] = model_name
            all_metrics.append(metrics)
            
            # Confusion matrix
            cm = confusion_matrix(y_test.values, (y_pred_proba >= 0.5).astype(int))
            cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'],
                                columns=['Predicted Negative', 'Predicted Positive'])
            cm_df['year'] = year
            cm_df['model'] = model_name
            all_confusion_matrices.append(cm_df)
            
            # Save sample predictions
            sample_preds = test_data[['Year', 'Month', 'ontime15']].copy()
            sample_preds['predicted_proba'] = y_pred_proba
            sample_preds['predicted_class'] = (y_pred_proba >= 0.5).astype(int)
            sample_preds = sample_preds.head(10000)  # Sample 10k for storage
            
            preds_path = tables_dir / f"tbl_35_prediction_scores_sample_{year}.parquet"
            if model_name == 'logreg':
                preds_path = tables_dir / f"tbl_35_prediction_scores_sample_{year}_lr.parquet"
            else:
                preds_path = tables_dir / f"tbl_36_prediction_scores_sample_{year}_lgbm.parquet"
            sample_preds.to_parquet(preds_path, index=False)
            
            print("✓")
            
            # Generate visualizations
            print(f"      Generating visualizations for {model_name}...", end=" ")
            
            # ROC curve
            fpr, tpr, _ = get_roc_curve(y_test.values, y_pred_proba)
            fig = create_roc_curve_plot(fpr, tpr, metrics['roc_auc'], year, model_name.upper())
            
            viz_num = 34 if (year == 1993 and model_name == 'logreg') else \
                     35 if (year == 1993 and model_name == 'lightgbm') else \
                     38 if (year == 2003 and model_name == 'logreg') else 39
            save_dual(
                fig,
                str(viz_dir / f"viz_{viz_num:02d}_roc_within_year_{year}_{model_name}.plotly.json"),
                str(fig_dir / f"fig_{viz_num:02d}_roc_within_year_{year}_{model_name}.png"),
                export_png=export_png
            )
            
            # PR curve
            precision, recall, _ = get_pr_curve(y_test.values, y_pred_proba)
            fig = create_pr_curve_plot(precision, recall, metrics['pr_auc'], year, model_name.upper())
            
            viz_num = 35 if (year == 1993 and model_name == 'logreg') else \
                     36 if (year == 1993 and model_name == 'lightgbm') else \
                     39 if (year == 2003 and model_name == 'logreg') else 40
            save_dual(
                fig,
                str(viz_dir / f"viz_{viz_num:02d}_pr_within_year_{year}_{model_name}.plotly.json"),
                str(fig_dir / f"fig_{viz_num:02d}_pr_within_year_{year}_{model_name}.png"),
                export_png=export_png
            )
            
            # Calibration curve
            frac_pos, mean_pred, _ = get_calibration_data(y_test.values, y_pred_proba)
            fig = create_calibration_plot(frac_pos, mean_pred, year, model_name.upper())
            
            viz_num = 36 if (year == 1993 and model_name == 'logreg') else \
                     37 if (year == 1993 and model_name == 'lightgbm') else \
                     40 if (year == 2003 and model_name == 'logreg') else 41
            save_dual(
                fig,
                str(viz_dir / f"viz_{viz_num:02d}_calibration_within_year_{year}_{model_name}.plotly.json"),
                str(fig_dir / f"fig_{viz_num:02d}_calibration_within_year_{year}_{model_name}.png"),
                export_png=export_png
            )
            
            print("✓")
    
    # Save metrics
    print("\n[4/8] Saving metrics...")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = tables_dir / "tbl_29_within_year_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  ✓ Saved: {metrics_path}")
    
    # Save baseline comparison
    baseline_df = metrics_df[metrics_df['model'].isin(['always_ontime', 'majority_class', 'random'])].copy()
    if len(baseline_df) > 0:
        baseline_path = tables_dir / "tbl_baseline_comparison.csv"
        baseline_df.to_csv(baseline_path, index=False)
        print(f"  ✓ Saved: {baseline_path}")
        
        # Create baseline comparison visualization
        print("\n[4.5/8] Creating baseline comparison chart...", end=" ")
        fig = create_baseline_comparison_chart(metrics_df)
        save_dual(
            fig,
            str(viz_dir / "viz_41_baseline_comparison.plotly.json"),
            str(fig_dir / "fig_41_baseline_comparison.png"),
            export_png=export_png
        )
        print("✓")
    
    # Save confusion matrices
    for cm_df in all_confusion_matrices:
        year = cm_df['year'].iloc[0]
        model = cm_df['model'].iloc[0]
        cm_df_clean = cm_df.drop(columns=['year', 'model'])
        
        cm_path = tables_dir / f"tbl_31_confusion_matrix_{year}_{model}.csv"
        cm_df_clean.to_csv(cm_path)
        print(f"  ✓ Saved: {cm_path}")
    
    # Summary
    print("\n[5/8] Model Performance Summary:")
    print("=" * 60)
    for year in [1993, 2003]:
        print(f"\n{year}:")
        year_metrics = metrics_df[metrics_df['year'] == year]
        for _, row in year_metrics.iterrows():
            # Handle both model entries (with 'model' key) and baseline entries (with 'baseline_type' key)
            if 'model' in row and pd.notna(row['model']) and isinstance(row['model'], str):
                model_name = row['model'].upper()
            elif 'baseline_type' in row and pd.notna(row['baseline_type']):
                model_name = str(row['baseline_type']).upper()
            else:
                model_name = "Unknown"
            
            print(f"  {model_name}:")
            if 'roc_auc' in row and pd.notna(row['roc_auc']):
                print(f"    ROC-AUC: {row['roc_auc']:.4f}")
            if 'pr_auc' in row and pd.notna(row['pr_auc']):
                print(f"    PR-AUC: {row['pr_auc']:.4f}")
            if 'brier_score' in row and pd.notna(row['brier_score']):
                print(f"    Brier Score: {row['brier_score']:.4f}")
            if 'precision' in row and pd.notna(row['precision']):
                print(f"    Precision: {row['precision']:.4f}")
            if 'recall' in row and pd.notna(row['recall']):
                print(f"    Recall: {row['recall']:.4f}")
            if 'f1_score' in row and pd.notna(row['f1_score']):
                print(f"    F1-Score: {row['f1_score']:.4f}")
            if 'accuracy' in row and pd.notna(row['accuracy']):
                print(f"    Accuracy: {row['accuracy']:.4f}")
    
    print("\n✓ Stage 07 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
