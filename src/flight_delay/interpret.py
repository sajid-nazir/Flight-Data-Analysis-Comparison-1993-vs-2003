"""
Model interpretability utilities
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.inspection import permutation_importance
import shap
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


def get_lgbm_feature_importance(model: lgb.Booster, feature_names: List[str] = None) -> pd.DataFrame:
    """
    Get feature importance (gain) from LightGBM model.
    
    Args:
        model: Trained LightGBM Booster
        feature_names: Optional list of feature names (if None, uses model's feature names)
        
    Returns:
        DataFrame with feature importance
    """
    importance = model.feature_importance(importance_type='gain')
    
    # Use model's feature names if not provided
    if feature_names is None:
        feature_names = model.feature_name()
    
    # Ensure lengths match
    if len(feature_names) != len(importance):
        # Use model's feature names if mismatch
        feature_names = model.feature_name()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_gain': importance
    })
    
    # Normalize to percentage
    importance_df['importance_pct'] = (importance_df['importance_gain'] / importance_df['importance_gain'].sum() * 100)
    
    # Rank
    importance_df['rank'] = importance_df['importance_gain'].rank(ascending=False, method='min')
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance_gain', ascending=False)
    
    return importance_df


def compute_permutation_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = 'lgbm',
    n_repeats: int = 3,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute permutation importance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_type: 'lr' or 'lgbm'
        n_repeats: Number of permutation repeats
        random_state: Random seed
        
    Returns:
        DataFrame with permutation importance
    """
    from sklearn.metrics import roc_auc_score
    
    # Baseline score
    if model_type == 'lgbm':
        baseline_pred = model.predict(X_test.values, num_iteration=model.best_iteration)
    else:
        baseline_pred = model.predict_proba(X_test)[:, 1]
    baseline_score = roc_auc_score(y_test, baseline_pred)
    
    # Permutation importance for each feature
    importance_scores = []
    
    for feature in X_test.columns:
        feature_scores = []
        X_test_perm = X_test.copy()
        
        for _ in range(n_repeats):
            # Permute feature
            np.random.seed(random_state + _)
            X_test_perm[feature] = np.random.permutation(X_test_perm[feature].values)
            
            # Predict
            if model_type == 'lgbm':
                perm_pred = model.predict(X_test_perm.values, num_iteration=model.best_iteration)
            else:
                perm_pred = model.predict_proba(X_test_perm)[:, 1]
            
            # Score
            perm_score = roc_auc_score(y_test, perm_pred)
            feature_scores.append(baseline_score - perm_score)  # Importance = drop in score
        
        importance_scores.append({
            'feature': feature,
            'importance_mean': np.mean(feature_scores),
            'importance_std': np.std(feature_scores)
        })
    
    perm_df = pd.DataFrame(importance_scores)
    
    # Rank
    perm_df['rank'] = perm_df['importance_mean'].rank(ascending=False, method='min')
    
    # Sort by importance
    perm_df = perm_df.sort_values('importance_mean', ascending=False)
    
    return perm_df


def compute_partial_dependence(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    model_type: str = 'lgbm',
    grid_points: int = 30,
    sample_size: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute partial dependence for a feature.
    
    Args:
        model: Trained model
        X: Feature DataFrame (sample for efficiency)
        feature: Feature name to compute PD for
        model_type: 'lr' or 'lgbm'
        grid_points: Number of grid points
        sample_size: Sample size for PD computation (for efficiency)
        
    Returns:
        (feature_values, predicted_values)
    """
    # Sample for efficiency
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X.copy()
    
    # Get feature range
    feature_values = np.linspace(X[feature].min(), X[feature].max(), grid_points)
    
    # Create grid
    X_grid = X_sample.copy()
    predicted_values = []
    
    for val in feature_values:
        X_grid[feature] = val
        
        if model_type == 'lgbm':
            pred = model.predict(X_grid.values, num_iteration=model.best_iteration)
        else:
            pred = model.predict_proba(X_grid)[:, 1]
        
        predicted_values.append(pred.mean())
    
    return feature_values, np.array(predicted_values)


def compute_shap_values(
    model: lgb.Booster,
    X: pd.DataFrame,
    sample_size: int = 1000,
    random_state: int = 42
) -> Tuple[np.ndarray, float]:
    """
    Compute SHAP values for LightGBM model.
    
    Args:
        model: Trained LightGBM Booster
        X: Feature DataFrame
        sample_size: Sample size for SHAP computation (for efficiency)
        random_state: Random seed
        
    Returns:
        Tuple of (shap_values, expected_value)
    """
    # Sample for efficiency
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=random_state)
    else:
        X_sample = X.copy()
    
    # Create TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample.values)
    expected_value = explainer.expected_value
    
    # For binary classification, SHAP returns list [shap_values_class_0, shap_values_class_1]
    # We want class 1 (on-time) SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Class 1 (on-time)
    
    return shap_values, expected_value


def create_shap_summary_data(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 15
) -> pd.DataFrame:
    """
    Create SHAP summary data for visualization.
    
    Args:
        shap_values: SHAP values array
        X: Feature DataFrame (sampled)
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with SHAP summary statistics
    """
    # Calculate mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
        'mean_shap': shap_values.mean(axis=0),
        'std_shap': shap_values.std(axis=0)
    })
    
    # Sort by mean absolute SHAP
    summary_df = summary_df.sort_values('mean_abs_shap', ascending=False)
    
    # Get top N
    top_features = summary_df.head(top_n).copy()
    
    return top_features


def get_shap_dependence_data(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: str = None
) -> pd.DataFrame:
    """
    Get SHAP dependence plot data for a feature.
    
    Args:
        shap_values: SHAP values array
        X: Feature DataFrame (sampled)
        feature: Feature name for dependence plot
        interaction_feature: Optional feature for interaction coloring
        
    Returns:
        DataFrame with feature values, SHAP values, and interaction values
    """
    feature_idx = X.columns.get_loc(feature)
    feature_values = X[feature].values
    shap_for_feature = shap_values[:, feature_idx]
    
    data = {
        'feature_value': feature_values,
        'shap_value': shap_for_feature
    }
    
    if interaction_feature and interaction_feature in X.columns:
        data['interaction_value'] = X[interaction_feature].values
    else:
        # Use most correlated feature as interaction
        feature_corr = X.corrwith(pd.Series(feature_values, index=X.index)).abs()
        feature_corr = feature_corr.drop(feature)
        if len(feature_corr) > 0:
            interaction_feature = feature_corr.idxmax()
            data['interaction_value'] = X[interaction_feature].values
        else:
            data['interaction_value'] = np.zeros(len(feature_values))
    
    return pd.DataFrame(data)
