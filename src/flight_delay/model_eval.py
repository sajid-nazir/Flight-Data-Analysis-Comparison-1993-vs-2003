"""
Model evaluation utilities
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def evaluate_model(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'brier_score': brier_score_loss(y_true, y_pred_proba),
        'threshold': threshold
    }
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics.update({
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    })
    
    return metrics


def get_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get ROC curve data."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    return fpr, tpr, thresholds


def get_pr_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get Precision-Recall curve data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    return precision, recall, thresholds


def get_calibration_data(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get calibration curve data.
    
    Returns:
        (fraction_of_positives, mean_predicted_value, counts)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    fraction_of_positives = []
    mean_predicted_value = []
    counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        mask = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
        if mask.sum() > 0:
            fraction_of_positives.append(y_true[mask].mean())
            mean_predicted_value.append(y_pred_proba[mask].mean())
            counts.append(mask.sum())
        else:
            fraction_of_positives.append(0)
            mean_predicted_value.append((bin_lower + bin_upper) / 2)
            counts.append(0)
    
    return (
        np.array(fraction_of_positives),
        np.array(mean_predicted_value),
        np.array(counts)
    )


def calculate_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) for a feature.
    
    Args:
        expected: Expected distribution (training data)
        actual: Actual distribution (test data)
        n_bins: Number of bins for discretization
        
    Returns:
        PSI value
    """
    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    
    # Create bins based on expected distribution
    _, bin_edges = np.histogram(expected, bins=n_bins)
    
    # Calculate expected and actual proportions
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    # Normalize to proportions
    expected_prop = expected_counts / len(expected) if len(expected) > 0 else expected_counts
    actual_prop = actual_counts / len(actual) if len(actual) > 0 else actual_counts
    
    # Avoid division by zero
    expected_prop = np.where(expected_prop == 0, 0.0001, expected_prop)
    actual_prop = np.where(actual_prop == 0, 0.0001, actual_prop)
    
    # Calculate PSI
    psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
    
    return psi


def compute_naive_baselines(y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compute naive baseline metrics for comparison.
    
    Args:
        y_true: True labels (binary: 0 or 1)
        
    Returns:
        Dictionary with baseline metrics for each baseline type
    """
    n = len(y_true)
    majority_class = int(np.bincount(y_true.astype(int)).argmax())
    majority_rate = y_true.mean()
    
    # Always on-time baseline (predict all as on-time = 1)
    always_ontime_pred = np.ones(n)
    always_ontime_auc = roc_auc_score(y_true, always_ontime_pred)
    always_ontime_pr = average_precision_score(y_true, always_ontime_pred)
    always_ontime_acc = (y_true == always_ontime_pred).mean()
    
    # Majority class baseline (predict majority class)
    majority_pred = np.full(n, majority_class)
    majority_auc = roc_auc_score(y_true, majority_pred)
    majority_pr = average_precision_score(y_true, majority_pred)
    majority_acc = (y_true == majority_pred).mean()
    
    # Random baseline (random predictions, AUC should be ~0.5)
    np.random.seed(42)
    random_pred = np.random.random(n)
    random_auc = roc_auc_score(y_true, random_pred)
    random_pr = average_precision_score(y_true, random_pred)
    random_acc = (y_true == (random_pred >= 0.5).astype(int)).mean()
    
    baselines = {
        'always_ontime': {
            'roc_auc': always_ontime_auc,
            'pr_auc': always_ontime_pr,
            'accuracy': always_ontime_acc,
            'description': 'Always predict on-time (probability = 1.0)'
        },
        'majority_class': {
            'roc_auc': majority_auc,
            'pr_auc': majority_pr,
            'accuracy': majority_acc,
            'description': f'Always predict majority class (on-time rate = {majority_rate:.3f})'
        },
        'random': {
            'roc_auc': random_auc,
            'pr_auc': random_pr,
            'accuracy': random_acc,
            'description': 'Random predictions (AUC ≈ 0.5)'
        }
    }
    
    return baselines
