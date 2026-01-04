"""
Model training utilities
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import joblib
import optuna
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> LogisticRegression:
    """
    Train Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed
        
    Returns:
        Trained LogisticRegression model
    """
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        n_jobs=-1,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    return model


def tune_lightgbm_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    random_state: int = 42
) -> Tuple[Dict[str, Any], float]:
    """
    Tune LightGBM hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        n_trials: Number of Optuna trials
        random_state: Random seed
        
    Returns:
        Tuple of (best_params, best_score)
    """
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'feature_pre_filter': False,  # Required when tuning min_child_samples
            'verbose': -1,
            'random_state': random_state
        }
        
        try:
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
            )
        except Exception as e:
            raise
        
        # Get validation predictions
        y_pred = model.predict(X_val.values, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_val, y_pred)
        
        return auc
    
    study = optuna.create_study(direction='maximize', study_name='lightgbm_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_score = study.best_value
    
    # Add fixed parameters
    best_params['objective'] = 'binary'
    best_params['metric'] = 'binary_logloss'
    best_params['boosting_type'] = 'gbdt'
    best_params['feature_pre_filter'] = False  # Required when tuning min_child_samples
    best_params['verbose'] = -1
    best_params['random_state'] = random_state
    
    return best_params, best_score


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    random_state: int = 42,
    best_params: Optional[Dict[str, Any]] = None
) -> lgb.Booster:
    """
    Train LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Optional validation features (for early stopping)
        y_val: Optional validation target
        random_state: Random seed
        
    Returns:
        Trained LightGBM Booster
    """
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Use best_params if provided, otherwise use defaults
    if best_params is not None:
        params = best_params.copy()
    else:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': random_state
        }
    
    # Use validation set if provided, otherwise just train
    if X_val is not None and y_val is not None:
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
        )
    else:
        # No validation set - train without early stopping
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.log_evaluation(0)]
        )
    
    return model


def calibrate_model(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str = 'lr'
) -> CalibratedClassifierCV:
    """
    Calibrate model predictions.
    
    Args:
        model: Trained model (LR or LGBM)
        X_val: Validation features
        y_val: Validation target
        model_type: 'lr' or 'lgbm'
        
    Returns:
        Calibrated model
    """
    if model_type == 'lr':
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated.fit(X_val, y_val)
    else:
        # For LightGBM, we'll skip calibration for now as it requires sklearn wrapper
        # Return None to indicate calibration not performed
        return None
    
    return calibrated
