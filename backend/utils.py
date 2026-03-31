"""
Utility functions for metrics calculation, inverse normalization, and visualization helpers.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


def inverse_normalize_price(normalized_prices, scaler, feature_index=3):
    """
    Convert normalized prices back to original scale using the stored MinMaxScaler.
    
    Args:
        normalized_prices (np.ndarray or list): Normalized prices in [0,1] range
        scaler (sklearn.preprocessing.MinMaxScaler): Fitted scaler from data preparation
        feature_index (int): Index of Close price in feature array (default=3 for [O,H,L,C,V])
    
    Returns:
        np.ndarray: Prices in original scale
    """
    # Create dummy array with correct number of features (5: Open, High, Low, Close, Volume)
    normalized_prices = np.asarray(normalized_prices).flatten()
    
    # Create matrix for inverse transform (need all 5 features)
    dummy_features = np.zeros((len(normalized_prices), 5))
    dummy_features[:, feature_index] = normalized_prices
    
    # Inverse transform
    original_scale = scaler.inverse_transform(dummy_features)
    
    # Extract only the Close price column
    return original_scale[:, feature_index]


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive error metrics.
    
    Args:
        y_true (np.ndarray): Ground truth values
        y_pred (np.ndarray): Predicted values
    
    Returns:
        dict: Dictionary containing MSE, MAE, MAPE, RMSE, R²
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Handle edge cases
    if len(y_true) == 0:
        return {}
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # MAPE with safeguards for zero division
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100
    
    # R² score
    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2 = np.nan
        
    # Helper to convert numpy numbers to native python floats for JSON serialization
    def to_float(val):
        return None if (val is None or np.isnan(val) or np.isinf(val)) else float(val)
    
    return {
        'mse': to_float(mse),
        'mae': to_float(mae),
        'rmse': to_float(rmse),
        'mape': to_float(mape),
        'r2': to_float(r2)
    }


def calculate_prediction_errors(y_true, y_pred):
    """
    Calculate residuals and error statistics.
    
    Args:
        y_true (np.ndarray): Ground truth values
        y_pred (np.ndarray): Predicted values
    
    Returns:
        dict: Residuals and error statistics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    residuals = y_pred - y_true
    
    return {
        'residuals': residuals,
        'mean_error': np.mean(residuals),
        'std_error': np.std(residuals),
        'max_error': np.max(np.abs(residuals)),
        'min_error': np.min(np.abs(residuals))
    }


def calculate_directional_accuracy(y_true, y_pred):
    """
    Calculate if the model predicts correct price direction (up/down).
    Useful for trading decisions.
    
    Args:
        y_true (np.ndarray): Ground truth prices (in chronological order)
        y_pred (np.ndarray): Predicted prices
    
    Returns:
        float: Percentage of correct directional predictions
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) < 2:
        return np.nan
    
    # Calculate direction (1 if up, 0 if down)
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    
    accuracy = np.mean(true_direction == pred_direction) * 100
    return accuracy


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate Sharpe ratio for a strategy.
    
    Args:
        returns (np.ndarray): Daily returns as decimals (e.g., 0.02 for 2%)
        risk_free_rate (float): Annual risk-free rate (default: 0% for simplicity)
        periods_per_year (int): Trading days per year (default: 252)
    
    Returns:
        float: Sharpe ratio
    """
    if len(returns) == 0:
        return np.nan
    
    annual_return = np.mean(returns) * periods_per_year
    annual_std = np.std(returns) * np.sqrt(periods_per_year)
    
    if annual_std == 0:
        return np.nan
    
    sharpe = (annual_return - risk_free_rate) / annual_std
    return sharpe


def log_metrics(metrics_dict, prefix=""):
    """
    Log metrics in a readable format.
    
    Args:
        metrics_dict (dict): Dictionary of metrics from calculate_metrics()
        prefix (str): Prefix for log messages (e.g., "Train", "Val", "Test")
    """
    if not metrics_dict:
        logger.warning(f"{prefix} metrics: Empty")
        return
    
    msg = f"{prefix} Metrics: "
    msg += f"MSE={metrics_dict.get('mse', np.nan):.6f}, "
    msg += f"MAE={metrics_dict.get('mae', np.nan):.6f}, "
    msg += f"RMSE={metrics_dict.get('rmse', np.nan):.6f}, "
    msg += f"MAPE={metrics_dict.get('mape', np.nan):.2f}%, "
    msg += f"R²={metrics_dict.get('r2', np.nan):.4f}"
    
    logger.info(msg)


def normalize_for_mse(y_true, y_pred, price_range=None):
    """
    Normalize MSE by price range for better interpretability.
    
    Args:
        y_true (np.ndarray): Ground truth values
        y_pred (np.ndarray): Predicted values
        price_range (tuple): (min_price, max_price). If None, calculated from y_true.
    
    Returns:
        float: Normalized MSE (typically 0-1 where lower is better)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if price_range is None:
        price_range = (np.min(y_true), np.max(y_true))
    
    price_diff = price_range[1] - price_range[0]
    
    if price_diff == 0:
        return np.nan
    
    mse = mean_squared_error(y_true, y_pred)
    normalized_mse = mse / (price_diff ** 2)
    
    return normalized_mse


def calculate_feature_importance_from_ablation(results_dict):
    """
    Calculate feature importance based on ablation study results.
    
    Args:
        results_dict (dict): Dictionary with keys like 'price_only', 'all_features' 
                           containing their respective MSE values
    
    Returns:
        dict: Feature importance metrics
    """
    try:
        mse_price_only = results_dict.get('price_only', {}).get('mse', np.nan)
        mse_all_features = results_dict.get('all_features', {}).get('mse', np.nan)
        
        if np.isnan(mse_price_only) or np.isnan(mse_all_features):
            return {'improvement': np.nan, 'improvement_pct': np.nan}
        
        improvement = mse_price_only - mse_all_features  # Positive means all_features is better
        improvement_pct = (improvement / mse_price_only * 100) if mse_price_only > 0 else 0
        
        return {
            'mse_price_only': mse_price_only,
            'mse_all_features': mse_all_features,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'features_help': improvement > 0  # Boolean: do additional features help?
        }
    
    except Exception as e:
        logger.error(f"Error calculating feature importance: {e}")
        return {}
