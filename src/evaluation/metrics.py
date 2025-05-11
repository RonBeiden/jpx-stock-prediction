import numpy as np
from typing import Dict, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error

def weighted_correlation(y_true: np.ndarray, 
                        y_pred: np.ndarray, 
                        weights: np.ndarray) -> float:
    """Calculate weighted correlation metric."""
    numerator = np.sum(weights * (y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred)))
    denominator = np.sqrt(
        np.sum(weights * (y_true - np.mean(y_true))**2) * 
        np.sum(weights * (y_pred - np.mean(y_pred))**2)
    )
    return numerator / denominator

def evaluate_predictions(y_true: np.ndarray, 
                       y_pred: np.ndarray, 
                       weights: Union[np.ndarray, None] = None) -> Dict[str, float]:
    """Evaluate predictions using multiple metrics."""
    if weights is None:
        weights = np.ones_like(y_true)
    
    metrics = {
        'weighted_correlation': weighted_correlation(y_true, y_pred, weights),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }
    
    return metrics
