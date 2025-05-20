from typing import Dict, Optional, Tuple
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def check_stationarity(data: np.ndarray) -> bool:
    """Check if the time series is stationary using Augmented Dickey-Fuller test."""
    result = adfuller(data)
    return result[1] < 0.05  # Return True if stationary (p-value < 0.05)

def difference_series(data: np.ndarray, order: int = 1) -> np.ndarray:
    """Difference the time series to make it stationary."""
    return np.diff(data, n=order)

def create_arima_model(params: Optional[Dict] = None) -> Tuple[ARIMA, Dict]:
    """Create ARIMA model with default or custom parameters."""
    default_params = {
        'order': (1, 1, 1),  # (p, d, q) order
        'seasonal_order': None,  # No seasonal component by default
        'trend': None,  # No trend by default
        'enforce_stationarity': True,
        'enforce_invertibility': True
    }
    
    if params:
        default_params.update(params)
    
    # Create a dummy model (will be fitted with actual data later)
    dummy_data = np.array([0, 1, 2, 3, 4])
    
    # Remove trend if d > 0
    if default_params['order'][1] > 0:
        default_params['trend'] = None
    
    model = ARIMA(
        dummy_data,
        order=default_params['order'],
        seasonal_order=default_params['seasonal_order'],
        trend=default_params['trend'],
        enforce_stationarity=default_params['enforce_stationarity'],
        enforce_invertibility=default_params['enforce_invertibility']
    )
    
    return model, default_params

def prepare_arima_data(data: np.ndarray, target_column: int = -1) -> np.ndarray:
    """Prepare data for ARIMA model."""
    # Extract target column
    series = data[:, target_column]
    
    # Check stationarity and difference if necessary
    if not check_stationarity(series):
        series = difference_series(series)
    
    return series

def find_best_arima_order(data: np.ndarray, max_p: int = 2, max_d: int = 2, max_q: int = 2) -> Tuple[int, int, int]:
    """Find the best ARIMA order using AIC criterion."""
    best_aic = float('inf')
    best_order = (0, 0, 0)
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                except:
                    continue
    
    return best_order 