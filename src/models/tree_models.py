from typing import Dict, Optional
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

def create_lightgbm_model(params: Optional[Dict] = None) -> lgb.LGBMRegressor:
    """Create LightGBM model with default or custom parameters."""
    default_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    
    if params:
        default_params.update(params)
    
    return lgb.LGBMRegressor(**default_params)

def create_xgboost_model(params: Optional[Dict] = None) -> xgb.XGBRegressor:
    """Create XGBoost model with default or custom parameters."""
    default_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8
    }
    
    if params:
        default_params.update(params)
    
    return xgb.XGBRegressor(**default_params)

def create_catboost_model(params: Optional[Dict] = None) -> CatBoostRegressor:
    """Create CatBoost model with default or custom parameters."""
    default_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'RMSE'
    }
    
    if params:
        default_params.update(params)
    
    return CatBoostRegressor(**default_params)
