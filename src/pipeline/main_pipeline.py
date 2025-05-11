import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any

from src.utils.logger import setup_logger
from src.data_processing.feature_engineering import (
    create_lag_features,
    create_rolling_features,
    create_time_features,
    calculate_technical_indicators
)
from src.models.tree_models import (
    create_lightgbm_model,
    create_xgboost_model,
    create_catboost_model
)
from src.models.lstm_model import create_lstm_model
from src.evaluation.metrics import evaluate_predictions
from src import DATA_DIR

class JPXPipeline:
    def __init__(self, config: Dict = None):
        self.config = config or self.get_default_config()
        self.logger = setup_logger('JPXPipeline')
        
    def get_default_config(self) -> Dict:
        """Default configuration for the pipeline."""
        return {
            'data': {
                'train_path': str(DATA_DIR / 'raw/train'),
                'test_path': str(DATA_DIR / 'raw/test'),
                'processed_path': str(DATA_DIR / 'processed'),
                'submission_path': str(DATA_DIR / 'submissions')
            },
            'features': {
                'lag_features': ['Open', 'High', 'Low', 'Close', 'Volume', 'Target'],
                'lag_periods': [1, 2, 3, 5, 10],
                'rolling_windows': [5, 10, 21],
                'technical_indicators': True
            },
            'model': {
                'type': 'lightgbm',
                'params': {
                    'lightgbm': {
                        'objective': 'regression',
                        'metric': 'rmse',
                        'boosting_type': 'gbdt',
                        'num_leaves': 31,
                        'learning_rate': 0.05,
                        'feature_fraction': 0.9
                    }
                }
            },
            'training': {
                'test_size': 0.2,
                'random_state': 42,
                'cv_folds': 5
            }
        }

    def load_data(self) -> pd.DataFrame:
        """Load and merge all necessary data."""
        self.logger.info("Loading data...")
        
        # Load main datasets
        stock_prices = pd.read_csv(
            Path(self.config['data']['train_path']) / 'stock_prices.csv'
        )
        stock_info = pd.read_csv(
            Path(self.config['data']['train_path']) / 'stock_list.csv'
        )
        
        # Convert date columns
        stock_prices['Date'] = pd.to_datetime(stock_prices['Date'])
        
        # Merge datasets
        df = stock_prices.merge(
            stock_info[['SecuritiesCode', 'Section/Products']], 
            on='SecuritiesCode', 
            how='left'
        )
        
        self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features for the model."""
        self.logger.info("Creating features...")
        
        # Create lag features
        df = create_lag_features(
            df, 
            self.config['features']['lag_features'],
            self.config['features']['lag_periods']
        )
        
        # Create rolling features
        df = create_rolling_features(
            df,
            self.config['features']['lag_features'],
            self.config['features']['rolling_windows']
        )
        
        # Create time features
        df = create_time_features(df)
        
        # Add technical indicators if configured
        if self.config['features']['technical_indicators']:
            df = calculate_technical_indicators(df)
        
        self.logger.info(f"Features created. New shape: {df.shape}")
        return df

    def prepare_train_test_split(self, 
                               df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare train/test split based on time."""
        self.logger.info("Preparing train/test split...")
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Calculate split point
        split_idx = int(len(df) * (1 - self.config['training']['test_size']))
        split_date = df.iloc[split_idx]['Date']
        
        # Split data
        train_df = df[df['Date'] < split_date]
        test_df = df[df['Date'] >= split_date]
        
        self.logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train the selected model."""
        self.logger.info(f"Training {self.config['model']['type']} model...")
        
        model_type = self.config['model']['type']
        model_params = self.config['model']['params'].get(model_type, {})
        
        if model_type == 'lightgbm':
            model = create_lightgbm_model(model_params)
        elif model_type == 'xgboost':
            model = create_xgboost_model(model_params)
        elif model_type == 'catboost':
            model = create_catboost_model(model_params)
        elif model_type == 'lstm':
            # Reshape data for LSTM
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            model = create_lstm_model(input_shape=(1, X_train.shape[2]))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, 
                      model: Any, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """Evaluate the model performance."""
        self.logger.info("Evaluating model...")
        
        predictions = model.predict(X_test)
        metrics = evaluate_predictions(y_test, predictions)
        
        for metric_name, value in metrics.items():
            self.logger.info(f"{metric_name}: {value:.4f}")
        
        return metrics, predictions

    def prepare_submission(self, 
                         predictions: np.ndarray, 
                         test_dates: pd.Series, 
                         securities_codes: pd.Series) -> pd.DataFrame:
        """Prepare submission file."""
        self.logger.info("Preparing submission file...")
        
        submission = pd.DataFrame({
            'Date': test_dates,
            'SecuritiesCode': securities_codes,
            'Target': predictions
        })
        
        # Rank predictions by date
        submission['Rank'] = submission.groupby('Date')['Target'].rank(ascending=False)
        
        # Keep only top 200 predictions per date
        submission = submission[submission['Rank'] <= 200]
        
        return submission

    def run_pipeline(self):
        """Run the complete pipeline."""
        try:
            # Load data
            df = self.load_data()
            
            # Create features
            df = self.create_features(df)
            
            # Handle missing values
            df = df.dropna(subset=['Target'])  # Remove rows with NaN targets
            self.feature_cols = [col for col in df.columns 
                               if col not in ['Date', 'SecuritiesCode', 'Target', 'Section/Products'] 
                               and df[col].dtype in ['int64', 'float64']]
            
            # Fill NaN in features with median
            for col in self.feature_cols:
                df[col] = df[col].fillna(df[col].median())
            
            # Split data
            train_df, test_df = self.prepare_train_test_split(df)
            
            # Prepare features and target
            X_train = train_df[self.feature_cols].values
            y_train = train_df['Target'].values
            X_test = test_df[self.feature_cols].values
            y_test = test_df['Target'].values
            
            # Train model
            model = self.train_model(X_train, y_train)
            
            # Evaluate model
            metrics, predictions = self.evaluate_model(model, X_test, y_test)
            
            return {
                'model': model,
                'metrics': metrics,
                'predictions': predictions,
                'feature_names': self.feature_cols,  # Store feature names
                'y_test': y_test,
                'test_dates': test_df['Date']
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise
