import pandas as pd
import numpy as np
from typing import List, Union

def create_lag_features(df: pd.DataFrame, 
                       columns: List[str], 
                       lags: List[int]) -> pd.DataFrame:
    """Create lag features for specified columns."""
    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df.groupby('SecuritiesCode')[col].shift(lag)
    return df

def create_rolling_features(df: pd.DataFrame, 
                          columns: List[str], 
                          windows: List[int]) -> pd.DataFrame:
    """Create rolling mean and std features."""
    df = df.copy()
    for col in columns:
        for window in windows:
            df[f'{col}_rolling_mean_{window}'] = (
                df.groupby('SecuritiesCode')[col]
                .rolling(window)
                .mean()
                .reset_index(0, drop=True)
            )
            df[f'{col}_rolling_std_{window}'] = (
                df.groupby('SecuritiesCode')[col]
                .rolling(window)
                .std()
                .reset_index(0, drop=True)
            )
    return df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features."""
    df = df.copy()
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year
    df['DayOfMonth'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    return df

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators."""
    df = df.copy()
    
    # Group by SecuritiesCode and calculate indicators
    for security in df['SecuritiesCode'].unique():
        mask = df['SecuritiesCode'] == security
        
        # RSI
        delta = df.loc[mask, 'Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df.loc[mask, 'RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df.loc[mask, 'Close'].ewm(span=12, adjust=False).mean()
        exp2 = df.loc[mask, 'Close'].ewm(span=26, adjust=False).mean()
        df.loc[mask, 'MACD'] = exp1 - exp2
        df.loc[mask, 'MACD_Signal'] = df.loc[mask, 'MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        rolling_mean = df.loc[mask, 'Close'].rolling(window=20).mean()
        rolling_std = df.loc[mask, 'Close'].rolling(window=20).std()
        df.loc[mask, 'BB_upper'] = rolling_mean + (rolling_std * 2)
        df.loc[mask, 'BB_middle'] = rolling_mean
        df.loc[mask, 'BB_lower'] = rolling_mean - (rolling_std * 2)
    
    return df
