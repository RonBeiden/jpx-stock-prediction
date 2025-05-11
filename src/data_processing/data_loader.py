import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

class JPXDataLoader:
    """Data loader for JPX stock market data."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = Path(data_dir)
        self.stock_prices: Optional[pd.DataFrame] = None
        self.financials: Optional[pd.DataFrame] = None
        self.stock_info: Optional[pd.DataFrame] = None
        
    def load_stock_prices(self) -> pd.DataFrame:
        """Load and process stock prices data."""
        self.stock_prices = pd.read_csv(
            self.data_dir / "stock_prices.csv",
            parse_dates=["Date"]
        )
        return self.stock_prices
    
    def load_financials(self) -> pd.DataFrame:
        """Load and process financial statements data."""
        self.financials = pd.read_csv(
            self.data_dir / "financials.csv",
            parse_dates=["Date"]
        )
        return self.financials
    
    def load_stock_info(self) -> pd.DataFrame:
        """Load and process stock information data."""
        self.stock_info = pd.read_csv(
            self.data_dir / "stock_info.csv"
        )
        return self.stock_info
    
    def create_features(self, window_sizes: List[int] = [5, 10, 21]) -> pd.DataFrame:
        """
        Create features from stock prices data.
        
        Args:
            window_sizes: List of window sizes for rolling calculations
            
        Returns:
            DataFrame with calculated features
        """
        if self.stock_prices is None:
            self.load_stock_prices()
            
        df = self.stock_prices.copy()
        
        # Calculate returns
        df["Return_1D"] = df.groupby("SecuritiesCode")["Close"].pct_change()
        
        # Create lag features
        df["Return_1D_Lag1"] = df.groupby("SecuritiesCode")["Return_1D"].shift(1)
        
        # Create rolling features
        for window in window_sizes:
            # Rolling mean of returns
            df[f"Return_Mean_{window}D"] = df.groupby("SecuritiesCode")["Return_1D"].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            # Rolling std of returns
            df[f"Return_Std_{window}D"] = df.groupby("SecuritiesCode")["Return_1D"].rolling(
                window=window, min_periods=1
            ).std().reset_index(0, drop=True)
            
            # Rolling volume features
            df[f"Volume_Mean_{window}D"] = df.groupby("SecuritiesCode")["Volume"].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
        
        # Market features
        df["Market_Return"] = df.groupby("Date")["Return_1D"].transform("mean")
        df["Market_Return_Std"] = df.groupby("Date")["Return_1D"].transform("std")
        
        # Time features
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["Month"] = df["Date"].dt.month
        
        return df
    
    def prepare_training_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepare final training dataset with all features.
        
        Args:
            start_date: Start date for filtering data
            end_date: End date for filtering data
            
        Returns:
            DataFrame ready for model training
        """
        # Create features
        df = self.create_features()
        
        # Add stock info if available
        if self.stock_info is None:
            self.load_stock_info()
        df = df.merge(self.stock_info, on="SecuritiesCode", how="left")
        
        # Filter dates if specified
        if start_date:
            df = df[df["Date"] >= start_date]
        if end_date:
            df = df[df["Date"] <= end_date]
        
        # Handle missing values
        df = df.fillna(method="ffill")
        
        return df 