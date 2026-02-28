"""
Missing value handler for data preprocessing
"""
import pandas as pd
import numpy as np
import logging
from typing import Literal

logger = logging.getLogger(__name__)


class MissingValueHandler:
    """Handles missing values in time series data"""
    
    def __init__(self, strategy: Literal["forward_fill", "mean", "drop"] = "forward_fill"):
        """
        Initialize missing value handler
        
        Args:
            strategy: Strategy for handling missing values
                - forward_fill: Forward fill missing values
                - mean: Fill with column mean
                - drop: Drop rows with missing values
        """
        self.strategy = strategy
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in DataFrame
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        initial_missing = df.isnull().sum().sum()
        logger.info(f"Handling {initial_missing} missing values using strategy: {self.strategy}")
        
        if initial_missing == 0:
            logger.info("No missing values found")
            return df
        
        df_clean = df.copy()
        
        if self.strategy == "forward_fill":
            df_clean = df_clean.fillna(method='ffill')
            # Backward fill any remaining NaN at the start
            df_clean = df_clean.fillna(method='bfill')
            
        elif self.strategy == "mean":
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    mean_val = df_clean[col].mean()
                    df_clean[col].fillna(mean_val, inplace=True)
                    
        elif self.strategy == "drop":
            df_clean = df_clean.dropna()
            
        final_missing = df_clean.isnull().sum().sum()
        logger.info(f"Missing values after handling: {final_missing}")
        
        return df_clean
