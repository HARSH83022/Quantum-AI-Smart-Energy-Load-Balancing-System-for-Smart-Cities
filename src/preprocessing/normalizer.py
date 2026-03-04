"""
Data normalizer using MinMaxScaler
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class Normalizer:
    """Normalizes demand data to [0, 1] range"""
    
    def __init__(self):
        """Initialize normalizer with MinMaxScaler"""
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
        
    def normalize_demand(self, df: pd.DataFrame, demand_col: str = "demand_mw") -> Tuple[pd.DataFrame, MinMaxScaler]:
        """
        Normalize demand values to [0, 1] range
        
        Args:
            df: DataFrame with demand data
            demand_col: Name of demand column
            
        Returns:
            Tuple of (normalized DataFrame, fitted scaler)
        """
        logger.info(f"Normalizing {demand_col} column")
        
        df_normalized = df.copy()
        
        # Extract demand values
        demand_values = df_normalized[demand_col].values.reshape(-1, 1)
        
        # Fit and transform
        normalized_values = self.scaler.fit_transform(demand_values)
        self.is_fitted = True
        
        # Update DataFrame
        df_normalized[demand_col] = normalized_values.flatten()
        
        logger.info(f"Normalized {demand_col}: min={normalized_values.min():.4f}, max={normalized_values.max():.4f}")
        
        return df_normalized, self.scaler
    
    def inverse_transform(self, normalized_values: np.ndarray) -> np.ndarray:
        """
        Convert normalized values back to original scale
        
        Args:
            normalized_values: Normalized values in [0, 1]
            
        Returns:
            Values in original scale
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
        
        if normalized_values.ndim == 1:
            normalized_values = normalized_values.reshape(-1, 1)
            
        original_values = self.scaler.inverse_transform(normalized_values)
        return original_values.flatten()
