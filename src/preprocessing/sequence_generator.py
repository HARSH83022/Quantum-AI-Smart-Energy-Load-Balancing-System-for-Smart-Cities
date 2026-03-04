"""
Rolling window sequence generator for time series
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class SequenceGenerator:
    """Generates rolling window sequences for LSTM training"""
    
    def __init__(self, window_size: int = 96):
        """
        Initialize sequence generator
        
        Args:
            window_size: Number of time steps in each sequence (default: 96 for 24 hours at 15-min intervals)
        """
        self.window_size = window_size
        
    def create_sequences(self, df: pd.DataFrame, target_col: str = "demand_mw") -> Tuple[np.ndarray, np.ndarray]:
        """
        Create rolling window sequences from time series data
        
        Args:
            df: DataFrame with time series data
            target_col: Column to use as target
            
        Returns:
            Tuple of (sequences, targets)
            - sequences: shape (n_samples, window_size, n_features)
            - targets: shape (n_samples,)
        """
        logger.info(f"Creating sequences with window size: {self.window_size}")
        
        # Select numeric columns for features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Ensure target column is in numeric columns
        if target_col not in numeric_cols:
            raise ValueError(f"Target column '{target_col}' not found in numeric columns")
        
        data = df[numeric_cols].values
        
        sequences = []
        targets = []
        
        for i in range(len(data) - self.window_size):
            # Extract sequence
            seq = data[i:i + self.window_size]
            # Extract target (next value after sequence)
            target_idx = numeric_cols.index(target_col)
            target = data[i + self.window_size, target_idx]
            
            sequences.append(seq)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        logger.info(f"Created {len(sequences)} sequences with shape {sequences.shape}")
        
        return sequences, targets
