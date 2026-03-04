"""
Train-test data splitter
"""
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class DataSplitter:
    """Splits data into training and testing sets"""
    
    def __init__(self, test_size: float = 0.2):
        """
        Initialize data splitter
        
        Args:
            test_size: Fraction of data to use for testing (default: 0.2)
        """
        self.test_size = test_size
        
    def train_test_split(self, sequences: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split sequences into training and testing sets
        
        Args:
            sequences: Input sequences
            targets: Target values
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data with test_size={self.test_size}")
        
        n_samples = len(sequences)
        split_idx = int(n_samples * (1 - self.test_size))
        
        X_train = sequences[:split_idx]
        X_test = sequences[split_idx:]
        y_train = targets[:split_idx]
        y_test = targets[split_idx:]
        
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Verify no overlap
        assert len(X_train) + len(X_test) == n_samples, "Train and test sets don't sum to total"
        
        return X_train, X_test, y_train, y_test
