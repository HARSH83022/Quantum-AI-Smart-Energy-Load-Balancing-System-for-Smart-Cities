"""
Main preprocessing pipeline
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from .missing_value_handler import MissingValueHandler
from .normalizer import Normalizer
from .sequence_generator import SequenceGenerator
from .data_splitter import DataSplitter

logger = logging.getLogger(__name__)


class Preprocessor:
    """Main preprocessing pipeline for smart grid data"""
    
    def __init__(
        self,
        missing_strategy: str = "forward_fill",
        window_size: int = 96,
        test_size: float = 0.2
    ):
        """
        Initialize preprocessor
        
        Args:
            missing_strategy: Strategy for handling missing values
            window_size: Size of rolling window for sequences
            test_size: Fraction of data for testing
        """
        self.missing_handler = MissingValueHandler(strategy=missing_strategy)
        self.normalizer = Normalizer()
        self.sequence_generator = SequenceGenerator(window_size=window_size)
        self.data_splitter = DataSplitter(test_size=test_size)
        
    def preprocess(
        self,
        df: pd.DataFrame,
        target_col: str = "demand_mw"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run full preprocessing pipeline
        
        Args:
            df: Raw DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Starting preprocessing pipeline")
        
        # Step 1: Handle missing values
        df_clean = self.missing_handler.handle_missing_values(df)
        
        # Step 2: Normalize demand
        df_normalized, scaler = self.normalizer.normalize_demand(df_clean, target_col)
        
        # Step 3: Create sequences
        sequences, targets = self.sequence_generator.create_sequences(df_normalized, target_col)
        
        # Step 4: Train-test split
        X_train, X_test, y_train, y_test = self.data_splitter.train_test_split(sequences, targets)
        
        logger.info("Preprocessing pipeline completed")
        logger.info(f"Training set: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Test set: X={X_test.shape}, y={y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    async def store_preprocessed_data(self, X_train, X_test, y_train, y_test, session):
        """
        Store preprocessed data to database
        
        Args:
            X_train, X_test, y_train, y_test: Preprocessed data
            session: Database session
        """
        from src.database.models import PreprocessedData
        
        logger.info("Storing preprocessed data to database")
        
        records = []
        
        # Store training data
        for i in range(len(X_train)):
            record = PreprocessedData(
                sequence_data=X_train[i].tolist(),
                target_data=[float(y_train[i])],
                is_training=True
            )
            records.append(record)
        
        # Store test data
        for i in range(len(X_test)):
            record = PreprocessedData(
                sequence_data=X_test[i].tolist(),
                target_data=[float(y_test[i])],
                is_training=False
            )
            records.append(record)
        
        session.add_all(records)
        await session.commit()
        
        logger.info(f"Stored {len(records)} preprocessed records to database")
