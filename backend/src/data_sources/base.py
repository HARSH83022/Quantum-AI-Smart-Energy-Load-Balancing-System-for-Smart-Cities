"""
Abstract base class for data sources
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import List


class DataSource(ABC):
    """Abstract interface for data sources"""
    
    REQUIRED_COLUMNS = [
        "timestamp",
        "zone",
        "demand_MW",
        "transformer",
        "capacity_MW",
        "temperature"
    ]
    
    @abstractmethod
    async def load_data(self) -> pd.DataFrame:
        """
        Load data and return as DataFrame with standard schema
        
        Returns:
            pd.DataFrame: Loaded data with standard columns
        """
        pass
    
    @abstractmethod
    async def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required columns
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if schema is valid, False otherwise
        """
        pass
    
    def _check_required_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Check for missing required columns
        
        Args:
            df: DataFrame to check
            
        Returns:
            List of missing column names
        """
        df_columns = set(df.columns)
        required = set(self.REQUIRED_COLUMNS)
        missing = required - df_columns
        return list(missing)
