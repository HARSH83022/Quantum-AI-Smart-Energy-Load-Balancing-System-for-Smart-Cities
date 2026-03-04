"""
CSV data loader implementation
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Optional
from .base import DataSource

logger = logging.getLogger(__name__)


class CSVDataLoader(DataSource):
    """Loads data from CSV files"""
    
    def __init__(self, file_path: str):
        """
        Initialize CSV loader
        
        Args:
            file_path: Path to CSV file
        """
        self.file_path = Path(file_path)
        
    async def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If schema validation fails
        """
        if not self.file_path.exists():
            error_msg = f"CSV file not found: {self.file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            logger.info(f"Loading data from {self.file_path}")
            df = pd.read_csv(self.file_path)
            logger.info(f"Loaded {len(df)} rows from CSV")
            
            # Validate schema
            if not await self.validate_schema(df):
                missing = self._check_required_columns(df)
                error_msg = f"Invalid CSV schema. Missing columns: {missing}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
            
            logger.info("CSV data loaded and validated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    async def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate DataFrame schema
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        missing = self._check_required_columns(df)
        
        if missing:
            logger.warning(f"Schema validation failed. Missing columns: {missing}")
            return False
        
        logger.info("Schema validation passed")
        return True
    
    async def store_to_database(self, df: pd.DataFrame, session):
        """
        Store loaded data to database
        
        Args:
            df: DataFrame to store
            session: Database session
        """
        from src.database.models import RawData
        
        logger.info(f"Storing {len(df)} rows to database")
        
        # Convert DataFrame to database records
        records = []
        for _, row in df.iterrows():
            record = RawData(
                timestamp=row['timestamp'],
                zone=row['zone'],
                demand_mw=row['demand_MW'],
                transformer=row['transformer'],
                capacity_mw=row['capacity_MW'],
                voltage=row.get('voltage'),
                current=row.get('current'),
                temperature=row.get('temperature'),
                humidity=row.get('humidity'),
                hour=row.get('hour'),
                day_of_week=row.get('day_of_week'),
                month=row.get('month')
            )
            records.append(record)
        
        # Bulk insert
        session.add_all(records)
        await session.commit()
        
        logger.info(f"Successfully stored {len(records)} records to database")
