"""
Property-based and unit tests for CSV loader
Feature: quantum-energy-load-balancing
"""
import pytest
import pandas as pd
from hypothesis import given, strategies as st, settings
import sys
import os
from io import StringIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_sources.csv_loader import CSVDataLoader


# Feature: quantum-energy-load-balancing, Property 1: CSV Schema Validation
@pytest.mark.asyncio
@given(
    missing_cols=st.lists(
        st.sampled_from(["timestamp", "zone", "demand_MW", "transformer", "capacity_MW", "temperature"]),
        min_size=1,
        max_size=3,
        unique=True
    )
)
@settings(max_examples=100, deadline=None)
async def test_csv_schema_validation_rejects_invalid(missing_cols):
    """
    Property 1: CSV Schema Validation
    For any CSV file with missing or invalid required columns, 
    the validation function should reject the file and return an error.
    Validates: Requirements 1.3
    """
    # Create DataFrame with some columns missing
    all_cols = ["timestamp", "zone", "demand_MW", "transformer", "capacity_MW", "temperature", "voltage", "current"]
    valid_cols = [col for col in all_cols if col not in missing_cols]
    
    df = pd.DataFrame({col: [1, 2, 3] for col in valid_cols})
    
    loader = CSVDataLoader("dummy.csv")
    is_valid = await loader.validate_schema(df)
    
    # Should be invalid since we removed required columns
    assert not is_valid


# Feature: quantum-energy-load-balancing, Property 19: CSV Column Parsing Completeness
@pytest.mark.asyncio
async def test_csv_column_parsing_completeness():
    """
    Property 19: CSV Column Parsing Completeness
    For any valid CSV file with all required columns, the parser should 
    successfully extract all column values.
    Validates: Requirements 1.2
    """
    # Create valid CSV data
    csv_data = """timestamp,zone,demand_MW,transformer,capacity_MW,voltage,current,temperature,humidity,hour,day_of_week,month
01-04-2023 00:00,North,275.47,T1,1200,231.92,1187.82,30.51,49.03,0,5,4
01-04-2023 00:00,South,461.31,T2,1000,230.51,2001.25,33.55,44.43,0,5,4"""
    
    # Write to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(csv_data)
        temp_path = f.name
    
    try:
        loader = CSVDataLoader(temp_path)
        df = await loader.load_data()
        
        # Verify all columns are present
        required_cols = ["timestamp", "zone", "demand_MW", "transformer", "capacity_MW", "temperature"]
        for col in required_cols:
            assert col in df.columns or 'timestamp' == col  # timestamp gets parsed
        
        # Verify data was parsed
        assert len(df) == 2
        assert df['zone'].tolist() == ["North", "South"]
        
    finally:
        os.unlink(temp_path)


# Unit test for CSV loader with valid file
@pytest.mark.asyncio
async def test_csv_loader_with_valid_file():
    """
    Unit test: CSV loader with valid file
    Test loading a valid CSV file and verify all expected columns are present.
    Requirements: 1.1, 1.2
    """
    csv_data = """timestamp,zone,demand_MW,transformer,capacity_MW,voltage,current,temperature,humidity,hour,day_of_week,month
01-04-2023 00:00,North,275.47,T1,1200,231.92,1187.82,30.51,49.03,0,5,4
01-04-2023 00:15,South,461.31,T2,1000,230.51,2001.25,33.55,44.43,0,5,4
01-04-2023 00:30,East,479.92,T3,900,231.14,2076.32,33.19,45.35,0,5,4"""
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(csv_data)
        temp_path = f.name
    
    try:
        loader = CSVDataLoader(temp_path)
        df = await loader.load_data()
        
        # Verify file was loaded
        assert df is not None
        assert len(df) == 3
        
        # Verify all expected columns are present
        expected_cols = ["zone", "demand_MW", "transformer", "capacity_MW", "voltage", "current", "temperature", "humidity"]
        for col in expected_cols:
            assert col in df.columns
        
        # Verify timestamp was parsed
        assert 'timestamp' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
        
    finally:
        os.unlink(temp_path)
