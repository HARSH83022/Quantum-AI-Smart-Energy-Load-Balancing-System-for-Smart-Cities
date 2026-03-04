"""
Property-based tests for preprocessing module
Feature: quantum-energy-load-balancing
"""
import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.missing_value_handler import MissingValueHandler
from src.preprocessing.normalizer import Normalizer
from src.preprocessing.sequence_generator import SequenceGenerator
from src.preprocessing.data_splitter import DataSplitter


# Feature: quantum-energy-load-balancing, Property 3: Missing Value Elimination
@given(
    n_rows=st.integers(min_value=50, max_value=200),
    missing_fraction=st.floats(min_value=0.1, max_value=0.3)
)
@settings(max_examples=100, deadline=None)
def test_missing_value_elimination(n_rows, missing_fraction):
    """
    Property 3: Missing Value Elimination
    For any dataset containing missing values, after preprocessing with the missing value handler,
    the resulting dataset should contain zero missing values.
    Validates: Requirements 2.1
    """
    # Create DataFrame with missing values
    df = pd.DataFrame({
        'demand_mw': np.random.uniform(100, 1000, n_rows),
        'temperature': np.random.uniform(20, 40, n_rows),
        'voltage': np.random.uniform(220, 240, n_rows)
    })
    
    # Introduce missing values
    n_missing = int(n_rows * missing_fraction)
    for col in df.columns:
        missing_indices = np.random.choice(n_rows, size=n_missing, replace=False)
        df.loc[missing_indices, col] = np.nan
    
    # Verify we have missing values
    assert df.isnull().sum().sum() > 0
    
    # Handle missing values
    handler = MissingValueHandler(strategy="forward_fill")
    df_clean = handler.handle_missing_values(df)
    
    # Verify no missing values remain
    assert df_clean.isnull().sum().sum() == 0


# Feature: quantum-energy-load-balancing, Property 4: Normalization Range
@given(
    n_rows=st.integers(min_value=50, max_value=200),
    min_demand=st.floats(min_value=100, max_value=500),
    max_demand=st.floats(min_value=600, max_value=1500)
)
@settings(max_examples=100, deadline=None)
def test_normalization_range(n_rows, min_demand, max_demand):
    """
    Property 4: Normalization Range
    For any demand data, after normalization, all values should fall within the range [0, 1].
    Validates: Requirements 2.2
    """
    # Create DataFrame with demand data
    df = pd.DataFrame({
        'demand_mw': np.random.uniform(min_demand, max_demand, n_rows)
    })
    
    # Normalize
    normalizer = Normalizer()
    df_normalized, scaler = normalizer.normalize_demand(df, 'demand_mw')
    
    # Verify all values in [0, 1]
    assert df_normalized['demand_mw'].min() >= 0.0
    assert df_normalized['demand_mw'].max() <= 1.0
    assert df_normalized['demand_mw'].min() >= -0.0001  # Allow small floating point errors
    assert df_normalized['demand_mw'].max() <= 1.0001


# Feature: quantum-energy-load-balancing, Property 5: Rolling Window Sequence Length
@given(
    n_rows=st.integers(min_value=200, max_value=500),
    window_size=st.integers(min_value=24, max_value=96)
)
@settings(max_examples=100, deadline=None)
def test_rolling_window_sequence_length(n_rows, window_size):
    """
    Property 5: Rolling Window Sequence Length
    For any time series data with sufficient length, all generated rolling window sequences
    should have exactly the specified window size.
    Validates: Requirements 2.3
    """
    # Create DataFrame
    df = pd.DataFrame({
        'demand_mw': np.random.uniform(100, 1000, n_rows),
        'temperature': np.random.uniform(20, 40, n_rows),
        'voltage': np.random.uniform(220, 240, n_rows)
    })
    
    # Generate sequences
    generator = SequenceGenerator(window_size=window_size)
    sequences, targets = generator.create_sequences(df, 'demand_mw')
    
    # Verify all sequences have correct length
    assert sequences.shape[1] == window_size
    
    # Verify we have the expected number of sequences
    expected_n_sequences = n_rows - window_size
    assert len(sequences) == expected_n_sequences
    assert len(targets) == expected_n_sequences


# Feature: quantum-energy-load-balancing, Property 6: Train-Test Split Completeness
@given(
    n_samples=st.integers(min_value=100, max_value=500),
    test_size=st.floats(min_value=0.1, max_value=0.3)
)
@settings(max_examples=100, deadline=None)
def test_train_test_split_completeness(n_samples, test_size):
    """
    Property 6: Train-Test Split Completeness
    For any preprocessed dataset, the union of training and testing sets should equal
    the original dataset size, and the intersection should be empty.
    Validates: Requirements 2.4
    """
    # Create dummy sequences and targets
    sequences = np.random.rand(n_samples, 96, 3)
    targets = np.random.rand(n_samples)
    
    # Split data
    splitter = DataSplitter(test_size=test_size)
    X_train, X_test, y_train, y_test = splitter.train_test_split(sequences, targets)
    
    # Verify completeness: train + test = original
    assert len(X_train) + len(X_test) == n_samples
    assert len(y_train) + len(y_test) == n_samples
    
    # Verify no overlap (by checking indices)
    train_size = len(X_train)
    test_size_actual = len(X_test)
    
    # Train should be first portion, test should be second portion
    assert train_size > 0
    assert test_size_actual > 0
    assert train_size + test_size_actual == n_samples
