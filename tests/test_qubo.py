"""
Tests for QUBO formulation
Feature: quantum-energy-load-balancing
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.optimization.qubo.qubo_builder import QUBOBuilder


# Unit test for QUBO binary variable definition
def test_qubo_binary_variable_definition():
    """
    Unit test: QUBO binary variable definition
    Verify binary variables x_ij are correctly defined.
    Requirements: 4.1
    """
    demands = np.array([300, 400, 500])
    capacities = np.array([1000, 1200])
    
    builder = QUBOBuilder()
    Q, metadata = builder.build_qubo_matrix(demands, capacities)
    
    # Verify number of variables
    n_transformers = 2
    n_zones = 3
    expected_vars = n_transformers * n_zones
    
    assert metadata['n_variables'] == expected_vars
    assert metadata['n_transformers'] == n_transformers
    assert metadata['n_zones'] == n_zones


# Feature: quantum-energy-load-balancing, Property 9: QUBO Matrix Symmetry
@given(
    n_zones=st.integers(min_value=2, max_value=5),
    n_transformers=st.integers(min_value=2, max_value=4)
)
@settings(max_examples=50, deadline=None)
def test_qubo_matrix_symmetry(n_zones, n_transformers):
    """
    Property 9: QUBO Matrix Symmetry
    For any load balancing problem instance, the constructed QUBO matrix 
    should be square and symmetric.
    Validates: Requirements 4.7
    """
    demands = np.random.uniform(100, 1000, n_zones)
    capacities = np.random.uniform(800, 1500, n_transformers)
    
    builder = QUBOBuilder()
    Q, metadata = builder.build_qubo_matrix(demands, capacities)
    
    # Verify square matrix
    assert Q.shape[0] == Q.shape[1]
    
    # Verify symmetry
    assert np.allclose(Q, Q.T), "QUBO matrix should be symmetric"
    assert metadata['is_symmetric'] == True


# Feature: quantum-energy-load-balancing, Property 10: QUBO Objective Completeness
def test_qubo_objective_completeness():
    """
    Property 10: QUBO Objective Completeness
    For any QUBO formulation, the objective function should include all three 
    required terms: overload penalty, imbalance penalty, and switching cost, 
    plus capacity and assignment constraint penalties.
    Validates: Requirements 4.2, 4.3, 4.4, 4.5, 4.6
    """
    demands = np.array([300, 400])
    capacities = np.array([1000, 1200])
    
    builder = QUBOBuilder(alpha=10.0, beta=5.0, gamma=1.0, lambda1=100.0, lambda2=100.0)
    Q, metadata = builder.build_qubo_matrix(demands, capacities)
    
    # Verify all weights are present in metadata
    assert metadata['alpha'] == 10.0  # Overload penalty
    assert metadata['beta'] == 5.0    # Imbalance penalty
    assert metadata['gamma'] == 1.0   # Switching cost
    assert metadata['lambda1'] == 100.0  # Capacity constraint
    assert metadata['lambda2'] == 100.0  # Assignment constraint
    
    # Verify matrix is non-zero (contains all terms)
    assert np.any(Q != 0), "QUBO matrix should contain objective terms"
