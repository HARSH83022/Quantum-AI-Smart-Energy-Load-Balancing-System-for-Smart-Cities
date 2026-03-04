"""
Tests for robust QUBO formulation
Feature: quantum-energy-load-balancing
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.optimization.robust_qubo.robust_builder import RobustQUBOBuilder


# Feature: quantum-energy-load-balancing, Property 24: Robust QUBO Risk Term Inclusion
def test_robust_qubo_risk_term_inclusion():
    """
    Property 24: Robust QUBO Risk Term Inclusion
    For any robust QUBO formulation, the objective function should include 
    the risk penalty term weighted by scenario probabilities.
    Validates: Requirements 15.2, 15.3
    """
    # Create scenarios
    scenarios = np.array([
        [300, 400, 500],
        [350, 450, 550],
        [280, 380, 480]
    ])
    scenario_probs = np.array([0.33, 0.34, 0.33])
    capacities = np.array([1000, 1200])
    
    # Build robust QUBO
    builder = RobustQUBOBuilder(delta=15.0, mode="scenario-based")
    Q, metadata = builder.build_robust_qubo(scenarios, scenario_probs, capacities)
    
    # Verify risk weight is included
    assert metadata['delta'] == 15.0
    assert metadata['mode'] == "scenario-based"
    assert metadata['n_scenarios'] == 3
    
    # Verify matrix is non-zero (contains risk terms)
    assert np.any(Q != 0), "Robust QUBO should contain risk penalty terms"
    
    # Verify symmetry is maintained
    assert metadata['is_symmetric'] == True
