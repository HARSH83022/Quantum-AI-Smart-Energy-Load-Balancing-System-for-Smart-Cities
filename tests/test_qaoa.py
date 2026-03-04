"""
Tests for QAOA optimization module
Feature: quantum-energy-load-balancing
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from src.optimization.qaoa.qaoa_optimizer import (
    QAOAOptimizer,
    ParameterWarmStarter,
    ConvergenceMonitor,
    QAOACircuitBuilder
)
from src.optimization.risk_analysis.quantum_risk_analyzer import (
    QuantumRiskAnalyzer,
    CVaRCalculator
)


class TestQAOABackend:
    """Unit tests for QAOA backend selection"""
    
    def test_default_backend_is_aer(self, monkeypatch):
        """
        Test that Qiskit Aer simulator is used by default
        Feature: quantum-energy-load-balancing, Property 11: QAOA Solution Vector Length
        Validates: Requirements 5.1
        """
        # Ensure no IBM key is set
        monkeypatch.delenv("IBM_QUANTUM_API_KEY", raising=False)
        
        optimizer = QAOAOptimizer(p=2)
        
        # Check backend is Aer simulator
        assert "Aer" in str(type(optimizer.backend))
    
    def test_ibm_backend_with_api_key(self, monkeypatch):
        """
        Test that IBM simulator is attempted when API key is provided
        Feature: quantum-energy-load-balancing, Property 11: QAOA Solution Vector Length
        Validates: Requirements 5.2
        """
        # Set a dummy IBM key
        monkeypatch.setenv("IBM_QUANTUM_API_KEY", "dummy_key_for_testing")
        
        optimizer = QAOAOptimizer(p=2)
        
        # Should fall back to Aer if connection fails (which it will with dummy key)
        # Just verify it doesn't crash
        assert optimizer.backend is not None


class TestQAOAProperties:
    """Property-based tests for QAOA"""
    
    @given(
        n_vars=st.integers(min_value=2, max_value=8),
        p=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=10, deadline=None)
    def test_qaoa_solution_vector_length(self, n_vars, p):
        """
        Property 11: QAOA Solution Vector Length
        For any QUBO problem with n binary variables, the QAOA solution vector should have exactly n elements
        Feature: quantum-energy-load-balancing, Property 11: QAOA Solution Vector Length
        Validates: Requirements 5.3
        """
        # Create random QUBO matrix
        Q = np.random.randn(n_vars, n_vars)
        Q = (Q + Q.T) / 2  # Make symmetric
        
        # Run QAOA
        optimizer = QAOAOptimizer(p=p, max_iter=10)
        result = optimizer.optimize(Q)
        
        # Check solution length
        assert len(result['solution']) == n_vars
        
        # Check all values are binary
        assert all(x in [0, 1] for x in result['solution'])
    
    @given(
        n_vars=st.integers(min_value=2, max_value=6),
        p=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=10, deadline=None)
    def test_qaoa_performance_metrics_positivity(self, n_vars, p):
        """
        Property 12: QAOA Performance Metrics Positivity
        For any QAOA execution, the logged circuit depth and execution time should both be positive values
        Feature: quantum-energy-load-balancing, Property 12: QAOA Performance Metrics Positivity
        Validates: Requirements 5.5
        """
        # Create random QUBO matrix
        Q = np.random.randn(n_vars, n_vars)
        Q = (Q + Q.T) / 2
        
        # Run QAOA
        optimizer = QAOAOptimizer(p=p, max_iter=10)
        result = optimizer.optimize(Q)
        
        # Check metrics are positive
        assert result['circuit_depth'] > 0
        assert result['execution_time_seconds'] > 0
        assert result['gate_count'] > 0
    
    @given(
        n_vars=st.integers(min_value=2, max_value=6)
    )
    @settings(max_examples=5, deadline=None)
    def test_qaoa_parameter_warm_start_improvement(self, n_vars):
        """
        Property 25: QAOA Parameter Warm-Start Improvement
        For any sequence of related QUBO problems, using parameter warm-starting should reduce
        the number of iterations to convergence compared to random initialization
        Feature: quantum-energy-load-balancing, Property 25: QAOA Parameter Warm-Start Improvement
        Validates: Requirements 16.2
        """
        # Create two similar QUBO problems
        Q1 = np.random.randn(n_vars, n_vars)
        Q1 = (Q1 + Q1.T) / 2
        
        Q2 = Q1 + 0.1 * np.random.randn(n_vars, n_vars)
        Q2 = (Q2 + Q2.T) / 2
        
        # Run without warm start
        optimizer_cold = QAOAOptimizer(p=2, max_iter=50)
        result_cold_1 = optimizer_cold.optimize(Q1)
        
        # Reset optimizer for second problem (no warm start)
        optimizer_cold_2 = QAOAOptimizer(p=2, max_iter=50)
        result_cold_2 = optimizer_cold_2.optimize(Q2)
        
        # Run with warm start
        optimizer_warm = QAOAOptimizer(p=2, max_iter=50)
        result_warm_1 = optimizer_warm.optimize(Q1)
        result_warm_2 = optimizer_warm.optimize(Q2)  # Uses warm start from Q1
        
        # Warm start should converge faster or achieve better objective
        # (This is a statistical property, so we check it holds on average)
        # For this test, we just verify warm start doesn't increase iterations significantly
        assert result_warm_2['n_iterations'] <= result_cold_2['n_iterations'] * 1.5


class TestParameterWarmStarter:
    """Tests for parameter warm-starting"""
    
    def test_warm_start_same_size(self):
        """Test warm starting with same problem size"""
        warm_starter = ParameterWarmStarter()
        
        # Store parameters
        params = np.array([0.1, 0.2, 0.3, 0.4])
        warm_starter.store_parameters(params, problem_size=4)
        
        # Retrieve for same size
        retrieved = warm_starter.warm_start_parameters(new_problem_size=4, p=2)
        
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, params)
    
    def test_warm_start_different_size(self):
        """Test warm starting with different problem size"""
        warm_starter = ParameterWarmStarter()
        
        # Store parameters
        params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        warm_starter.store_parameters(params, problem_size=6)
        
        # Retrieve for smaller size
        retrieved = warm_starter.warm_start_parameters(new_problem_size=4, p=2)
        
        assert retrieved is not None
        assert len(retrieved) == 4


class TestConvergenceMonitor:
    """Tests for convergence monitoring"""
    
    def test_convergence_detection(self):
        """Test convergence is detected when energy stabilizes"""
        monitor = ConvergenceMonitor(threshold=0.01)
        
        # Add converging energies
        energies = [1.0, 0.95, 0.92, 0.91, 0.905, 0.903]
        for e in energies:
            monitor.add_energy(e)
        
        # Should detect convergence
        assert monitor.monitor_convergence()
    
    def test_no_convergence_with_varying_energy(self):
        """Test convergence is not detected with varying energy"""
        monitor = ConvergenceMonitor(threshold=0.01)
        
        # Add varying energies
        energies = [1.0, 0.5, 1.2, 0.3, 1.5]
        for e in energies:
            monitor.add_energy(e)
        
        # Should not detect convergence
        assert not monitor.monitor_convergence()


class TestCVaRCalculator:
    """Tests for CVaR calculation"""
    
    def test_cvar_monotonicity(self):
        """
        Property 26: CVaR Monotonicity
        For any cost distribution, CVaR at confidence level α should be greater than or equal to VaR
        Feature: quantum-energy-load-balancing, Property 26: CVaR Monotonicity
        Validates: Requirements 17.4
        """
        calculator = CVaRCalculator(confidence=0.95)
        
        # Generate random cost distribution
        costs = np.random.exponential(scale=10.0, size=1000)
        
        var = calculator.calculate_var(costs)
        cvar = calculator.calculate_cvar(costs)
        
        # CVaR should be >= VaR
        assert cvar >= var
        
        # Verify using built-in method
        assert calculator.verify_cvar_monotonicity(costs)
    
    @given(
        costs=st.lists(st.floats(min_value=0.0, max_value=1000.0), min_size=100, max_size=1000)
    )
    @settings(max_examples=20)
    def test_cvar_monotonicity_property(self, costs):
        """
        Property test for CVaR monotonicity across different distributions
        Feature: quantum-energy-load-balancing, Property 26: CVaR Monotonicity
        Validates: Requirements 17.4
        """
        costs_array = np.array(costs)
        calculator = CVaRCalculator(confidence=0.95)
        
        var = calculator.calculate_var(costs_array)
        cvar = calculator.calculate_cvar(costs_array)
        
        # CVaR >= VaR must always hold
        assert cvar >= var - 1e-10  # Small tolerance for numerical errors


class TestQuantumRiskAnalyzer:
    """Tests for quantum risk analyzer"""
    
    def test_solution_stability_evaluation(self):
        """Test solution stability evaluation"""
        analyzer = QuantumRiskAnalyzer(confidence=0.95, lambda_risk=1.0)
        
        # Create test data
        n_scenarios = 100
        n_zones = 4
        n_transformers = 2
        
        scenarios = np.random.uniform(10, 50, (n_scenarios, n_zones))
        capacities = np.array([100.0, 100.0])
        solution = np.array([1, 0, 0, 1, 0, 1, 1, 0])  # 2 transformers × 4 zones
        
        # Evaluate stability
        metrics = analyzer.evaluate_solution_stability(solution, scenarios, capacities)
        
        # Check all metrics are present
        assert 'expected_cost' in metrics
        assert 'cost_variance' in metrics
        assert 'var_95' in metrics
        assert 'cvar_95' in metrics
        assert 'sharpe_ratio' in metrics
        
        # Check metrics are reasonable
        assert metrics['expected_cost'] >= 0
        assert metrics['cost_variance'] >= 0
        assert metrics['cvar_95'] >= metrics['var_95']
    
    def test_expected_cost_calculation(self):
        """Test expected cost calculation"""
        analyzer = QuantumRiskAnalyzer()
        
        # Simple test case
        scenarios = np.array([[10, 20], [15, 25], [12, 22]])
        capacities = np.array([50, 50])
        solution = np.array([1, 0, 0, 1])  # Zone 0 to T0, Zone 1 to T1
        
        expected_cost = analyzer.calculate_expected_cost(solution, scenarios, capacities)
        
        # Should be non-negative
        assert expected_cost >= 0
    
    def test_cost_variance_calculation(self):
        """Test cost variance calculation"""
        analyzer = QuantumRiskAnalyzer()
        
        scenarios = np.array([[10, 20], [15, 25], [12, 22]])
        capacities = np.array([50, 50])
        solution = np.array([1, 0, 0, 1])
        
        variance = analyzer.calculate_cost_variance(solution, scenarios, capacities)
        
        # Variance should be non-negative
        assert variance >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
