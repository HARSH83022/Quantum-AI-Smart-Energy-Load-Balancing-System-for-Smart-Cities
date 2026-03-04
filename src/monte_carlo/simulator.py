"""
Monte Carlo simulator for stress testing
"""
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """Runs Monte Carlo stress testing across demand scenarios"""
    
    def __init__(self):
        """Initialize Monte Carlo simulator"""
        pass
    
    def run_simulation(
        self,
        scenarios: np.ndarray,
        capacities: np.ndarray
    ) -> Dict:
        """
        Run Monte Carlo simulation
        
        Args:
            scenarios: Demand scenarios (n_scenarios, n_zones)
            capacities: Transformer capacities
            
        Returns:
            Dict with risk metrics
        """
        n_scenarios = scenarios.shape[0]
        logger.info(f"Running Monte Carlo simulation with {n_scenarios} scenarios")
        
        # Calculate overload frequencies
        overload_counts = np.zeros(len(capacities))
        
        for scenario in scenarios:
            # Simple load assignment (each zone to nearest transformer)
            loads = self._assign_loads(scenario, len(capacities))
            overloads = loads > capacities
            overload_counts += overloads
        
        overload_frequencies = overload_counts / n_scenarios
        
        # Calculate stress probabilities
        stress_probabilities = {
            f'transformer_{i}': float(freq)
            for i, freq in enumerate(overload_frequencies)
        }
        
        # Calculate expected imbalance
        expected_imbalance = float(np.mean([
            np.std(self._assign_loads(scenario, len(capacities)))
            for scenario in scenarios
        ]))
        
        metrics = {
            'overload_frequencies': overload_frequencies.tolist(),
            'stress_probabilities': stress_probabilities,
            'expected_imbalance': expected_imbalance,
            'n_scenarios': n_scenarios
        }
        
        logger.info(f"Simulation complete: avg overload freq={np.mean(overload_frequencies):.3f}")
        
        return metrics
    
    def _assign_loads(self, demands: np.ndarray, n_transformers: int) -> np.ndarray:
        """Simple load assignment strategy"""
        loads = np.zeros(n_transformers)
        zones_per_transformer = len(demands) // n_transformers
        
        for i in range(n_transformers):
            start_idx = i * zones_per_transformer
            end_idx = start_idx + zones_per_transformer if i < n_transformers - 1 else len(demands)
            loads[i] = np.sum(demands[start_idx:end_idx])
        
        return loads
