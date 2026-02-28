"""
Probabilistic scenario generator for demand forecasting
"""
import numpy as np
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class ScenarioGenerator:
    """Generates probabilistic demand scenarios for robust optimization"""
    
    def __init__(self, n_scenarios: int = 100, random_seed: int = None):
        """
        Initialize scenario generator
        
        Args:
            n_scenarios: Number of scenarios to generate
            random_seed: Random seed for reproducibility
        """
        self.n_scenarios = n_scenarios
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_scenarios(
        self,
        forecast_mean: np.ndarray,
        variance: float,
        method: str = "gaussian"
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate demand scenarios
        
        Args:
            forecast_mean: Mean forecast values
            variance: Forecast variance
            method: Generation method ('gaussian' or 'bootstrap')
            
        Returns:
            Tuple of (scenarios array, statistics dict)
        """
        logger.info(f"Generating {self.n_scenarios} scenarios using {method} method")
        
        if method == "gaussian":
            scenarios = self._generate_gaussian_scenarios(forecast_mean, variance)
        elif method == "bootstrap":
            scenarios = self._generate_bootstrap_scenarios(forecast_mean, variance)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate statistics
        stats = {
            'mean': float(np.mean(scenarios)),
            'std': float(np.std(scenarios)),
            'min': float(np.min(scenarios)),
            'max': float(np.max(scenarios)),
            'n_scenarios': self.n_scenarios
        }
        
        logger.info(f"Scenarios generated: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        
        return scenarios, stats
    
    def _generate_gaussian_scenarios(
        self,
        forecast_mean: np.ndarray,
        variance: float
    ) -> np.ndarray:
        """Generate scenarios using Gaussian noise"""
        std = np.sqrt(variance)
        scenarios = np.random.normal(
            loc=forecast_mean,
            scale=std,
            size=(self.n_scenarios, len(forecast_mean))
        )
        # Ensure non-negative demands
        scenarios = np.maximum(scenarios, 0)
        return scenarios
    
    def _generate_bootstrap_scenarios(
        self,
        forecast_mean: np.ndarray,
        variance: float
    ) -> np.ndarray:
        """Generate scenarios using bootstrap sampling"""
        std = np.sqrt(variance)
        # Generate residuals
        residuals = np.random.normal(0, std, size=(self.n_scenarios, len(forecast_mean)))
        scenarios = forecast_mean + residuals
        # Ensure non-negative demands
        scenarios = np.maximum(scenarios, 0)
        return scenarios
