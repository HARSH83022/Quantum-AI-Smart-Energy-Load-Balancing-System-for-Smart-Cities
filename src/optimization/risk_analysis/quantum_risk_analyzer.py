"""
Quantum risk analyzer for solution stability and CVaR optimization
"""
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class CVaRCalculator:
    """Computes Conditional Value at Risk (CVaR)"""
    
    def __init__(self, confidence: float = 0.95):
        """
        Initialize CVaR calculator
        
        Args:
            confidence: Confidence level (default: 0.95 for 95%)
        """
        self.confidence = confidence
        self.alpha = 1 - confidence  # Tail probability
    
    def calculate_var(self, costs: np.ndarray) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            costs: Array of cost values
            
        Returns:
            VaR at specified confidence level
        """
        var = np.percentile(costs, self.confidence * 100)
        logger.debug(f"VaR at {self.confidence*100}%: {var:.4f}")
        return float(var)
    
    def calculate_cvar(self, costs: np.ndarray) -> float:
        """
        Calculate Conditional Value at Risk (CVaR)
        
        CVaR is the expected cost in the worst (1-confidence)% of scenarios
        
        Args:
            costs: Array of cost values
            
        Returns:
            CVaR at specified confidence level
        """
        var = self.calculate_var(costs)
        
        # CVaR is the mean of costs exceeding VaR
        tail_costs = costs[costs >= var]
        
        if len(tail_costs) == 0:
            cvar = var
        else:
            cvar = np.mean(tail_costs)
        
        logger.debug(f"CVaR at {self.confidence*100}%: {cvar:.4f}")
        return float(cvar)
    
    def verify_cvar_monotonicity(self, costs: np.ndarray) -> bool:
        """
        Verify that CVaR >= VaR (monotonicity property)
        
        Args:
            costs: Array of cost values
            
        Returns:
            True if CVaR >= VaR
        """
        var = self.calculate_var(costs)
        cvar = self.calculate_cvar(costs)
        
        is_monotonic = cvar >= var
        
        if not is_monotonic:
            logger.warning(f"CVaR monotonicity violated: CVaR={cvar:.4f} < VaR={var:.4f}")
        
        return is_monotonic


class QuantumRiskAnalyzer:
    """Evaluates solution stability and optimizes for risk-adjusted objectives"""
    
    def __init__(self, confidence: float = 0.95, lambda_risk: float = 1.0):
        """
        Initialize quantum risk analyzer
        
        Args:
            confidence: Confidence level for CVaR (default: 0.95)
            lambda_risk: Risk aversion parameter (default: 1.0)
        """
        self.confidence = confidence
        self.lambda_risk = lambda_risk
        self.cvar_calculator = CVaRCalculator(confidence)
    
    def evaluate_solution_stability(
        self,
        solution: np.ndarray,
        scenarios: np.ndarray,
        capacities: np.ndarray
    ) -> Dict:
        """
        Evaluate solution robustness across scenarios
        
        Args:
            solution: Binary solution vector
            scenarios: Demand scenarios (n_scenarios, n_zones)
            capacities: Transformer capacities
            
        Returns:
            Dict with stability metrics
        """
        n_scenarios = scenarios.shape[0]
        logger.info(f"Evaluating solution stability across {n_scenarios} scenarios")
        
        # Calculate costs for each scenario
        costs = self._calculate_scenario_costs(solution, scenarios, capacities)
        
        # Calculate risk metrics
        expected_cost = self.calculate_expected_cost(solution, scenarios, capacities)
        cost_variance = self.calculate_cost_variance(solution, scenarios, capacities)
        cost_std = np.sqrt(cost_variance)
        var = self.cvar_calculator.calculate_var(costs)
        cvar = self.cvar_calculator.calculate_cvar(costs)
        
        # Calculate Sharpe ratio (assuming baseline cost of 0)
        sharpe_ratio = -expected_cost / cost_std if cost_std > 0 else 0.0
        
        stability_metrics = {
            'expected_cost': float(expected_cost),
            'cost_variance': float(cost_variance),
            'cost_std': float(cost_std),
            'var_95': float(var),
            'cvar_95': float(cvar),
            'sharpe_ratio': float(sharpe_ratio),
            'n_scenarios': n_scenarios,
            'confidence_level': self.confidence
        }
        
        logger.info(f"Stability analysis: E[cost]={expected_cost:.4f}, CVaR={cvar:.4f}")
        
        return stability_metrics
    
    def calculate_expected_cost(
        self,
        solution: np.ndarray,
        scenarios: np.ndarray,
        capacities: np.ndarray
    ) -> float:
        """
        Calculate expected cost across scenarios
        
        Args:
            solution: Binary solution vector
            scenarios: Demand scenarios
            capacities: Transformer capacities
            
        Returns:
            Expected cost
        """
        costs = self._calculate_scenario_costs(solution, scenarios, capacities)
        expected_cost = np.mean(costs)
        return float(expected_cost)
    
    def calculate_cost_variance(
        self,
        solution: np.ndarray,
        scenarios: np.ndarray,
        capacities: np.ndarray
    ) -> float:
        """
        Calculate variance of cost across scenarios
        
        Args:
            solution: Binary solution vector
            scenarios: Demand scenarios
            capacities: Transformer capacities
            
        Returns:
            Cost variance
        """
        costs = self._calculate_scenario_costs(solution, scenarios, capacities)
        variance = np.var(costs)
        return float(variance)
    
    def optimize_cvar_objective(
        self,
        scenarios: np.ndarray,
        capacities: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Optimize risk-adjusted objective: E[cost] + λ·CVaR[cost]
        
        Args:
            scenarios: Demand scenarios
            capacities: Transformer capacities
            
        Returns:
            Tuple of (optimal solution, metrics)
        """
        n_scenarios, n_zones = scenarios.shape
        n_transformers = len(capacities)
        n_vars = n_transformers * n_zones
        
        logger.info(f"Optimizing CVaR objective with λ={self.lambda_risk}")
        
        # Simplified: Use greedy heuristic for demonstration
        # In practice, this would use QAOA with CVaR-modified objective
        best_solution = None
        best_objective = float('inf')
        
        # Try random solutions and pick best
        for _ in range(100):
            # Generate random feasible solution
            solution = self._generate_random_solution(n_transformers, n_zones)
            
            # Calculate objective
            expected_cost = self.calculate_expected_cost(solution, scenarios, capacities)
            costs = self._calculate_scenario_costs(solution, scenarios, capacities)
            cvar = self.cvar_calculator.calculate_cvar(costs)
            
            objective = expected_cost + self.lambda_risk * cvar
            
            if objective < best_objective:
                best_objective = objective
                best_solution = solution
        
        # Calculate final metrics
        metrics = self.evaluate_solution_stability(best_solution, scenarios, capacities)
        metrics['risk_adjusted_objective'] = float(best_objective)
        metrics['lambda_risk'] = self.lambda_risk
        
        logger.info(f"CVaR optimization complete: objective={best_objective:.4f}")
        
        return best_solution, metrics
    
    def _calculate_scenario_costs(
        self,
        solution: np.ndarray,
        scenarios: np.ndarray,
        capacities: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cost for each scenario
        
        Args:
            solution: Binary solution vector
            scenarios: Demand scenarios
            capacities: Transformer capacities
            
        Returns:
            Array of costs for each scenario
        """
        n_scenarios, n_zones = scenarios.shape
        n_transformers = len(capacities)
        
        costs = np.zeros(n_scenarios)
        
        for s, scenario in enumerate(scenarios):
            # Reshape solution to matrix form
            assignment = solution.reshape(n_transformers, n_zones)
            
            # Calculate load on each transformer
            transformer_loads = assignment @ scenario
            
            # Calculate overload penalty
            overloads = np.maximum(0, transformer_loads - capacities)
            overload_cost = np.sum(overloads ** 2)
            
            # Calculate imbalance penalty
            avg_load = np.mean(transformer_loads)
            imbalance_cost = np.sum((transformer_loads - avg_load) ** 2)
            
            # Total cost
            costs[s] = overload_cost + imbalance_cost
        
        return costs
    
    def _generate_random_solution(self, n_transformers: int, n_zones: int) -> np.ndarray:
        """Generate random feasible solution (each zone to one transformer)"""
        solution = np.zeros(n_transformers * n_zones)
        
        for j in range(n_zones):
            # Assign zone j to random transformer
            i = np.random.randint(n_transformers)
            idx = i * n_zones + j
            solution[idx] = 1
        
        return solution
    
    async def store_risk_metrics(self, metrics: Dict, scenario_id: int, session):
        """
        Store risk metrics to database
        
        Args:
            metrics: Risk analysis metrics
            scenario_id: Reference to scenario
            session: Database session
        """
        from src.database.models import RiskMetric
        
        logger.info("Storing risk metrics to database")
        
        record = RiskMetric(
            scenario_id=scenario_id,
            expected_cost=metrics['expected_cost'],
            cost_variance=metrics['cost_variance'],
            var_95=metrics['var_95'],
            cvar_95=metrics['cvar_95'],
            overload_frequencies={},  # Placeholder
            stress_probabilities={}   # Placeholder
        )
        
        session.add(record)
        await session.commit()
        await session.refresh(record)
        
        logger.info(f"Risk metrics stored with ID: {record.id}")
        
        return record.id
