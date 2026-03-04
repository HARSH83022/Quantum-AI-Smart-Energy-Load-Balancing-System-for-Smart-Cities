"""
Robust QUBO formulation with risk penalties
"""
import numpy as np
import logging
from typing import Tuple, Dict
from ..qubo.qubo_builder import QUBOBuilder

logger = logging.getLogger(__name__)


class RobustQUBOBuilder(QUBOBuilder):
    """Constructs robust QUBO matrix with scenario-based risk penalties"""
    
    def __init__(
        self,
        alpha: float = 10.0,
        beta: float = 5.0,
        gamma: float = 1.0,
        delta: float = 15.0,
        lambda1: float = 100.0,
        lambda2: float = 100.0,
        mode: str = "scenario-based"
    ):
        """
        Initialize robust QUBO builder
        
        Args:
            alpha: Overload penalty weight
            beta: Imbalance penalty weight
            gamma: Switching cost weight
            delta: Risk penalty weight
            lambda1: Capacity constraint penalty
            lambda2: Assignment constraint penalty
            mode: Optimization mode ('deterministic', 'scenario-based', 'worst-case', 'cvar')
        """
        super().__init__(alpha, beta, gamma, lambda1, lambda2)
        self.delta = delta
        self.mode = mode
        
    def build_robust_qubo(
        self,
        scenarios: np.ndarray,
        scenario_probs: np.ndarray,
        capacities: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Build robust QUBO matrix from scenarios
        
        Args:
            scenarios: Demand scenarios (n_scenarios, n_zones)
            scenario_probs: Scenario probabilities
            capacities: Transformer capacities
            
        Returns:
            Tuple of (robust QUBO matrix, metadata)
        """
        n_scenarios, n_zones = scenarios.shape
        n_transformers = len(capacities)
        
        logger.info(f"Building robust QUBO: mode={self.mode}, {n_scenarios} scenarios")
        
        if self.mode == "deterministic":
            # Use mean forecast only
            mean_demands = np.mean(scenarios, axis=0)
            Q, metadata = self.build_qubo_matrix(mean_demands, capacities)
            
        elif self.mode == "scenario-based":
            # Weighted combination across scenarios
            Q = None
            for i, (scenario, prob) in enumerate(zip(scenarios, scenario_probs)):
                Q_scenario, _ = self.build_qubo_matrix(scenario, capacities)
                if Q is None:
                    Q = prob * Q_scenario
                else:
                    Q += prob * Q_scenario
            
            # Add risk penalty
            Q = self._add_risk_penalty(Q, scenarios, scenario_probs, capacities, n_transformers, n_zones)
            
            metadata = {
                'n_transformers': n_transformers,
                'n_zones': n_zones,
                'n_variables': n_transformers * n_zones,
                'n_scenarios': n_scenarios,
                'mode': self.mode,
                'delta': self.delta,
                'is_symmetric': np.allclose(Q, Q.T)
            }
            
        elif self.mode == "worst-case":
            # Optimize for worst-case scenario
            worst_costs = []
            Q_matrices = []
            
            for scenario in scenarios:
                Q_scenario, _ = self.build_qubo_matrix(scenario, capacities)
                Q_matrices.append(Q_scenario)
                # Estimate worst-case cost (simplified)
                worst_costs.append(np.sum(np.abs(Q_scenario)))
            
            # Use worst-case scenario
            worst_idx = np.argmax(worst_costs)
            Q = Q_matrices[worst_idx]
            
            metadata = {
                'n_transformers': n_transformers,
                'n_zones': n_zones,
                'n_variables': n_transformers * n_zones,
                'mode': self.mode,
                'worst_scenario_idx': int(worst_idx),
                'is_symmetric': np.allclose(Q, Q.T)
            }
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        logger.info(f"Robust QUBO built: shape={Q.shape}, mode={self.mode}")
        
        return Q, metadata
    
    def _add_risk_penalty(
        self,
        Q: np.ndarray,
        scenarios: np.ndarray,
        scenario_probs: np.ndarray,
        capacities: np.ndarray,
        n_transformers: int,
        n_zones: int
    ) -> np.ndarray:
        """
        Add risk penalty term based on scenario probabilities
        
        Args:
            Q: Base QUBO matrix
            scenarios: Demand scenarios
            scenario_probs: Scenario probabilities
            capacities: Transformer capacities
            n_transformers: Number of transformers
            n_zones: Number of zones
            
        Returns:
            QUBO matrix with risk penalty
        """
        # Calculate expected overload across scenarios
        for s, (scenario, prob) in enumerate(zip(scenarios, scenario_probs)):
            for i in range(n_transformers):
                for j in range(n_zones):
                    idx = i * n_zones + j
                    
                    # Risk penalty for potential overload
                    potential_overload = max(0, scenario[j] - capacities[i])
                    risk_contrib = self.delta * prob * (potential_overload ** 2)
                    
                    Q[idx, idx] += risk_contrib
        
        return Q
    
    async def store_robust_qubo(self, Q: np.ndarray, metadata: Dict, scenario_id: int, session):
        """
        Store robust QUBO matrix to database
        
        Args:
            Q: Robust QUBO matrix
            metadata: Matrix metadata
            scenario_id: Reference to scenario
            session: Database session
        """
        from src.database.models import RobustQUBOMatrix
        
        logger.info("Storing robust QUBO matrix to database")
        
        record = RobustQUBOMatrix(
            scenario_id=scenario_id,
            optimization_mode=self.mode,
            matrix_data=Q.tolist(),
            risk_weight=self.delta,
            num_variables=metadata['n_variables']
        )
        
        session.add(record)
        await session.commit()
        await session.refresh(record)
        
        logger.info(f"Robust QUBO stored with ID: {record.id}")
        
        return record.id
