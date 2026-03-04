"""
QUBO formulation for load balancing optimization
"""
import numpy as np
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class QUBOBuilder:
    """Constructs QUBO matrix for load balancing problem"""
    
    def __init__(
        self,
        alpha: float = 10.0,
        beta: float = 5.0,
        gamma: float = 1.0,
        lambda1: float = 100.0,
        lambda2: float = 100.0
    ):
        """
        Initialize QUBO builder
        
        Args:
            alpha: Overload penalty weight
            beta: Imbalance penalty weight
            gamma: Switching cost weight
            lambda1: Capacity constraint penalty
            lambda2: Assignment constraint penalty
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
    def build_qubo_matrix(
        self,
        demands: np.ndarray,
        capacities: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Build QUBO matrix from demands and capacities
        
        Args:
            demands: Zone demands (n_zones,)
            capacities: Transformer capacities (n_transformers,)
            
        Returns:
            Tuple of (QUBO matrix, metadata)
        """
        n_zones = len(demands)
        n_transformers = len(capacities)
        n_vars = n_transformers * n_zones
        
        logger.info(f"Building QUBO matrix: {n_transformers} transformers, {n_zones} zones, {n_vars} variables")
        
        # Initialize QUBO matrix
        Q = np.zeros((n_vars, n_vars))
        
        # Add objective terms
        Q = self._add_overload_penalty(Q, demands, capacities, n_transformers, n_zones)
        Q = self._add_imbalance_penalty(Q, demands, n_transformers, n_zones)
        Q = self._add_switching_cost(Q, n_transformers, n_zones)
        
        # Add constraint penalties
        Q = self._add_capacity_constraints(Q, demands, capacities, n_transformers, n_zones)
        Q = self._add_assignment_constraints(Q, n_transformers, n_zones)
        
        # Ensure symmetry
        Q = (Q + Q.T) / 2
        
        metadata = {
            'n_transformers': n_transformers,
            'n_zones': n_zones,
            'n_variables': n_vars,
            'is_symmetric': np.allclose(Q, Q.T),
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'lambda1': self.lambda1,
            'lambda2': self.lambda2
        }
        
        logger.info(f"QUBO matrix built: shape={Q.shape}, symmetric={metadata['is_symmetric']}")
        
        return Q, metadata
    
    def _add_overload_penalty(
        self,
        Q: np.ndarray,
        demands: np.ndarray,
        capacities: np.ndarray,
        n_transformers: int,
        n_zones: int
    ) -> np.ndarray:
        """Add overload penalty term"""
        for i in range(n_transformers):
            for j in range(n_zones):
                idx = i * n_zones + j
                # Quadratic term for overload
                overload_contrib = self.alpha * (demands[j] ** 2)
                Q[idx, idx] += overload_contrib
        
        return Q
    
    def _add_imbalance_penalty(
        self,
        Q: np.ndarray,
        demands: np.ndarray,
        n_transformers: int,
        n_zones: int
    ) -> np.ndarray:
        """Add load imbalance penalty term"""
        avg_load = np.sum(demands) / n_transformers
        
        for i in range(n_transformers):
            for j in range(n_zones):
                idx = i * n_zones + j
                # Penalty for deviation from average
                imbalance_contrib = self.beta * ((demands[j] - avg_load) ** 2)
                Q[idx, idx] += imbalance_contrib
        
        return Q
    
    def _add_switching_cost(
        self,
        Q: np.ndarray,
        n_transformers: int,
        n_zones: int
    ) -> np.ndarray:
        """Add switching cost term"""
        for i in range(n_transformers):
            for j in range(n_zones):
                idx = i * n_zones + j
                # Simple switching cost
                Q[idx, idx] += self.gamma
        
        return Q
    
    def _add_capacity_constraints(
        self,
        Q: np.ndarray,
        demands: np.ndarray,
        capacities: np.ndarray,
        n_transformers: int,
        n_zones: int
    ) -> np.ndarray:
        """Add capacity constraint penalties"""
        for i in range(n_transformers):
            for j1 in range(n_zones):
                for j2 in range(n_zones):
                    idx1 = i * n_zones + j1
                    idx2 = i * n_zones + j2
                    
                    # Penalty for exceeding capacity
                    penalty = self.lambda1 * demands[j1] * demands[j2]
                    
                    if idx1 == idx2:
                        Q[idx1, idx1] += penalty
                    else:
                        Q[idx1, idx2] += penalty / 2
        
        return Q
    
    def _add_assignment_constraints(
        self,
        Q: np.ndarray,
        n_transformers: int,
        n_zones: int
    ) -> np.ndarray:
        """Add assignment constraint penalties (each zone to exactly one transformer)"""
        for j in range(n_zones):
            for i1 in range(n_transformers):
                for i2 in range(n_transformers):
                    idx1 = i1 * n_zones + j
                    idx2 = i2 * n_zones + j
                    
                    if i1 == i2:
                        # Penalty for not assigning
                        Q[idx1, idx1] += self.lambda2
                    else:
                        # Penalty for double assignment
                        Q[idx1, idx2] += 2 * self.lambda2
        
        return Q
    
    async def store_qubo_matrix(self, Q: np.ndarray, metadata: Dict, forecast_id: int, session):
        """
        Store QUBO matrix to database
        
        Args:
            Q: QUBO matrix
            metadata: Matrix metadata
            forecast_id: Reference to forecast
            session: Database session
        """
        from src.database.models import QUBOMatrix
        
        logger.info("Storing QUBO matrix to database")
        
        record = QUBOMatrix(
            forecast_id=forecast_id,
            matrix_data=Q.tolist(),
            num_variables=metadata['n_variables']
        )
        
        session.add(record)
        await session.commit()
        await session.refresh(record)
        
        logger.info(f"QUBO matrix stored with ID: {record.id}")
        
        return record.id
