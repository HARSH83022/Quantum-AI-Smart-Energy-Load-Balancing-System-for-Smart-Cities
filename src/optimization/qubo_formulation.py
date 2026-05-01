"""
QUBO (Quadratic Unconstrained Binary Optimization) Formulation
for Smart Grid Transformer Load Balancing.

This module converts the load distribution problem into a QUBO matrix
that can be solved by the QAOA quantum optimizer.

Mathematical Formulation:
─────────────────────────
Given:
    N transformers with capacities C_1, ..., C_N
    Total predicted load L (MW)
    K discrete load blocks of size w each

Binary variables:
    x_{i,k} = 1 if load block k is assigned to transformer i

Objective (QUBO):
    min  α · Σ_i (Σ_k w·x_{i,k} - C_i)²     [capacity penalty]
       + β · (Σ_{i,k} w·x_{i,k} - L)²        [load conservation]
       + γ · Σ_{i≠j} (util_i - util_j)²       [balance penalty]

The QUBO matrix Q is constructed such that:
    f(x) = x^T Q x
"""

import numpy as np
from src.utils.helpers import get_transformer_config


class QUBOFormulator:
    """
    Constructs QUBO matrices for the transformer load balancing problem.

    The predicted load is discretized into binary load blocks,
    and the QUBO encodes capacity constraints + balance objectives.
    """

    def __init__(self, block_size_mw=200.0, alpha=1.0, beta=2.0, gamma=0.5):
        """
        Parameters:
            block_size_mw (float): Size of each discrete load block in MW.
                                   Smaller = more precise but more qubits.
            alpha (float): Weight for capacity penalty term.
            beta  (float): Weight for load conservation constraint.
            gamma (float): Weight for load balance penalty.
        """
        self.block_size = block_size_mw
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.transformers = get_transformer_config()
        self.n_transformers = len(self.transformers)

    def formulate(self, predicted_load_mw):
        """
        Build the QUBO matrix for the given predicted load.

        Parameters:
            predicted_load_mw (float): Total predicted peak load in MW.

        Returns:
            dict: {
                "Q":              np.ndarray — QUBO matrix (n_qubits × n_qubits),
                "n_qubits":       int — total number of binary variables,
                "n_transformers": int — number of transformers,
                "n_blocks":       int — load blocks per transformer,
                "block_size_mw":  float — MW per block,
                "predicted_load": float — target load,
                "capacities":     list — per-transformer capacity
            }
        """
        # Number of load blocks per transformer
        # Each transformer gets up to ceil(capacity / block_size) blocks
        max_blocks = max(
            int(np.ceil(t["capacity_mw"] / self.block_size))
            for t in self.transformers
        )

        # For a tractable problem on a simulator, limit blocks
        # to keep qubit count manageable (N * K qubits total)
        # 5 transformers × 4 blocks = 20 qubits (fits in ~16MB RAM)
        n_blocks = min(max_blocks, 4)

        n_qubits = self.n_transformers * n_blocks
        Q = np.zeros((n_qubits, n_qubits))

        capacities = [t["capacity_mw"] for t in self.transformers]
        target_blocks = predicted_load_mw / self.block_size

        # ─────────────────────────────────────────
        # Term 1: Capacity Penalty
        # Penalize: (Σ_k w·x_{i,k} - C_i)²
        # ─────────────────────────────────────────
        for i in range(self.n_transformers):
            cap_blocks = capacities[i] / self.block_size

            for k1 in range(n_blocks):
                idx1 = i * n_blocks + k1

                # Linear term: -2 * cap_blocks * w  (on diagonal)
                Q[idx1, idx1] += self.alpha * (1.0 - 2.0 * cap_blocks)

                # Quadratic terms: w² * x_{i,k1} * x_{i,k2}
                for k2 in range(k1 + 1, n_blocks):
                    idx2 = i * n_blocks + k2
                    Q[idx1, idx2] += self.alpha * 2.0

        # ─────────────────────────────────────────
        # Term 2: Load Conservation Constraint
        # Penalize: (Σ_{i,k} x_{i,k} - target_blocks)²
        # ─────────────────────────────────────────
        for idx1 in range(n_qubits):
            # Linear: (1 - 2*target_blocks)
            Q[idx1, idx1] += self.beta * (1.0 - 2.0 * target_blocks)

            # Quadratic: 2 * x_a * x_b for all pairs
            for idx2 in range(idx1 + 1, n_qubits):
                Q[idx1, idx2] += self.beta * 2.0

        # ─────────────────────────────────────────
        # Term 3: Balance Penalty
        # Encourage even utilization across transformers
        # Penalize: Σ_{i<j} (util_i - util_j)²
        # ─────────────────────────────────────────
        for i in range(self.n_transformers):
            for j in range(i + 1, self.n_transformers):
                ci = capacities[i]
                cj = capacities[j]

                for k1 in range(n_blocks):
                    idx_i = i * n_blocks + k1

                    # Cross terms between transformer i and j
                    for k2 in range(n_blocks):
                        idx_j = j * n_blocks + k2

                        # (x_i/C_i - x_j/C_j)² expansion
                        coeff = self.gamma * 2.0 * (
                            -self.block_size / (ci * cj)
                        )

                        if idx_i < idx_j:
                            Q[idx_i, idx_j] += coeff
                        else:
                            Q[idx_j, idx_i] += coeff

                    # Diagonal terms from balance
                    Q[idx_i, idx_i] += self.gamma * (
                        self.block_size / ci
                    ) ** 2

        # Make upper triangular (QUBO convention)
        for i in range(n_qubits):
            for j in range(i):
                Q[i, j] = 0.0

        return {
            "Q": Q,
            "n_qubits": n_qubits,
            "n_transformers": self.n_transformers,
            "n_blocks": n_blocks,
            "block_size_mw": self.block_size,
            "predicted_load": predicted_load_mw,
            "capacities": capacities,
        }

    def decode_solution(self, bitstring, metadata):
        """
        Decode a QUBO solution bitstring into transformer load assignments.

        Parameters:
            bitstring (str or list): Binary solution (e.g., "10110...")
            metadata  (dict): Metadata from formulate()

        Returns:
            list[dict]: Per-transformer allocation details.
        """
        n_blocks = metadata["n_blocks"]
        block_size = metadata["block_size_mw"]
        capacities = metadata["capacities"]

        bits = [int(b) for b in bitstring]

        allocations = []
        for i in range(self.n_transformers):
            start = i * n_blocks
            end = start + n_blocks
            assigned_blocks = sum(bits[start:end])
            assigned_mw = assigned_blocks * block_size

            utilization = (assigned_mw / capacities[i]) * 100.0

            allocations.append({
                "name": self.transformers[i]["name"],
                "assigned_load_mw": round(assigned_mw, 2),
                "capacity_mw": capacities[i],
                "utilization_pct": round(utilization, 2),
                "assigned_blocks": assigned_blocks,
            })

        return allocations
