"""
Quantum Load Balancer — Orchestrator Module.

Ties together the QUBO formulation and QAOA solver to perform
end-to-end quantum-optimized load distribution across Delhi's
power grid transformers.

Workflow:
    1. Accept predicted peak load from ML model
    2. Formulate the QUBO matrix (load balancing problem)
    3. Solve via QAOA on Qiskit simulator
    4. Decode solution into per-transformer allocations
    5. Compute balance metrics and return structured results
"""

import numpy as np
import time
import traceback

from src.optimization.qubo_formulation import QUBOFormulator
from src.quantum.qaoa_solver import QAOASolver
from src.utils.helpers import (
    compute_balance_score,
    get_transformer_status,
    get_transformer_config,
)


class QuantumLoadBalancer:
    """
    Orchestrates quantum-optimized load distribution across grid transformers.

    Uses QUBO formulation + QAOA to find the optimal load assignment
    that minimizes overload risk and maximizes distribution balance.
    """

    def __init__(self, reps=2, shots=4096, block_size_mw=200.0):
        """
        Parameters:
            reps          (int):   QAOA circuit depth.
            shots         (int):   Quantum measurement shots.
            block_size_mw (float): Load discretization block size (MW).
        """
        self.reps = reps
        self.shots = shots
        self.block_size = block_size_mw

        self.formulator = QUBOFormulator(block_size_mw=block_size_mw)
        self.solver = QAOASolver(reps=reps, shots=shots)

    def optimize(self, predicted_load_mw):
        """
        Run quantum optimization for the given predicted load.

        Parameters:
            predicted_load_mw (float): Predicted peak load in MW.

        Returns:
            dict: {
                "transformers":         list[dict] — per-transformer results,
                "total_assigned_mw":    float,
                "predicted_load_mw":    float,
                "balance_score":        float (0–1),
                "quantum_energy":       float — QUBO objective value,
                "optimization_method":  str,
                "n_qubits":             int,
                "execution_time_sec":   float,
                "status":               str — "success" or "error"
            }
        """
        start_time = time.time()

        try:
            # ── Step 1: Formulate QUBO ──
            qubo_result = self.formulator.formulate(predicted_load_mw)
            Q = qubo_result["Q"]

            print(f"\n{'='*50}")
            print(f"[QUANTUM] LOAD OPTIMIZATION")
            print(f"{'='*50}")
            print(f"Predicted Load:  {predicted_load_mw:.1f} MW")
            print(f"Transformers:    {qubo_result['n_transformers']}")
            print(f"Load Blocks:     {qubo_result['n_blocks']} per transformer")
            print(f"Block Size:      {qubo_result['block_size_mw']} MW")
            print(f"Total Qubits:    {qubo_result['n_qubits']}")
            print(f"QAOA Layers:     {self.reps}")
            print(f"Shots:           {self.shots}")
            print(f"{'='*50}")

            # ── Step 2: Solve with QAOA ──
            print("\n[*] Running QAOA optimization...")
            qaoa_result = self.solver.solve(Q)

            optimal_bitstring = qaoa_result["optimal_bitstring"]
            print(f"[OK] Optimal bitstring: {optimal_bitstring}")
            print(f"   Quantum energy:    {qaoa_result['optimal_cost']}")

            # ── Step 3: Decode Solution ──
            allocations = self.formulator.decode_solution(
                optimal_bitstring, qubo_result
            )

            # ── Step 4: Compute Metrics ──
            assigned_loads = [a["assigned_load_mw"] for a in allocations]
            capacities = [a["capacity_mw"] for a in allocations]
            total_assigned = sum(assigned_loads)

            balance_score = compute_balance_score(assigned_loads, capacities)

            # Add status to each transformer
            for alloc in allocations:
                alloc["status"] = get_transformer_status(alloc["utilization_pct"])

            elapsed = round(time.time() - start_time, 3)

            # ── Print Results ──
            print(f"\n{'-'*50}")
            print(f"[RESULTS] OPTIMIZATION RESULTS")
            print(f"{'-'*50}")
            for a in allocations:
                filled = int(a["utilization_pct"] / 5)
                bar = "#" * filled + "." * (20 - filled)
                print(f"  {a['name']:18s} | {a['assigned_load_mw']:7.1f} MW | "
                      f"{bar} {a['utilization_pct']:5.1f}%")
            print(f"{'-'*50}")
            print(f"  Total Assigned: {total_assigned:.1f} MW "
                  f"(target: {predicted_load_mw:.1f} MW)")
            print(f"  Balance Score:  {balance_score:.4f}")
            print(f"  Execution Time: {elapsed}s")
            print(f"{'='*50}\n")

            return {
                "transformers": allocations,
                "total_assigned_mw": round(total_assigned, 2),
                "predicted_load_mw": round(predicted_load_mw, 2),
                "balance_score": balance_score,
                "quantum_energy": qaoa_result["optimal_cost"],
                "optimization_method": f"QAOA (p={self.reps}, {self.shots} shots)",
                "n_qubits": qaoa_result["n_qubits"],
                "execution_time_sec": elapsed,
                "status": "success",
            }

        except Exception as e:
            elapsed = round(time.time() - start_time, 3)
            print(f"\n[ERROR] Quantum optimization failed: {e}")
            traceback.print_exc()

            # ── Fallback: Classical proportional distribution ──
            return self._classical_fallback(predicted_load_mw, elapsed, str(e))

    def _classical_fallback(self, predicted_load_mw, elapsed, error_msg):
        """
        Fallback to classical proportional load distribution
        if quantum optimization fails.

        Distributes load proportionally based on transformer capacity.
        """
        print("[WARN] Using classical fallback distribution...")

        transformers = get_transformer_config()
        total_capacity = sum(t["capacity_mw"] for t in transformers)

        allocations = []
        for t in transformers:
            proportion = t["capacity_mw"] / total_capacity
            assigned = predicted_load_mw * proportion
            utilization = (assigned / t["capacity_mw"]) * 100.0

            allocations.append({
                "name": t["name"],
                "assigned_load_mw": round(assigned, 2),
                "capacity_mw": t["capacity_mw"],
                "utilization_pct": round(utilization, 2),
                "assigned_blocks": int(assigned / self.block_size),
                "status": get_transformer_status(utilization),
            })

        assigned_loads = [a["assigned_load_mw"] for a in allocations]
        capacities = [a["capacity_mw"] for a in allocations]
        balance_score = compute_balance_score(assigned_loads, capacities)

        return {
            "transformers": allocations,
            "total_assigned_mw": round(sum(assigned_loads), 2),
            "predicted_load_mw": round(predicted_load_mw, 2),
            "balance_score": balance_score,
            "quantum_energy": None,
            "optimization_method": f"Classical Fallback (QAOA failed: {error_msg})",
            "n_qubits": 0,
            "execution_time_sec": elapsed,
            "status": "fallback",
        }


# ─────────────────────────────────────────────
# Standalone Test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    balancer = QuantumLoadBalancer(reps=2, shots=2048, block_size_mw=200.0)
    result = balancer.optimize(predicted_load_mw=5500.0)

    print("\n🔑 Final Result Keys:", list(result.keys()))
    print(f"📈 Balance Score: {result['balance_score']}")
    print(f"⚡ Method: {result['optimization_method']}")
