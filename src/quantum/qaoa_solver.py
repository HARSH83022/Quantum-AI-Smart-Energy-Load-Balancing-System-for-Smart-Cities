"""
QAOA (Quantum Approximate Optimization Algorithm) Solver
for Smart Grid Load Optimization.

This module implements a QAOA-based solver that takes a QUBO matrix
and finds the optimal binary assignment using quantum circuits
simulated via Qiskit.

QAOA Circuit Structure:
───────────────────────
    |0⟩ ─── H ─── e^{-iγC} ─── e^{-iβB} ─── ... ─── Measure
    |0⟩ ─── H ─── e^{-iγC} ─── e^{-iβB} ─── ... ─── Measure
    ...

Where:
    C = Cost Hamiltonian (derived from QUBO)
    B = Mixer Hamiltonian (standard X-mixer)
    γ, β = Variational parameters optimized classically

The solver uses Qiskit's AerSimulator for statevector simulation
with a classical COBYLA optimizer for the variational loop.
"""

import numpy as np
from scipy.optimize import minimize

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    from qiskit_aer import AerSimulator
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False


class QAOASolver:
    """
    Solves QUBO problems using the Quantum Approximate Optimization Algorithm.

    Uses Qiskit's quantum circuit simulation to find the binary variable
    assignment that minimizes the QUBO objective function.

    Attributes:
        reps  (int):   Number of QAOA layers (circuit depth parameter p).
        shots (int):   Number of measurement shots for sampling.
        method (str):  Classical optimizer method (default: COBYLA).
    """

    def __init__(self, reps=2, shots=4096, method="COBYLA"):
        """
        Parameters:
            reps   (int): QAOA depth parameter p. Higher → better solutions,
                          slower execution. Recommended: 2–3 for simulators.
            shots  (int): Number of quantum measurement shots.
            method (str): Scipy optimizer method for variational parameters.
        """
        self.reps = reps
        self.shots = shots
        self.method = method

        if not QISKIT_AVAILABLE:
            raise ImportError(
                "Qiskit is required. Install with: pip install qiskit"
            )

    def _qubo_to_ising(self, Q):
        """
        Convert a QUBO matrix to an Ising Hamiltonian (SparsePauliOp).

        The transformation uses: x_i = (1 - Z_i) / 2

        Parameters:
            Q (np.ndarray): Upper-triangular QUBO matrix (n × n).

        Returns:
            SparsePauliOp: Ising Hamiltonian operator.
        """
        n = Q.shape[0]
        pauli_list = []

        offset = 0.0

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    # Diagonal: Q_ii * x_i = Q_ii * (1 - Z_i) / 2
                    coeff = Q[i, i] / 2.0
                    offset += Q[i, i] / 2.0

                    if abs(coeff) > 1e-10:
                        # Z_i term (negative because (1-Z)/2)
                        z_label = ["I"] * n
                        z_label[n - 1 - i] = "Z"
                        pauli_list.append(("".join(z_label), -coeff))
                else:
                    # Off-diagonal: Q_ij * x_i * x_j
                    # = Q_ij/4 * (1 - Z_i)(1 - Z_j)
                    # = Q_ij/4 * (1 - Z_i - Z_j + Z_i*Z_j)
                    coeff = Q[i, j] / 4.0
                    offset += Q[i, j] / 4.0

                    if abs(coeff) > 1e-10:
                        # -Z_i term
                        zi_label = ["I"] * n
                        zi_label[n - 1 - i] = "Z"
                        pauli_list.append(("".join(zi_label), -coeff))

                        # -Z_j term
                        zj_label = ["I"] * n
                        zj_label[n - 1 - j] = "Z"
                        pauli_list.append(("".join(zj_label), -coeff))

                        # +Z_i Z_j term
                        zz_label = ["I"] * n
                        zz_label[n - 1 - i] = "Z"
                        zz_label[n - 1 - j] = "Z"
                        pauli_list.append(("".join(zz_label), coeff))

        # Add constant offset as identity
        if abs(offset) > 1e-10:
            pauli_list.append(("I" * n, offset))

        # Combine duplicate Pauli terms
        hamiltonian = SparsePauliOp.from_list(pauli_list).simplify()

        return hamiltonian

    def _build_qaoa_circuit(self, hamiltonian, params):
        """
        Build a QAOA circuit with the given variational parameters.

        Parameters:
            hamiltonian (SparsePauliOp): Cost Hamiltonian.
            params      (np.ndarray):   Variational parameters [γ_1,...,γ_p, β_1,...,β_p].

        Returns:
            QuantumCircuit: Parameterized QAOA circuit.
        """
        n_qubits = hamiltonian.num_qubits
        gamma_params = params[:self.reps]
        beta_params = params[self.reps:]

        qc = QuantumCircuit(n_qubits)

        # Initial superposition
        qc.h(range(n_qubits))

        for layer in range(self.reps):
            gamma = gamma_params[layer]
            beta = beta_params[layer]

            # ── Cost Unitary: e^{-iγC} ──
            # Apply ZZ and Z rotations based on Hamiltonian
            for term, coeff in zip(
                hamiltonian.paulis.to_labels(),
                hamiltonian.coeffs
            ):
                if term == "I" * n_qubits:
                    continue  # skip identity (global phase)

                coeff_val = float(np.real(coeff))
                z_indices = [
                    n_qubits - 1 - k
                    for k, p in enumerate(term)
                    if p == "Z"
                ]

                if len(z_indices) == 1:
                    qc.rz(2 * gamma * coeff_val, z_indices[0])
                elif len(z_indices) == 2:
                    i, j = z_indices
                    qc.cx(i, j)
                    qc.rz(2 * gamma * coeff_val, j)
                    qc.cx(i, j)

            # ── Mixer Unitary: e^{-iβB} ──
            for qubit in range(n_qubits):
                qc.rx(2 * beta, qubit)

        # Measurement
        qc.measure_all()

        return qc

    def _evaluate_cost(self, bitstring, Q):
        """
        Evaluate the QUBO cost function for a given bitstring.

        Parameters:
            bitstring (str): Binary string (e.g., "101010").
            Q (np.ndarray):  QUBO matrix.

        Returns:
            float: Cost value x^T Q x.
        """
        x = np.array([int(b) for b in bitstring], dtype=float)
        return float(x @ Q @ x)

    def solve(self, Q):
        """
        Solve the QUBO problem using QAOA.

        Parameters:
            Q (np.ndarray): Upper-triangular QUBO matrix.

        Returns:
            dict: {
                "optimal_bitstring": str    — best solution found,
                "optimal_cost":     float   — QUBO objective value,
                "counts":           dict    — measurement count distribution,
                "optimal_params":   list    — optimized [γ, β] parameters,
                "n_qubits":         int     — problem size,
                "reps":             int     — QAOA depth,
                "shots":            int     — measurement shots used,
                "method":           str     — "QAOA"
            }
        """
        n_qubits = Q.shape[0]

        # Convert QUBO to Ising Hamiltonian
        hamiltonian = self._qubo_to_ising(Q)

        # Setup simulator backend
        if AER_AVAILABLE:
            # Use qasm_simulator — memory efficient shot-based sampling
            backend = AerSimulator(method="automatic")
        else:
            # Fallback: use Qiskit's built-in simulator
            from qiskit.providers.basic_provider import BasicSimulator
            backend = BasicSimulator()

        # ── Classical optimization loop ──
        # Optimize γ and β parameters to minimize ⟨ψ(γ,β)|C|ψ(γ,β)⟩
        best_result = {
            "optimal_bitstring": "0" * n_qubits,
            "optimal_cost": float("inf"),
            "counts": {},
        }

        def objective(params):
            """Objective function for the classical optimizer."""
            qc = self._build_qaoa_circuit(hamiltonian, params)

            # Run on simulator (no transpile needed for ideal sim)
            job = backend.run(qc, shots=self.shots)
            result = job.result()
            counts = result.get_counts()

            # Compute expectation value of cost
            total_cost = 0.0
            total_shots = sum(counts.values())

            for bitstring, count in counts.items():
                # Remove spaces if present in bitstring
                clean_bits = bitstring.replace(" ", "")
                cost = self._evaluate_cost(clean_bits, Q)
                total_cost += cost * count

                # Track best solution
                if cost < best_result["optimal_cost"]:
                    best_result["optimal_bitstring"] = clean_bits
                    best_result["optimal_cost"] = cost
                    best_result["counts"] = counts

            expectation = total_cost / total_shots
            return expectation

        # Initial parameters: random starting point
        n_params = 2 * self.reps
        initial_params = np.random.uniform(0, np.pi, n_params)

        # Run optimization
        opt_result = minimize(
            objective,
            initial_params,
            method=self.method,
            options={"maxiter": 100, "disp": False},
        )

        # Final evaluation with optimal parameters to get best counts
        final_qc = self._build_qaoa_circuit(hamiltonian, opt_result.x)

        job = backend.run(final_qc, shots=self.shots)
        result = job.result()
        final_counts = result.get_counts()

        # Find the best bitstring from final run
        for bitstring, count in final_counts.items():
            clean_bits = bitstring.replace(" ", "")
            cost = self._evaluate_cost(clean_bits, Q)
            if cost < best_result["optimal_cost"]:
                best_result["optimal_bitstring"] = clean_bits
                best_result["optimal_cost"] = cost

        best_result["counts"] = final_counts

        return {
            "optimal_bitstring": best_result["optimal_bitstring"],
            "optimal_cost": round(best_result["optimal_cost"], 4),
            "counts": best_result["counts"],
            "optimal_params": opt_result.x.tolist(),
            "n_qubits": n_qubits,
            "reps": self.reps,
            "shots": self.shots,
            "method": "QAOA",
        }
