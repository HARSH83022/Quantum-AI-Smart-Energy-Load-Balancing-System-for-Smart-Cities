"""
Enhanced QAOA optimizer for QUBO problems with parameter warm-starting and convergence monitoring
"""
import numpy as np
import logging
import time
import os
from typing import Dict, List, Tuple, Optional
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class ParameterWarmStarter:
    """Handles parameter warm-starting from previous solutions"""
    
    def __init__(self):
        self.previous_params = None
        self.previous_problem_size = None
    
    def warm_start_parameters(self, new_problem_size: int, p: int) -> Optional[np.ndarray]:
        """
        Initialize parameters from previous solutions
        
        Args:
            new_problem_size: Size of new problem
            p: Number of QAOA layers
            
        Returns:
            Warm-started parameters or None
        """
        if self.previous_params is None:
            return None
        
        # If problem size matches, use previous params directly
        if self.previous_problem_size == new_problem_size:
            logger.info("Using exact warm-start parameters")
            return self.previous_params.copy()
        
        # Otherwise, interpolate or use subset
        logger.info("Adapting parameters for new problem size")
        return self.previous_params[:2*p] if len(self.previous_params) >= 2*p else None
    
    def store_parameters(self, params: np.ndarray, problem_size: int):
        """Store parameters for future warm-starting"""
        self.previous_params = params.copy()
        self.previous_problem_size = problem_size


class ConvergenceMonitor:
    """Tracks optimization progress and convergence"""
    
    def __init__(self, threshold: float = 1e-4):
        self.threshold = threshold
        self.energy_history: List[float] = []
        self.variance_history: List[float] = []
    
    def add_energy(self, energy: float):
        """Add energy value to history"""
        self.energy_history.append(energy)
    
    def monitor_convergence(self) -> bool:
        """
        Check if optimization has converged
        
        Returns:
            True if converged
        """
        if len(self.energy_history) < 5:
            return False
        
        # Check if energy change is below threshold
        recent_energies = self.energy_history[-5:]
        energy_change = max(recent_energies) - min(recent_energies)
        
        converged = energy_change < self.threshold
        if converged:
            logger.info(f"Convergence detected: energy change = {energy_change:.6f}")
        
        return converged
    
    def calculate_expectation_variance(self, measurements: Dict[str, int]) -> float:
        """
        Calculate variance of expectation value from measurements
        
        Args:
            measurements: Dict of bitstring -> count
            
        Returns:
            Variance of measurements
        """
        total_shots = sum(measurements.values())
        probabilities = {k: v/total_shots for k, v in measurements.items()}
        
        # Calculate variance
        mean_prob = np.mean(list(probabilities.values()))
        variance = np.mean([(p - mean_prob)**2 for p in probabilities.values()])
        
        self.variance_history.append(variance)
        return variance


class QAOACircuitBuilder:
    """Constructs parameterized QAOA quantum circuits"""
    
    def __init__(self, p: int = 3):
        """
        Initialize circuit builder
        
        Args:
            p: Number of QAOA layers
        """
        self.p = p
    
    def build_qaoa_circuit(self, Q: np.ndarray) -> Tuple[QuantumCircuit, List[Parameter]]:
        """
        Build QAOA circuit for QUBO problem
        
        Args:
            Q: QUBO matrix
            
        Returns:
            Tuple of (circuit, parameters)
        """
        n_qubits = Q.shape[0]
        
        # Create circuit
        qc = QuantumCircuit(n_qubits)
        
        # Initialize in superposition
        qc.h(range(n_qubits))
        
        # Create parameters
        gamma_params = [Parameter(f'γ_{i}') for i in range(self.p)]
        beta_params = [Parameter(f'β_{i}') for i in range(self.p)]
        
        # Add QAOA layers
        for layer in range(self.p):
            # Cost Hamiltonian (problem-dependent)
            self._add_cost_layer(qc, Q, gamma_params[layer])
            
            # Mixer Hamiltonian
            self._add_mixer_layer(qc, beta_params[layer])
        
        # Measurement
        qc.measure_all()
        
        all_params = gamma_params + beta_params
        
        logger.info(f"Built QAOA circuit: {n_qubits} qubits, {self.p} layers, depth={qc.depth()}")
        
        return qc, all_params
    
    def _add_cost_layer(self, qc: QuantumCircuit, Q: np.ndarray, gamma: Parameter):
        """Add cost Hamiltonian layer"""
        n_qubits = Q.shape[0]
        
        # Diagonal terms
        for i in range(n_qubits):
            if Q[i, i] != 0:
                qc.rz(2 * gamma * Q[i, i], i)
        
        # Off-diagonal terms (interactions)
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                if Q[i, j] != 0:
                    qc.cx(i, j)
                    qc.rz(2 * gamma * Q[i, j], j)
                    qc.cx(i, j)
    
    def _add_mixer_layer(self, qc: QuantumCircuit, beta: Parameter):
        """Add mixer Hamiltonian layer"""
        n_qubits = qc.num_qubits
        for i in range(n_qubits):
            qc.rx(2 * beta, i)


class QAOAOptimizer:
    """Enhanced QAOA optimizer with warm-starting and convergence monitoring"""
    
    def __init__(self, p: int = 3, max_iter: int = 1000):
        """
        Initialize QAOA optimizer
        
        Args:
            p: Number of QAOA layers (default: 3)
            max_iter: Maximum optimization iterations
        """
        self.p = p
        self.max_iter = max_iter
        self.backend = self._get_backend()
        self.warm_starter = ParameterWarmStarter()
        self.convergence_monitor = ConvergenceMonitor()
        self.circuit_builder = QAOACircuitBuilder(p)
        
    def _get_backend(self):
        """Get quantum backend (IBM or Aer simulator)"""
        ibm_key = os.getenv("IBM_QUANTUM_API_KEY")
        
        if ibm_key:
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
                service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_key)
                backend = service.least_busy(operational=True, simulator=False)
                logger.info(f"Using IBM Quantum backend: {backend.name}")
                return backend
            except Exception as e:
                logger.warning(f"Failed to connect to IBM Quantum: {e}. Using Aer simulator.")
        
        backend = AerSimulator()
        logger.info("Using Qiskit Aer simulator")
        return backend
    
    def optimize(self, Q: np.ndarray) -> Dict:
        """
        Optimize QUBO using enhanced QAOA
        
        Args:
            Q: QUBO matrix
            
        Returns:
            Dict with solution and performance metrics
        """
        start_time = time.time()
        n_qubits = Q.shape[0]
        
        logger.info(f"Running enhanced QAOA: {n_qubits} qubits, p={self.p}, max_iter={self.max_iter}")
        
        # Build QAOA circuit
        circuit, params = self.circuit_builder.build_qaoa_circuit(Q)
        
        # Initialize parameters (warm start if available)
        initial_params = self.warm_starter.warm_start_parameters(n_qubits, self.p)
        if initial_params is None:
            initial_params = np.random.uniform(0, 2*np.pi, 2 * self.p)
            logger.info("Using random initial parameters")
        else:
            logger.info("Using warm-started parameters")
        
        # Reset convergence monitor
        self.convergence_monitor = ConvergenceMonitor()
        
        # Optimize
        iteration_count = [0]
        
        def cost_function(param_values):
            iteration_count[0] += 1
            energy = self._evaluate_energy(circuit, params, param_values, Q)
            self.convergence_monitor.add_energy(energy)
            
            if iteration_count[0] % 10 == 0:
                logger.debug(f"Iteration {iteration_count[0]}: energy = {energy:.6f}")
            
            return energy
        
        result = minimize(
            fun=cost_function,
            x0=initial_params,
            method='COBYLA',
            options={'maxiter': self.max_iter}
        )
        
        # Store parameters for warm starting
        self.warm_starter.store_parameters(result.x, n_qubits)
        
        # Get solution
        optimal_params = result.x
        solution_bitstring, measurements = self._sample_solution(circuit, params, optimal_params)
        solution_vector = np.array([int(b) for b in solution_bitstring])
        
        # Calculate metrics
        execution_time = time.time() - start_time
        expectation_variance = self.convergence_monitor.calculate_expectation_variance(measurements)
        convergence_rate = len(self.convergence_monitor.energy_history) / self.max_iter
        
        metrics = {
            'solution': solution_vector.tolist(),
            'objective_value': float(result.fun),
            'execution_time_seconds': execution_time,
            'n_iterations': result.nit,
            'success': result.success,
            'circuit_depth': circuit.depth(),
            'gate_count': sum(circuit.count_ops().values()),
            'n_qubits': n_qubits,
            'p_layers': self.p,
            'convergence_rate': convergence_rate,
            'energy_variance': expectation_variance,
            'backend_used': str(self.backend)
        }
        
        logger.info(f"QAOA completed: objective={result.fun:.4f}, time={execution_time:.2f}s, iterations={result.nit}")
        
        return metrics
    
    def _evaluate_energy(self, circuit: QuantumCircuit, params: List[Parameter], param_values: np.ndarray, Q: np.ndarray) -> float:
        """
        Evaluate energy expectation value
        
        Args:
            circuit: QAOA circuit
            params: Circuit parameters
            param_values: Parameter values
            Q: QUBO matrix
            
        Returns:
            Energy expectation value
        """
        # Bind parameters
        param_dict = {params[i]: param_values[i] for i in range(len(params))}
        bound_circuit = circuit.bind_parameters(param_dict)
        
        # Execute circuit
        job = self.backend.run(bound_circuit, shots=1024)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate expectation value
        energy = 0.0
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            # Convert bitstring to solution vector
            x = np.array([int(b) for b in bitstring[::-1]])  # Reverse for Qiskit convention
            
            # Calculate QUBO energy
            qubo_energy = x.T @ Q @ x
            
            # Weight by probability
            probability = count / total_shots
            energy += probability * qubo_energy
        
        return energy
    
    def _sample_solution(self, circuit: QuantumCircuit, params: List[Parameter], param_values: np.ndarray) -> Tuple[str, Dict]:
        """
        Sample solution from QAOA circuit
        
        Args:
            circuit: QAOA circuit
            params: Circuit parameters
            param_values: Parameter values
            
        Returns:
            Tuple of (best bitstring, all measurements)
        """
        # Bind parameters
        param_dict = {params[i]: param_values[i] for i in range(len(params))}
        bound_circuit = circuit.bind_parameters(param_dict)
        
        # Execute circuit
        job = self.backend.run(bound_circuit, shots=1024)
        result = job.result()
        counts = result.get_counts()
        
        # Get most frequent bitstring
        best_bitstring = max(counts, key=counts.get)
        
        # Reverse for Qiskit convention
        best_bitstring = best_bitstring[::-1]
        
        return best_bitstring, counts
    
    async def store_result(self, metrics: Dict, qubo_id: int, session):
        """
        Store optimization result to database
        
        Args:
            metrics: Optimization metrics
            qubo_id: Reference to QUBO matrix
            session: Database session
        """
        from src.database.models import OptimizationResult
        
        logger.info("Storing optimization result to database")
        
        record = OptimizationResult(
            qubo_id=qubo_id,
            solution_vector=metrics['solution'],
            objective_value=metrics['objective_value'],
            execution_time_seconds=metrics['execution_time_seconds'],
            backend_used=metrics['backend_used'],
            circuit_depth=metrics['circuit_depth']
        )
        
        session.add(record)
        await session.commit()
        await session.refresh(record)
        
        logger.info(f"Optimization result stored with ID: {record.id}")
        
        return record.id
