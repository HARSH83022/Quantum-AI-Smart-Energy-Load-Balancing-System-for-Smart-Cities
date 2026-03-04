"""
Quantum Fourier Transform analyzer using Qiskit
"""
import numpy as np
import logging
from typing import Tuple, Dict
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

logger = logging.getLogger(__name__)


class QFTAnalyzer:
    """Performs Quantum Fourier Transform analysis"""
    
    def __init__(self, n_qubits: int = 8):
        """
        Initialize QFT analyzer
        
        Args:
            n_qubits: Number of qubits for QFT circuit
        """
        self.n_qubits = n_qubits
        self.simulator = AerSimulator(method='statevector')
        
    def encode_amplitude_state(self, signal: np.ndarray) -> QuantumCircuit:
        """
        Encode normalized signal into quantum amplitude state
        
        Args:
            signal: Normalized signal values in [0, 1]
            
        Returns:
            Quantum circuit with encoded state
        """
        logger.info(f"Encoding signal of length {len(signal)} into {self.n_qubits} qubits")
        
        # Truncate or pad signal to match 2^n_qubits
        n_amplitudes = 2 ** self.n_qubits
        
        if len(signal) > n_amplitudes:
            signal = signal[:n_amplitudes]
        elif len(signal) < n_amplitudes:
            signal = np.pad(signal, (0, n_amplitudes - len(signal)), mode='constant')
        
        # Normalize to unit vector
        signal_norm = signal / np.linalg.norm(signal)
        
        # Create quantum circuit
        qr = QuantumRegister(self.n_qubits, 'q')
        qc = QuantumCircuit(qr)
        
        # Initialize state using amplitude encoding
        qc.initialize(signal_norm, qr)
        
        logger.info("Amplitude encoding completed")
        
        return qc
    
    def apply_qft(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Apply Quantum Fourier Transform to circuit
        
        Args:
            circuit: Input quantum circuit
            
        Returns:
            Circuit with QFT applied
        """
        logger.info(f"Applying QFT to {self.n_qubits}-qubit circuit")
        
        # Create QFT circuit
        qft = QFT(num_qubits=self.n_qubits, do_swaps=True)
        
        # Append QFT to circuit
        circuit.compose(qft, inplace=True)
        
        logger.info("QFT applied successfully")
        
        return circuit
    
    def extract_qft_spectrum(self, circuit: QuantumCircuit) -> Tuple[np.ndarray, Dict]:
        """
        Extract frequency spectrum from QFT circuit
        
        Args:
            circuit: Quantum circuit with QFT applied
            
        Returns:
            Tuple of (spectrum, metadata)
        """
        logger.info("Extracting QFT spectrum")
        
        # Get statevector
        statevector = Statevector(circuit)
        amplitudes = statevector.data
        
        # Compute magnitude spectrum
        spectrum = np.abs(amplitudes)
        
        # Verify state normalization
        norm = np.linalg.norm(spectrum)
        is_normalized = np.isclose(norm, 1.0, atol=1e-6)
        
        metadata = {
            'n_qubits': self.n_qubits,
            'spectrum_size': len(spectrum),
            'is_normalized': is_normalized,
            'norm': float(norm),
            'max_amplitude': float(spectrum.max()),
            'mean_amplitude': float(spectrum.mean())
        }
        
        logger.info(f"QFT spectrum extracted: norm={norm:.6f}, normalized={is_normalized}")
        
        return spectrum, metadata
    
    def analyze_signal(self, signal: np.ndarray) -> Dict:
        """
        Complete QFT analysis of signal
        
        Args:
            signal: Input signal (normalized to [0, 1])
            
        Returns:
            Dict with QFT analysis results
        """
        logger.info("Starting QFT analysis")
        
        # Encode signal
        circuit = self.encode_amplitude_state(signal)
        
        # Apply QFT
        circuit = self.apply_qft(circuit)
        
        # Extract spectrum
        spectrum, metadata = self.extract_qft_spectrum(circuit)
        
        # Find dominant frequencies
        n_amplitudes = len(spectrum)
        frequencies = np.arange(n_amplitudes) / n_amplitudes
        
        # Get top 5 peaks
        peak_indices = np.argsort(spectrum)[-5:][::-1]
        dominant_freqs = []
        
        for idx in peak_indices:
            dominant_freqs.append({
                'frequency': float(frequencies[idx]),
                'amplitude': float(spectrum[idx]),
                'index': int(idx)
            })
        
        results = {
            'spectrum': spectrum.tolist(),
            'dominant_frequencies': dominant_freqs,
            'metadata': metadata
        }
        
        logger.info("QFT analysis completed")
        
        return results
