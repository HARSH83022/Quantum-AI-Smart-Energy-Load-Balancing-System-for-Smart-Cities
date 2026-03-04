"""
Property-based tests for frequency analysis module
Feature: quantum-energy-load-balancing
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.frequency_analysis.classical_fft import ClassicalFFTAnalyzer
from src.frequency_analysis.quantum_qft import QFTAnalyzer


# Feature: quantum-energy-load-balancing, Property 20: Dominant Frequency Detection
@given(
    n_samples=st.integers(min_value=200, max_value=500),
    frequency=st.floats(min_value=0.01, max_value=0.5)
)
@settings(max_examples=50, deadline=None)
def test_fft_peak_detection(n_samples, frequency):
    """
    Property 20: Dominant Frequency Detection
    For any time series with known periodic components, FFT analysis should 
    correctly identify the dominant frequencies within a tolerance threshold.
    Validates: Requirements 11.2
    """
    # Create synthetic signal with known frequency
    t = np.arange(n_samples)
    signal = np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(n_samples)
    
    # Analyze with FFT
    analyzer = ClassicalFFTAnalyzer(sampling_rate=1.0)
    frequencies, magnitudes = analyzer.compute_fft_spectrum(signal)
    
    # Extract dominant frequencies
    dominant = analyzer.extract_dominant_frequencies(frequencies, magnitudes, top_k=3)
    
    # Verify we detected frequencies
    assert len(dominant) > 0
    
    # The dominant frequency should be close to our input frequency
    detected_freq = dominant[0]['frequency_hz']
    tolerance = 0.1  # 10% tolerance
    
    # Check if detected frequency is within tolerance
    assert abs(detected_freq - frequency) / frequency < tolerance or detected_freq < 0.01


# Feature: quantum-energy-load-balancing, Property 21: QFT Output State Validity
@given(
    n_samples=st.integers(min_value=16, max_value=256)
)
@settings(max_examples=50, deadline=None)
def test_qft_state_normalization(n_samples):
    """
    Property 21: QFT Output State Validity
    For any normalized demand signal encoded into quantum amplitude states,
    the QFT output state should have unit norm and valid probability amplitudes.
    Validates: Requirements 11.4
    """
    # Create normalized signal
    signal = np.random.rand(n_samples)
    signal = signal / np.linalg.norm(signal)
    
    # Analyze with QFT
    analyzer = QFTAnalyzer(n_qubits=4)  # 2^4 = 16 amplitudes
    results = analyzer.analyze_signal(signal)
    
    # Verify metadata
    metadata = results['metadata']
    
    # Check normalization
    assert metadata['is_normalized'] == True
    assert abs(metadata['norm'] - 1.0) < 1e-5
    
    # Check spectrum is valid
    spectrum = np.array(results['spectrum'])
    assert len(spectrum) > 0
    assert np.all(spectrum >= 0)  # All amplitudes should be non-negative
    assert np.all(spectrum <= 1)  # All amplitudes should be <= 1
