"""
Classical FFT analyzer for time series periodicity detection
"""
import numpy as np
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class ClassicalFFTAnalyzer:
    """Performs classical Fast Fourier Transform analysis"""
    
    def __init__(self, sampling_rate: float = 4.0):
        """
        Initialize FFT analyzer
        
        Args:
            sampling_rate: Samples per hour (default: 4 for 15-minute intervals)
        """
        self.sampling_rate = sampling_rate
        
    def compute_fft_spectrum(self, time_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT spectrum of time series
        
        Args:
            time_series: 1D array of time series data
            
        Returns:
            Tuple of (frequencies, magnitudes)
        """
        logger.info(f"Computing FFT spectrum for {len(time_series)} data points")
        
        # Compute FFT
        fft_values = np.fft.fft(time_series)
        
        # Compute frequencies
        n = len(time_series)
        frequencies = np.fft.fftfreq(n, d=1/self.sampling_rate)
        
        # Compute magnitudes (only positive frequencies)
        magnitudes = np.abs(fft_values)
        
        # Keep only positive frequencies
        positive_freq_idx = frequencies > 0
        frequencies = frequencies[positive_freq_idx]
        magnitudes = magnitudes[positive_freq_idx]
        
        logger.info(f"FFT spectrum computed: {len(frequencies)} frequency components")
        
        return frequencies, magnitudes
    
    def extract_dominant_frequencies(
        self,
        frequencies: np.ndarray,
        magnitudes: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, float]]:
        """
        Extract dominant frequencies from spectrum
        
        Args:
            frequencies: Frequency values
            magnitudes: Magnitude values
            top_k: Number of top frequencies to extract
            
        Returns:
            List of dicts with frequency and magnitude
        """
        # Find peaks
        peak_indices = np.argsort(magnitudes)[-top_k:][::-1]
        
        dominant_freqs = []
        for idx in peak_indices:
            freq = frequencies[idx]
            mag = magnitudes[idx]
            period_hours = 1.0 / freq if freq > 0 else np.inf
            
            dominant_freqs.append({
                'frequency_hz': float(freq),
                'magnitude': float(mag),
                'period_hours': float(period_hours)
            })
        
        logger.info(f"Extracted {len(dominant_freqs)} dominant frequencies")
        
        return dominant_freqs
    
    def detect_cycles(
        self,
        time_series: np.ndarray,
        daily_threshold: float = 0.1,
        weekly_threshold: float = 0.05
    ) -> Dict[str, float]:
        """
        Detect daily and weekly cycles
        
        Args:
            time_series: Time series data
            daily_threshold: Threshold for daily cycle detection
            weekly_threshold: Threshold for weekly cycle detection
            
        Returns:
            Dict with cycle strengths
        """
        frequencies, magnitudes = self.compute_fft_spectrum(time_series)
        
        # Normalize magnitudes
        magnitudes_norm = magnitudes / magnitudes.max()
        
        # Daily cycle: ~24 hours = 1/24 Hz
        daily_freq = 1.0 / 24.0
        daily_idx = np.argmin(np.abs(frequencies - daily_freq))
        daily_strength = magnitudes_norm[daily_idx]
        
        # Weekly cycle: ~168 hours = 1/168 Hz
        weekly_freq = 1.0 / 168.0
        weekly_idx = np.argmin(np.abs(frequencies - weekly_freq))
        weekly_strength = magnitudes_norm[weekly_idx]
        
        cycles = {
            'daily_cycle_strength': float(daily_strength),
            'weekly_cycle_strength': float(weekly_strength),
            'has_daily_cycle': daily_strength > daily_threshold,
            'has_weekly_cycle': weekly_strength > weekly_threshold
        }
        
        logger.info(f"Cycle detection: daily={daily_strength:.4f}, weekly={weekly_strength:.4f}")
        
        return cycles
    
    def compute_spectral_entropy(self, magnitudes: np.ndarray) -> float:
        """
        Compute spectral entropy as measure of signal complexity
        
        Args:
            magnitudes: FFT magnitude spectrum
            
        Returns:
            Spectral entropy value
        """
        # Normalize to probability distribution
        power = magnitudes ** 2
        power_norm = power / power.sum()
        
        # Compute entropy
        entropy = -np.sum(power_norm * np.log2(power_norm + 1e-10))
        
        return float(entropy)
