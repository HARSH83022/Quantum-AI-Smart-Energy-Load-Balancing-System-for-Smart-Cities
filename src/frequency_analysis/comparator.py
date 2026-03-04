"""
Frequency analysis comparator for FFT vs QFT
"""
import numpy as np
import logging
from typing import Dict
from .classical_fft import ClassicalFFTAnalyzer
from .quantum_qft import QFTAnalyzer

logger = logging.getLogger(__name__)


class FrequencyComparator:
    """Compares FFT and QFT analysis results"""
    
    def __init__(self, sampling_rate: float = 4.0, n_qubits: int = 8):
        """
        Initialize comparator
        
        Args:
            sampling_rate: Sampling rate for FFT
            n_qubits: Number of qubits for QFT
        """
        self.fft_analyzer = ClassicalFFTAnalyzer(sampling_rate=sampling_rate)
        self.qft_analyzer = QFTAnalyzer(n_qubits=n_qubits)
        
    def compare_spectra(self, time_series: np.ndarray) -> Dict:
        """
        Compare FFT and QFT spectra
        
        Args:
            time_series: Input time series
            
        Returns:
            Dict with comparison results
        """
        logger.info("Comparing FFT and QFT spectra")
        
        # Normalize time series for QFT
        ts_normalized = (time_series - time_series.min()) / (time_series.max() - time_series.min())
        
        # FFT analysis
        fft_freqs, fft_mags = self.fft_analyzer.compute_fft_spectrum(time_series)
        fft_dominant = self.fft_analyzer.extract_dominant_frequencies(fft_freqs, fft_mags, top_k=5)
        fft_cycles = self.fft_analyzer.detect_cycles(time_series)
        fft_entropy = self.fft_analyzer.compute_spectral_entropy(fft_mags)
        
        # QFT analysis
        qft_results = self.qft_analyzer.analyze_signal(ts_normalized)
        
        # Compute correlation between spectra
        # Resample FFT spectrum to match QFT size
        qft_spectrum = np.array(qft_results['spectrum'])
        fft_spectrum_resampled = np.interp(
            np.linspace(0, 1, len(qft_spectrum)),
            np.linspace(0, 1, len(fft_mags)),
            fft_mags
        )
        
        # Normalize both for comparison
        fft_norm = fft_spectrum_resampled / np.linalg.norm(fft_spectrum_resampled)
        qft_norm = qft_spectrum / np.linalg.norm(qft_spectrum)
        
        # Compute correlation
        correlation = np.corrcoef(fft_norm, qft_norm)[0, 1]
        
        # Compute mean squared error
        mse = np.mean((fft_norm - qft_norm) ** 2)
        
        comparison = {
            'fft_analysis': {
                'dominant_frequencies': fft_dominant,
                'cycles': fft_cycles,
                'spectral_entropy': fft_entropy,
                'spectrum_size': len(fft_mags)
            },
            'qft_analysis': {
                'dominant_frequencies': qft_results['dominant_frequencies'],
                'metadata': qft_results['metadata'],
                'spectrum_size': len(qft_spectrum)
            },
            'comparison_metrics': {
                'correlation': float(correlation),
                'mse': float(mse),
                'similarity_score': float((1 + correlation) / 2)  # Normalize to [0, 1]
            }
        }
        
        logger.info(f"Comparison completed: correlation={correlation:.4f}, mse={mse:.6f}")
        
        return comparison
    
    async def store_frequency_features(self, comparison: Dict, data_id: int, session):
        """
        Store frequency features to database
        
        Args:
            comparison: Comparison results
            data_id: Reference to raw data
            session: Database session
        """
        from src.database.models import FrequencyFeature
        
        logger.info("Storing frequency features to database")
        
        # Store FFT features
        fft_record = FrequencyFeature(
            data_id=data_id,
            method='fft',
            dominant_frequencies=comparison['fft_analysis']['dominant_frequencies'],
            cycle_strengths=comparison['fft_analysis']['cycles'],
            spectral_entropy=comparison['fft_analysis']['spectral_entropy']
        )
        
        # Store QFT features
        qft_record = FrequencyFeature(
            data_id=data_id,
            method='qft',
            dominant_frequencies=comparison['qft_analysis']['dominant_frequencies'],
            cycle_strengths=comparison['qft_analysis']['metadata'],
            spectral_entropy=None  # QFT doesn't compute entropy directly
        )
        
        session.add_all([fft_record, qft_record])
        await session.commit()
        
        logger.info("Frequency features stored successfully")
