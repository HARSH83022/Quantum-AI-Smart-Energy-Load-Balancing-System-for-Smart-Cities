"""
Forecaster for generating predictions
"""
import torch
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class Forecaster:
    """Generates demand forecasts"""
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize forecaster
        
        Args:
            model: Trained LSTM model
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def forecast(
        self,
        input_sequence: np.ndarray,
        horizon_steps: int = 2
    ) -> np.ndarray:
        """
        Generate forecast for next 30 minutes (2 steps at 15-min intervals)
        
        Args:
            input_sequence: Input sequence (sequence_length, n_features)
            horizon_steps: Number of steps to forecast
            
        Returns:
            Forecasted values
        """
        logger.info(f"Generating forecast for {horizon_steps} steps")
        
        with torch.no_grad():
            # Add batch dimension
            if input_sequence.ndim == 2:
                input_sequence = input_sequence[np.newaxis, :]
            
            X = torch.FloatTensor(input_sequence).to(self.device)
            predictions = []
            
            for _ in range(horizon_steps):
                pred = self.model(X)
                predictions.append(pred.cpu().numpy())
                
                # For multi-step, would update X with prediction
                # For now, just predict once
                break
            
            predictions = np.array(predictions).flatten()
            
        logger.info(f"Forecast generated: {predictions}")
        return predictions
    
    def batch_forecast(self, sequences: np.ndarray) -> np.ndarray:
        """
        Generate forecasts for batch of sequences
        
        Args:
            sequences: Batch of sequences (batch_size, sequence_length, n_features)
            
        Returns:
            Batch of predictions
        """
        with torch.no_grad():
            X = torch.FloatTensor(sequences).to(self.device)
            predictions = self.model(X)
            return predictions.cpu().numpy().flatten()
