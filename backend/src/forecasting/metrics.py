"""
Metrics calculator for forecast evaluation
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates forecast accuracy metrics"""
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAE value
        """
        mae = np.mean(np.abs(y_true - y_pred))
        logger.info(f"MAE: {mae:.6f}")
        return float(mae)
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            RMSE value
        """
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        logger.info(f"RMSE: {rmse:.6f}")
        return float(rmse)
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAPE value (percentage)
        """
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        logger.info(f"MAPE: {mape:.2f}%")
        return float(mape)
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calculate all metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dict with all metrics
        """
        return {
            'mae': MetricsCalculator.calculate_mae(y_true, y_pred),
            'rmse': MetricsCalculator.calculate_rmse(y_true, y_pred),
            'mape': MetricsCalculator.calculate_mape(y_true, y_pred)
        }
