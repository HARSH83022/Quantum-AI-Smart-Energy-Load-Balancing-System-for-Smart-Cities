"""
LSTM model for demand forecasting with frequency features
"""
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """Enhanced LSTM model with frequency features"""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list = [128, 64, 32],
        output_size: int = 1,
        dropout: float = 0.2
    ):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output values
            dropout: Dropout rate
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_sizes[2], output_size)
        
        logger.info(f"LSTM model initialized: input={input_size}, hidden={hidden_sizes}, output={output_size}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor (batch_size, output_size)
        """
        # LSTM layer 1
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        # LSTM layer 2
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        
        # LSTM layer 3
        out, _ = self.lstm3(out)
        
        # Take last time step
        out = out[:, -1, :]
        
        # Output layer
        out = self.fc(out)
        
        return out
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
