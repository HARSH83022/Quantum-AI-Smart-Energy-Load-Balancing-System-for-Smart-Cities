"""
Model trainer with early stopping
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains LSTM model with early stopping"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        patience: int = 10
    ):
        """
        Initialize trainer
        
        Args:
            model: LSTM model to train
            learning_rate: Learning rate for optimizer
            patience: Patience for early stopping
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32
    ) -> dict:
        """
        Train model with early stopping
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Maximum epochs
            batch_size: Batch size
            
        Returns:
            Training history dict
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for i in range(0, len(X_train_t), batch_size):
                batch_X = X_train_t[i:i+batch_size]
                batch_y = y_train_t[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = self.criterion(val_outputs, y_val_t)
            
            avg_train_loss = np.mean(train_losses)
            val_loss_value = val_loss.item()
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss_value)
            
            # Early stopping
            if val_loss_value < best_val_loss:
                best_val_loss = val_loss_value
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss_value:.6f}")
            
            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        logger.info(f"Training completed. Best val loss: {best_val_loss:.6f}")
        
        return history
