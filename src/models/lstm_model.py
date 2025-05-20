from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def create_lstm_model(params: Optional[Dict] = None) -> Tuple[LSTMModel, Dict]:
    """Create LSTM model with default or custom parameters."""
    default_params = {
        'input_dim': 10,  # Number of features
        'hidden_dim': 64,
        'num_layers': 2,
        'output_dim': 1,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'sequence_length': 10
    }
    
    if params:
        default_params.update(params)
    
    model = LSTMModel(
        input_dim=default_params['input_dim'],
        hidden_dim=default_params['hidden_dim'],
        num_layers=default_params['num_layers'],
        output_dim=default_params['output_dim'],
        dropout=default_params['dropout']
    )
    
    return model, default_params

def prepare_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare sequences for LSTM training."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, -1])  # Assuming last column is target
    return np.array(X), np.array(y)

def create_data_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    """Create PyTorch DataLoader for training."""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
