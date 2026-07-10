import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

class GraphAwareMLP(nn.Module):
    def __init__(self, input_dim=32):
        super(GraphAwareMLP, self).__init__()
        
        # Ensures fixed random state for reproducibility per architecture rule
        torch.manual_seed(42)
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output_layer(x)
        return x

    def get_parameters(self):
        """Extracts parameters as a list of numpy arrays for Federated Learning Aggregation."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_parameters(self, parameters):
        """Loads parameters from a list of numpy arrays received from FL Server."""
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.load_state_dict(state_dict, strict=True)
        
    def evaluate(self, dataloader, criterion, device="cpu"):
        """Evaluates model and returns statistical metrics required by spec."""
        self.eval()
        total_loss = 0.0
        y_true = []
        y_pred_probs = []
        y_pred_classes = []
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                outputs = self(X_batch).squeeze()
                
                # Handle single element batch dimension collapsing edge case
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                
                loss = criterion(outputs, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                
                y_true.extend(y_batch.cpu().numpy())
                y_pred_probs.extend(outputs.cpu().numpy())
                y_pred_classes.extend((outputs.cpu().numpy() > 0.5).astype(int))
                
        avg_loss = total_loss / max(1, len(y_true))
        
        # Calculate evaluation metrics robustly
        try:
            auc = roc_auc_score(y_true, y_pred_probs)
        except ValueError:
            auc = 0.5 # Default if only one class exists in the evaluation batch subset
            
        f1 = f1_score(y_true, y_pred_classes, zero_division=0)
        acc = accuracy_score(y_true, y_pred_classes)
        
        return avg_loss, auc, f1, acc
