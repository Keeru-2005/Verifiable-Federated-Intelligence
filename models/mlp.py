import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


class ResidualMLPBlock(nn.Module):
    """Pre-normalized residual block for stable tabular representation learning."""

    def __init__(self, width=64, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x):
        update = self.norm(x)
        update = self.fc1(update)
        update = self.activation(update)
        update = self.dropout(update)
        update = self.fc2(update)
        return x + update


class GraphAwareMLP(nn.Module):
    def __init__(self, input_dim=32, dropout=0.1, return_logits=False):
        super(GraphAwareMLP, self).__init__()
        self.return_logits = return_logits

        # The preprocessed inputs are standardized continuous features. A
        # compact residual representation keeps gradients useful across local
        # client updates, while LayerNorm remains compatible with per-example
        # DP-SGD (unlike batch-dependent normalization).
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.residual_block1 = ResidualMLPBlock(width=64, dropout=dropout)
        self.residual_block2 = ResidualMLPBlock(width=64, dropout=dropout)
        self.representation = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.GELU(),
        )

        # Keep the final Linear(32,1) parameter shape and state-dict key stable:
        # the existing Groth16 circuit commits this selected aggregation head.
        self.output_layer = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward_logits(self, x):
        x = self.input_projection(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.representation(x)
        return self.output_layer[0](x)

    def forward(self, x):
        logits = self.forward_logits(x)
        return logits if self.return_logits else torch.sigmoid(logits)

    def get_parameters(self):
        """Extracts parameters as a list of independent numpy arrays for Federated Learning Aggregation."""
        return [val.cpu().detach().numpy().copy() for _, val in self.state_dict().items()]

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
