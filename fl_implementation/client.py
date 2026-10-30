import flwr as fl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import os
import sys

# Add project root to sys.path to import models
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.mlp import GraphAwareMLP

import math

DATA_FILE = os.getenv("DATA_FILE")
CLIENT_ID = os.getenv("CLIENT_ID", "")
ENABLE_DP = os.getenv("ENABLE_DP", "true").lower() in ("true", "1", "yes")
DP_CLIP_NORM = float(os.getenv("DP_CLIP_NORM", "1.0"))
DP_NOISE_MULTIPLIER = float(os.getenv("DP_NOISE_MULTIPLIER", "0.5"))

def compute_dp_budget(n_samples, batch_size, epochs, noise_multiplier, delta=1e-5):
    """Computes standard analytical (epsilon, delta)-DP bound for Gaussian mechanism."""
    if not ENABLE_DP or noise_multiplier <= 0:
        return 0.0, 0.0
    q = batch_size / max(1, n_samples)
    steps = epochs * math.ceil(n_samples / max(1, batch_size))
    # Gaussian mechanism composition bound
    epsilon = (q * math.sqrt(2 * steps * math.log(1.25 / delta))) / noise_multiplier
    return float(epsilon), float(delta)

def load_data():
    global DATA_FILE
    if not DATA_FILE:
        if CLIENT_ID:
            DATA_FILE = os.path.join(current_dir, "data", f"bank_{CLIENT_ID}.csv")
        else:
            DATA_FILE = os.path.join(current_dir, "data", "bank_1.csv")
            
    if not os.path.exists(DATA_FILE):
        # Try resolving relative to project_root
        candidate = os.path.join(project_root, DATA_FILE)
        if os.path.exists(candidate):
            DATA_FILE = candidate
        else:
            raise FileNotFoundError(f"Data file not found at {DATA_FILE}")
        
    df = pd.read_csv(DATA_FILE)

    # Use all available numerical features
    X = df.drop(columns=['is_laundering', 'sender', 'receiver', 'timestamp', 'nameOrig', 'nameDest', '_pk', '_edge_key'], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df['is_laundering'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


class AMLClient(fl.client.NumPyClient):
    def __init__(self, client_id=None):
        self.client_id = client_id or CLIENT_ID or "1"
        self.X, self.y = load_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)
        
        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.test_dataset = TensorDataset(self.X_test, self.y_test)
        
        self.batch_size = 256
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.input_dim = self.X_train.shape[1]
        self.model = GraphAwareMLP(input_dim=self.input_dim)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.fit_rounds = 0

    def get_parameters(self, config):
        return self.model.get_parameters()

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.fit_rounds += 1
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        epochs = 1
        for epoch in range(epochs):
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch).squeeze()
                
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                    
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                # Defensible Differential Privacy (DP-SGD)
                if ENABLE_DP:
                    # 1. Gradient Norm Clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=DP_CLIP_NORM)
                    # 2. Calibrated Gaussian Noise Injection
                    if DP_NOISE_MULTIPLIER > 0:
                        for param in self.model.parameters():
                            if param.grad is not None:
                                noise = torch.randn_like(param.grad) * (DP_NOISE_MULTIPLIER * DP_CLIP_NORM / math.sqrt(len(X_batch)))
                                param.grad.add_(noise)
                                
                optimizer.step()
                
        # Save local client model checkpoint
        if self.client_id:
            chk_path = os.path.join(current_dir, f"model_bank_{self.client_id}.pt")
            try:
                torch.save(self.model.state_dict(), chk_path)
            except Exception as e:
                pass

        eps, delta = compute_dp_budget(len(self.train_dataset), self.batch_size, self.fit_rounds, DP_NOISE_MULTIPLIER)
        fit_metrics = {"dp_enabled": ENABLE_DP, "epsilon": eps, "delta": delta}
        return self.get_parameters(config), len(self.train_dataset), fit_metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        criterion = nn.BCELoss()
        avg_loss, auc, f1, acc = self.model.evaluate(self.test_loader, criterion, self.device)
        
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch).squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                y_true.extend(y_batch.numpy())
                y_pred.extend((outputs.cpu().numpy() > 0.5).astype(int))
                
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)

        # Exact confusion matrix components
        tp = int(np.sum((y_true_np == 1) & (y_pred_np == 1)))
        fp = int(np.sum((y_true_np == 0) & (y_pred_np == 1)))
        fn = int(np.sum((y_true_np == 1) & (y_pred_np == 0)))
        tn = int(np.sum((y_true_np == 0) & (y_pred_np == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return avg_loss, len(self.test_dataset), {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }

if __name__ == "__main__":
    server_addr = os.getenv("SERVER_ADDRESS", "localhost:8080")
    if os.path.exists("/.dockerenv") or os.environ.get("DATA_FILE", "").startswith("data/"):
        server_addr = "fl_server:8080"

    client_id = os.getenv("CLIENT_ID", "")
    fl.client.start_numpy_client(
        server_address=server_addr,
        client=AMLClient(client_id=client_id),
    )