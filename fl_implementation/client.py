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

DATA_FILE = os.getenv("DATA_FILE")

def load_data():
    if not DATA_FILE or not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found at {DATA_FILE}")
        
    df = pd.read_csv(DATA_FILE)

    # Use all available numerical features
    X = df.drop(columns=['is_laundering', 'sender', 'receiver', 'timestamp', 'nameOrig', 'nameDest', '_pk', '_edge_key'], errors='ignore')
    
    # Filter out non-numeric columns just in case
    X = X.select_dtypes(include=[np.number])
    
    y = df['is_laundering'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


class AMLClient(fl.client.NumPyClient):
    def __init__(self):
        self.X, self.y = load_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        # Convert to PyTorch tensors
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)
        
        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.test_dataset = TensorDataset(self.X_test, self.y_test)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=256, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=256, shuffle=False)
        
        self.input_dim = self.X_train.shape[1]
        self.model = GraphAwareMLP(input_dim=self.input_dim)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config):
        return self.model.get_parameters()

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
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
                optimizer.step()
                
        return self.get_parameters(config), len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        criterion = nn.BCELoss()
        avg_loss, auc, f1, acc = self.model.evaluate(self.test_loader, criterion, self.device)
        
        # We need precision and recall for the server logging.
        # So we can manually compute it here from self.model predictions.
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
                
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        return avg_loss, len(self.test_dataset), {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        }

if __name__ == "__main__":
    server_addr = os.getenv("SERVER_ADDRESS", "localhost:8080")
    if os.path.exists("/.dockerenv") or os.environ.get("DATA_FILE", "").startswith("data/"):
        server_addr = "fl_server:8080" # Make sure this matches docker-compose hostname!

    fl.client.start_numpy_client(
        server_address=server_addr,
        client=AMLClient(),
    )