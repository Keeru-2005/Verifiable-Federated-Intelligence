import flwr as fl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import json
import math
import os
import sys
import random

# Fixed seed for reproducible client training. Each client process seeds the
# same generators; the data partition still differs per client.
SEED = int(os.getenv("VFI_SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Add project root to sys.path to import models
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.mlp import GraphAwareMLP
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant

DATA_FILE = os.getenv("DATA_FILE")
CLIENT_ID = os.getenv("CLIENT_ID", "")
ENABLE_DP = os.getenv("ENABLE_DP", "true").lower() in ("true", "1", "yes")
DP_CLIP_NORM = float(os.getenv("DP_CLIP_NORM", "1.0"))
DP_NOISE_MULTIPLIER = float(os.getenv("DP_NOISE_MULTIPLIER", "0.5"))
DP_DELTA = float(os.getenv("DP_DELTA", "1e-5"))
POSITIVE_LOSS_WEIGHT = float(os.getenv("POSITIVE_LOSS_WEIGHT", "10.0"))
CLIENT_LEARNING_RATE = float(os.getenv("CLIENT_LEARNING_RATE", "0.002"))
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "1"))
FEDPROX_MU = float(os.getenv("FEDPROX_MU", "0.0"))
DP_ACCOUNTANT_TAG = os.getenv("DP_ACCOUNTANT_TAG", "default")


def compute_dp_budget(n_samples, batch_size, epochs, noise_multiplier, delta=DP_DELTA):
    """Estimate (epsilon, delta) with the same RDP accountant used for training."""
    if not ENABLE_DP:
        return 0.0, 0.0
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if epochs < 0:
        raise ValueError("epochs must be non-negative.")
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive.")
    if not 0 < delta < 1:
        raise ValueError("delta must be between 0 and 1.")

    steps_per_epoch = math.ceil(n_samples / batch_size)
    sample_rate = 1.0 / steps_per_epoch
    accountant = RDPAccountant()
    for _ in range(epochs * steps_per_epoch):
        accountant.step(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
        )
    return float(accountant.get_epsilon(delta=delta)), float(delta)


def _save_privacy_accountant_state(accountant, path):
    state = {
        "history": accountant.history,
        "mechanism": accountant.mechanism(),
    }
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as state_file:
        json.dump(state, state_file, indent=2)
    os.replace(temporary_path, path)


def _load_privacy_accountant_state(accountant, path):
    with open(path, encoding="utf-8") as state_file:
        state = json.load(state_file)
    if not isinstance(state, dict) or state.get("mechanism") != accountant.mechanism():
        raise ValueError(f"Privacy accountant state at {path} has an unsupported mechanism.")
    history = state.get("history")
    if not isinstance(history, list) or any(not isinstance(entry, list) or len(entry) != 3 for entry in history):
        raise ValueError(f"Privacy accountant state at {path} has an invalid history.")
    state["history"] = [tuple(entry) for entry in history]
    state["mechanism"] = type(accountant).mechanism
    accountant.load_state_dict(state)


def _load_manifest():
    manifest_path = os.path.join(project_root, "data", "preprocess_manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, encoding="utf-8") as manifest_file:
            return json.load(manifest_file)
    except (OSError, json.JSONDecodeError):
        return {}


def load_data():
    """
    Load a bank shard.

    Features are already globally standardised (train-only scaler + PCA +
    train-only post-PCA scaler in data/preprocess.py).  We intentionally do NOT
    fit a client-local scaler here: doing so on the whole shard before the local
    split was one of the leakage paths, and it also made client inputs
    incompatible with the server's global holdout.
    """
    global DATA_FILE
    if not DATA_FILE:
        if CLIENT_ID:
            DATA_FILE = os.path.join(current_dir, "data", f"bank_{CLIENT_ID}.csv")
        else:
            DATA_FILE = os.path.join(current_dir, "data", "bank_1.csv")

    if not os.path.exists(DATA_FILE):
        candidate = os.path.join(project_root, DATA_FILE)
        if os.path.exists(candidate):
            DATA_FILE = candidate
        else:
            raise FileNotFoundError(f"Data file not found at {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    manifest = _load_manifest()
    if manifest.get("pipeline_version") != "causal-v3" or manifest.get("status") != "complete":
        raise RuntimeError("Client data must be generated by a completed causal-v3 preprocessing run.")

    X = df.drop(
        columns=['is_laundering', 'sender', 'receiver', 'timestamp', 'nameOrig',
                 'nameDest', '_pk', '_edge_key'],
        errors='ignore',
    )
    X = X.select_dtypes(include=[np.number])

    # Respect the training schema recorded by preprocess.py when available.
    feature_columns = manifest.get("feature_columns")
    if feature_columns and set(feature_columns).issubset(set(X.columns)):
        X = X.reindex(columns=feature_columns, fill_value=0.0)

    y = df['is_laundering'].values.astype(np.float32)
    return X.values.astype(np.float32), y


class AMLClient(fl.client.NumPyClient):
    def __init__(self, client_id=None):
        resolved_client_id = client_id or CLIENT_ID
        if not resolved_client_id and DATA_FILE:
            filename = os.path.basename(DATA_FILE)
            if filename.startswith("bank_") and filename.endswith(".csv"):
                resolved_client_id = filename[len("bank_"):-len(".csv")]
        self.client_id = resolved_client_id or "1"
        self.X, self.y = load_data()

        # Keep a natural local validation slice for client diagnostics only.
        stratify = self.y if np.bincount(self.y.astype(int)).min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=stratify
        )

        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)

        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.test_dataset = TensorDataset(self.X_test, self.y_test)

        self.batch_size = 256
        drop_last = False
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, drop_last=drop_last,
        )
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        self.input_dim = self.X_train.shape[1]
        self.model = GraphAwareMLP(input_dim=self.input_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.fit_rounds = 0
        # Independent noise streams per simulated client. The global seed is
        # intentionally offset so clients do not reuse identical DP noise.
        try:
            client_seed = SEED + int(self.client_id)
        except (TypeError, ValueError):
            client_seed = SEED + sum(ord(ch) for ch in str(self.client_id))
        random.seed(client_seed)
        np.random.seed(client_seed)
        torch.manual_seed(client_seed)

        self.privacy_engine = None
        self.private_model = None
        self.private_optimizer = None
        self.private_train_loader = None
        if ENABLE_DP:
            if DP_NOISE_MULTIPLIER <= 0 or DP_CLIP_NORM <= 0:
                raise ValueError("DP-SGD requires positive DP_NOISE_MULTIPLIER and DP_CLIP_NORM.")
            self.privacy_engine = PrivacyEngine(accountant="rdp")
            run_id = _load_manifest().get("run_id", "untracked-run")
            safe_tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in DP_ACCOUNTANT_TAG)
            self.privacy_state_path = os.path.join(
                current_dir, f"privacy_accountant_{run_id}_{safe_tag}_{self.client_id}.json"
            )
            if os.path.exists(self.privacy_state_path):
                _load_privacy_accountant_state(
                    self.privacy_engine.accountant,
                    self.privacy_state_path,
                )
            base_optimizer = optim.Adam(self.model.parameters(), lr=CLIENT_LEARNING_RATE)
            (
                self.private_model,
                self.private_optimizer,
                self.private_train_loader,
            ) = self.privacy_engine.make_private(
                module=self.model,
                optimizer=base_optimizer,
                data_loader=self.train_loader,
                noise_multiplier=DP_NOISE_MULTIPLIER,
                max_grad_norm=DP_CLIP_NORM,
                poisson_sampling=True,
            )

    def get_parameters(self, config):
        return self.model.get_parameters()

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # FedProx reference is the received global model for this round. A zero
        # value exactly recovers the existing local objective (FedAvg clients).
        global_parameters = [parameter.detach().clone() for parameter in self.model.parameters()]
        self.fit_rounds += 1
        criterion = nn.BCELoss(reduction="none")
        optimizer = self.private_optimizer if ENABLE_DP else optim.Adam(
            self.model.parameters(), lr=CLIENT_LEARNING_RATE
        )
        training_model = self.private_model if ENABLE_DP else self.model
        training_loader = self.private_train_loader if ENABLE_DP else self.train_loader
        training_model.train()
        if LOCAL_EPOCHS < 1:
            raise ValueError("LOCAL_EPOCHS must be at least 1.")
        epochs = LOCAL_EPOCHS
        for epoch in range(epochs):
            for X_batch, y_batch in training_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = training_model(X_batch).squeeze()

                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)

                per_example_loss = criterion(outputs, y_batch)
                weights = torch.where(
                    y_batch > 0.5,
                    torch.full_like(y_batch, POSITIVE_LOSS_WEIGHT),
                    torch.ones_like(y_batch),
                )
                weighted_loss = per_example_loss * weights
                if FEDPROX_MU > 0:
                    proximal_penalty = sum(
                        torch.sum((parameter - reference) ** 2)
                        for parameter, reference in zip(self.model.parameters(), global_parameters)
                    )
                    # Add the same proximal term to each per-example loss so
                    # Opacus can still clip per-sample gradients correctly.
                    weighted_loss = weighted_loss + (0.5 * FEDPROX_MU * proximal_penalty)
                loss = weighted_loss.mean()
                loss.backward()
                optimizer.step()

        fit_metrics = {
            "dp_enabled": ENABLE_DP,
            "dp_delta": DP_DELTA if ENABLE_DP else 0.0,
            "dp_accountant": "rdp" if ENABLE_DP else "disabled",
            # Opacus secure_mode is intentionally not enabled for this local PoC
            # because it requires the optional torchcsprng package. Do not present
            # this run as production-grade privacy protection.
            "dp_rng_mode": "standard_prng_demo_only" if ENABLE_DP else "disabled",
        }
        if ENABLE_DP:
            # Flower's metric records cannot contain None values. Omit epsilon
            # entirely for diagnostic runs with DP disabled.
            fit_metrics["dp_epsilon"] = float(self.privacy_engine.get_epsilon(DP_DELTA))
            _save_privacy_accountant_state(
                self.privacy_engine.accountant,
                self.privacy_state_path,
            )
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

        # Exact confusion matrix components on the untouched local test split
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
            "tn": tn,
            "positives": int((y_true_np == 1).sum()),
            "negatives": int((y_true_np == 0).sum()),
            "synthetic_in_test": 0,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VFI bank client (Flower NumPyClient)")
    parser.add_argument(
        "--cid", type=str, default=os.getenv("CLIENT_ID", ""),
        help="Bank/client id used to select fl_implementation/data/bank_<cid>.csv",
    )
    args = parser.parse_args()

    server_addr = os.getenv("SERVER_ADDRESS", "localhost:8080")
    if os.path.exists("/.dockerenv") or os.environ.get("DATA_FILE", "").startswith("data/"):
        server_addr = "fl_server:8080"

    client_id = args.cid
    if client_id:
        # load_data() reads the module-level CLIENT_ID to resolve its shard.
        CLIENT_ID = client_id
    fl.client.start_numpy_client(
        server_address=server_addr,
        client=AMLClient(client_id=client_id),
    )
