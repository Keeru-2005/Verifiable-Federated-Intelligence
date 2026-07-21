"""
Standalone training / sanity-check loop for GraphAwareMLP.
(Week 1, Day 2 — Likith)
"""

import os
import sys
import csv
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.mlp import GraphAwareMLP


LABEL_COL = "is_laundering"
DEFAULT_DATA_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "processed_partition_best.csv"),
    os.path.join(SCRIPT_DIR, "data", "bank_1.csv"),
]
METRICS_CSV_PATH = os.path.join(SCRIPT_DIR, "fl_metrics.csv")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "model_standalone.pt")


def load_dataset(data_path: str):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find {data_path}. Run data/preprocess.py (and, if using "
            f"the bank-shard fallback, fl_implementation/split_into_banks.py) first."
        )
    df = pd.read_csv(data_path)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Expected label column '{LABEL_COL}' not found in {data_path}")

    X = df.drop(columns=[LABEL_COL]).values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32)
    return X, y


def resolve_data_path(explicit_path):
    if explicit_path:
        return explicit_path
    for candidate in DEFAULT_DATA_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return DEFAULT_DATA_CANDIDATES[0]


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_prob, y_pred = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch).squeeze()
            if out.dim() == 0:
                out = out.unsqueeze(0)
            loss = criterion(out, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            y_true.extend(y_batch.cpu().numpy())
            y_prob.extend(out.cpu().numpy())
            y_pred.extend((out.cpu().numpy() > 0.5).astype(int))

    avg_loss = total_loss / max(1, len(y_true))
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5
    return {
        "loss": avg_loss,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
    }


def main():
    parser = argparse.ArgumentParser(description="Standalone GraphAwareMLP sanity-check trainer")
    parser.add_argument("--data", type=str, default=None, help="Path to preprocessed CSV")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--expected-input-dim", type=int, default=32)
    args = parser.parse_args()

    data_path = resolve_data_path(args.data)
    print(f"[train_standalone] Loading data from: {data_path}")
    X, y = load_dataset(data_path)

    actual_input_dim = X.shape[1]
    if actual_input_dim != args.expected_input_dim:
        raise AssertionError(
            f"Feature count mismatch! Data has {actual_input_dim} feature columns, "
            f"but GraphAwareMLP is being asked to expect {args.expected_input_dim}."
        )
    print(f"[train_standalone] Confirmed input_dim = {actual_input_dim} (matches GraphAwareMLP)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=args.batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=args.batch_size, shuffle=False,
    )

    model = GraphAwareMLP(input_dim=actual_input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    os.makedirs(os.path.dirname(METRICS_CSV_PATH), exist_ok=True)
    with open(METRICS_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "global_auc", "accuracy", "precision", "recall", "f1", "loss", "timestamp"])

    print(f"[train_standalone] Training on {device} for {args.epochs} epochs "
          f"({len(X_train)} train / {len(X_test)} test rows)")

    for epoch in range(1, args.epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch).squeeze()
            if out.dim() == 0:
                out = out.unsqueeze(0)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

        metrics = evaluate(model, test_loader, criterion, device)
        print(f"  epoch {epoch:02d}/{args.epochs} | loss={metrics['loss']:.4f} "
              f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}")

        with open(METRICS_CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, metrics["auc"], metrics["accuracy"], metrics["precision"],
                metrics["recall"], metrics["f1"], metrics["loss"],
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            ])

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"[train_standalone] Done. Metrics -> {METRICS_CSV_PATH}")
    print(f"[train_standalone] Checkpoint -> {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()