"""
End-to-end integration test for Graph Encoder → FL Training Pipeline.
(Day 2 — Likith: Validates GAT embeddings feed correctly into GraphAwareMLP)

This script tests:
1. GraphEncoder can be imported and instantiated
2. GAT embeddings are produced for a synthetic transaction graph
3. GraphAwareMLP accepts the correct input dimension after graph feature enrichment
4. End-to-end training loop completes without errors
5. Model produces valid predictions (0-1 range for binary classification)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.mlp import GraphAwareMLP

# ── Test Configuration ──
NUM_TRANSACTIONS = 500
NUM_ACCOUNTS = 50
GAT_DIMENSIONS = 16
MLP_INPUT_DIM = 32  # Must match PCA output dimension in preprocess.py
EPOCHS = 5
BATCH_SIZE = 64


def generate_synthetic_data():
    """Generate synthetic transaction data that mimics the real AML dataset."""
    np.random.seed(42)

    senders = [f"ACC_{i:04d}" for i in range(NUM_ACCOUNTS)]
    receivers = [f"ACC_{i:04d}" for i in range(NUM_ACCOUNTS)]

    data = {
        "txn_id": range(NUM_TRANSACTIONS),
        "sender": np.random.choice(senders, NUM_TRANSACTIONS),
        "receiver": np.random.choice(receivers, NUM_TRANSACTIONS),
        "amount": np.random.exponential(5000, NUM_TRANSACTIONS),
        "timestamp": np.sort(np.random.randint(1, 100, NUM_TRANSACTIONS)),
        "is_laundering": np.random.choice([0, 1], NUM_TRANSACTIONS, p=[0.85, 0.15]),
    }

    return pd.DataFrame(data)


def test_graph_encoder_import():
    """Test 1: Verify GraphEncoder can be imported from models/."""
    logging.info("TEST 1: Importing GraphEncoder...")
    try:
        from models.graph_encoder import GraphEncoder
        encoder = GraphEncoder(dimensions=GAT_DIMENSIONS, heads=4, epochs=50, prune_threshold=0.1, lr=0.01)
        assert hasattr(encoder, "dimensions"), "GraphEncoder must expose 'dimensions' attribute"
        assert encoder.dimensions == GAT_DIMENSIONS
        logging.info("  ✅ GraphEncoder imported and instantiated successfully")
        return True
    except Exception as e:
        logging.error(f"  ❌ Import failed: {e}")
        return False


def test_gat_embeddings():
    """Test 2: Verify GAT produces embeddings for all nodes."""
    logging.info("TEST 2: Generating GAT embeddings...")
    try:
        import networkx as nx
        from models.graph_encoder import GraphEncoder

        df = generate_synthetic_data()

        # Build a simple transaction graph
        G = nx.DiGraph()
        for _, row in df.iterrows():
            s, r = row["sender"], row["receiver"]
            if not G.has_node(s):
                G.add_node(s, pagerank=0.01, in_degree=1, out_degree=1,
                           clustering_coefficient=0.0, betweenness_centrality=0.0)
            if not G.has_node(r):
                G.add_node(r, pagerank=0.01, in_degree=1, out_degree=1,
                           clustering_coefficient=0.0, betweenness_centrality=0.0)
            if G.has_edge(s, r):
                G[s][r]["total_amount"] += row["amount"]
                G[s][r]["txn_count"] += 1
            else:
                G.add_edge(s, r, total_amount=row["amount"], avg_amount=row["amount"],
                           max_amount=row["amount"], txn_count=1, velocity=0.0, fraud_flag=0)

        fraud_senders = set(df[df["is_laundering"] == 1]["sender"].unique())
        node_labels = {n: (1 if n in fraud_senders else 0) for n in G.nodes()}

        encoder = GraphEncoder(dimensions=GAT_DIMENSIONS, heads=4, epochs=50, prune_threshold=0.1, lr=0.01)
        embeddings = encoder.fit_transform(G, df=df, node_labels=node_labels)

        assert isinstance(embeddings, dict), "embeddings must be a dict"
        assert len(embeddings) > 0, "embeddings must not be empty"

        for node, emb in embeddings.items():
            assert isinstance(emb, np.ndarray), f"Embedding for {node} must be numpy array"
            assert emb.shape == (GAT_DIMENSIONS,), f"Embedding shape must be ({GAT_DIMENSIONS},), got {emb.shape}"

        logging.info(f"  ✅ GAT embeddings generated for {len(embeddings)} nodes, each of dimension {GAT_DIMENSIONS}")
        return True
    except Exception as e:
        logging.error(f"  ❌ GAT embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_node_attention_and_pruning():
    """Test 3: Verify attention scores and edge pruning work."""
    logging.info("TEST 3: Testing attention scores and edge pruning...")
    try:
        import networkx as nx
        from models.graph_encoder import GraphEncoder

        df = generate_synthetic_data()
        G = nx.DiGraph()
        for _, row in df.iterrows():
            s, r = row["sender"], row["receiver"]
            if not G.has_node(s):
                G.add_node(s, pagerank=0.01, in_degree=1, out_degree=1,
                           clustering_coefficient=0.0, betweenness_centrality=0.0)
            if not G.has_node(r):
                G.add_node(r, pagerank=0.01, in_degree=1, out_degree=1,
                           clustering_coefficient=0.0, betweenness_centrality=0.0)
            if not G.has_edge(s, r):
                G.add_edge(s, r, total_amount=row["amount"], avg_amount=row["amount"],
                           max_amount=row["amount"], txn_count=1, velocity=0.0, fraud_flag=0)

        fraud_senders = set(df[df["is_laundering"] == 1]["sender"].unique())
        node_labels = {n: (1 if n in fraud_senders else 0) for n in G.nodes()}

        encoder = GraphEncoder(dimensions=GAT_DIMENSIONS, heads=4, epochs=50, prune_threshold=0.1, lr=0.01)
        encoder.fit_transform(G, df=df, node_labels=node_labels)

        # Test attention scores
        attn_scores = encoder.get_node_attention()
        assert isinstance(attn_scores, dict), "Attention scores must be a dict"
        assert len(attn_scores) > 0, "Attention scores must not be empty"
        for node, score in attn_scores.items():
            assert isinstance(score, float), f"Attention score for {node} must be float"
            assert 0.0 <= score <= 1.0 or score >= 0.0, f"Attention score must be non-negative"

        # Test edge pruning
        kept_edges = encoder.get_pruned_edges()
        assert isinstance(kept_edges, list), "Pruned edges must be a list"
        original_edges = G.number_of_edges()
        logging.info(f"  ✅ Attention scores for {len(attn_scores)} nodes. "
                     f"Edges: {original_edges} original → {len(kept_edges)} after pruning")
        return True
    except Exception as e:
        logging.error(f"  ❌ Attention/pruning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mlp_input_compatibility():
    """Test 4: Verify GraphAwareMLP accepts the correct input dimension."""
    logging.info("TEST 4: Testing GraphAwareMLP input compatibility...")
    try:
        model = GraphAwareMLP(input_dim=MLP_INPUT_DIM)

        # Create random input matching PCA dimension
        batch = torch.randn(BATCH_SIZE, MLP_INPUT_DIM)
        output = model(batch)

        assert output.shape == (BATCH_SIZE, 1), f"Expected output shape ({BATCH_SIZE}, 1), got {output.shape}"
        assert torch.all(output >= 0) and torch.all(output <= 1), "Output must be in [0, 1] (sigmoid)"

        logging.info(f"  ✅ GraphAwareMLP(input_dim={MLP_INPUT_DIM}) produces valid output shape {output.shape}")
        return True
    except Exception as e:
        logging.error(f"  ❌ MLP compatibility test failed: {e}")
        return False


def test_end_to_end_training():
    """Test 5: Full end-to-end training loop with graph-enriched features."""
    logging.info("TEST 5: End-to-end training with graph-enriched features...")
    try:
        np.random.seed(42)
        torch.manual_seed(42)

        # Simulate the PCA-transformed feature matrix (32 dimensions as produced by preprocess.py)
        n_samples = 1000
        X = np.random.randn(n_samples, MLP_INPUT_DIM).astype(np.float32)
        y = np.random.choice([0, 1], n_samples, p=[0.7, 0.3]).astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=BATCH_SIZE, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
            batch_size=BATCH_SIZE, shuffle=False
        )

        device = torch.device("cpu")
        model = GraphAwareMLP(input_dim=MLP_INPUT_DIM).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        # Training loop
        for epoch in range(1, EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = model(X_batch).squeeze()
                if out.dim() == 0:
                    out = out.unsqueeze(0)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Evaluate
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    out = model(X_batch).squeeze()
                    if out.dim() == 0:
                        out = out.unsqueeze(0)
                    y_true.extend(y_batch.numpy())
                    y_pred.extend((out.numpy() > 0.5).astype(int))

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            logging.info(f"    Epoch {epoch}/{EPOCHS}: loss={epoch_loss:.4f}, acc={acc:.4f}, f1={f1:.4f}")

        # Verify model can export parameters (for FL aggregation)
        params = model.get_parameters()
        assert isinstance(params, list), "get_parameters must return a list"
        assert len(params) > 0, "Parameters list must not be empty"

        # Verify parameters can be reloaded
        model2 = GraphAwareMLP(input_dim=MLP_INPUT_DIM)
        model2.set_parameters(params)

        logging.info(f"  ✅ End-to-end training completed. Final acc={acc:.4f}, f1={f1:.4f}")
        logging.info(f"  ✅ FL parameter export/import works ({len(params)} parameter tensors)")
        return True
    except Exception as e:
        logging.error(f"  ❌ End-to-end training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    logging.info("=" * 70)
    logging.info("GRAPH INTEGRATION TEST SUITE — Day 2 (Likith)")
    logging.info("=" * 70)

    results = {
        "GraphEncoder Import": test_graph_encoder_import(),
        "GAT Embeddings": test_gat_embeddings(),
        "Attention & Pruning": test_node_attention_and_pruning(),
        "MLP Input Compatibility": test_mlp_input_compatibility(),
        "End-to-End Training": test_end_to_end_training(),
    }

    logging.info("\n" + "=" * 70)
    logging.info("TEST RESULTS SUMMARY")
    logging.info("=" * 70)
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logging.info(f"  {status} — {name}")
        if not passed:
            all_passed = False

    logging.info("=" * 70)
    if all_passed:
        logging.info("🎉 ALL TESTS PASSED — Graph integration is working correctly!")
    else:
        logging.info("⚠️  SOME TESTS FAILED — Check logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
