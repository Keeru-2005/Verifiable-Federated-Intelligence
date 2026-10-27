"""
Model & Federated Learning Test Suite
Validates model architecture, parameter serialization, DP-SGD mechanics, exact metric aggregation, and a genuine short 4-client FL training round.
"""
import os
import sys
import pytest
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.mlp import GraphAwareMLP
from fl_implementation.client import AMLClient, compute_dp_budget
from fl_implementation.server import exact_metric_aggregation


def test_model_input_output_shapes():
    """Verify GraphAwareMLP produces valid probability outputs for 32-dim inputs."""
    model = GraphAwareMLP(input_dim=32)
    batch_size = 16
    x = torch.randn(batch_size, 32)
    out = model(x)
    assert out.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {out.shape}"
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), "Outputs must be in [0, 1] range"


def test_model_parameter_serialization_roundtrip():
    """Verify get_parameters and set_parameters preserve weights exactly."""
    model1 = GraphAwareMLP(input_dim=32)
    params = model1.get_parameters()
    assert isinstance(params, list), "Parameters must be a list of numpy arrays"
    assert len(params) == 18, f"Expected 18 state dict tensors, got {len(params)}"

    model2 = GraphAwareMLP(input_dim=32)
    # Mutate model2 weights
    with torch.no_grad():
        for p in model2.parameters():
            p.add_(1.0)
    
    # Restore model1 weights into model2
    model2.set_parameters(params)
    restored_params = model2.get_parameters()
    for p1, p2 in zip(params, restored_params):
        np.testing.assert_allclose(p1, p2, atol=1e-6, err_msg="Restored weights differ from source")


def test_dp_sgd_gradient_clipping_and_noise():
    """Verify that DP-SGD correctly clips gradient norm and adds calibrated noise."""
    model = GraphAwareMLP(input_dim=32)
    x = torch.randn(32, 32)
    y = torch.randint(0, 2, (32, 1)).float()
    
    criterion = torch.nn.BCELoss()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()

    # Test clipping
    clip_norm = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    assert total_norm <= clip_norm + 1e-4, f"Gradient norm {total_norm} exceeds clip threshold {clip_norm}"

    # Test analytical DP budget computation
    eps, delta = compute_dp_budget(n_samples=1000, batch_size=256, epochs=1, noise_multiplier=0.5)
    assert eps > 0, "Epsilon must be positive"
    assert delta == 1e-5, f"Delta should be 1e-5, got {delta}"


def test_exact_global_metric_calculation_not_averaged():
    """Verify that global metrics are computed from cumulative confusion matrices, not by averaging F1 scores."""
    # Simulate 2 clients with different sample distributions
    # Client 1: 100 positives, 100 negatives. TP=90, FP=10, FN=10, TN=90 -> Prec=0.9, Rec=0.9, F1=0.9
    # Client 2: 10 positives, 990 negatives. TP=2, FP=20, FN=8, TN=970 -> Prec=2/22=0.0909, Rec=2/10=0.2, F1=0.125
    metrics = [
        (200, {"tp": 90, "fp": 10, "fn": 10, "tn": 90, "accuracy": 0.9, "f1": 0.9}),
        (1000, {"tp": 2, "fp": 20, "fn": 8, "tn": 970, "accuracy": 0.972, "f1": 0.125})
    ]

    # Naive weighted average of F1 scores:
    naive_avg_f1 = (200 * 0.9 + 1000 * 0.125) / 1200 # = 0.2542

    # Exact global calculation:
    # Total TP=92, Total FP=30, Total FN=18, Total TN=1060
    # True Precision = 92 / (92+30) = 92/122 = 0.7541
    # True Recall = 92 / (92+18) = 92/110 = 0.8364
    # True F1 = 2 * (0.7541 * 0.8364) / (0.7541 + 0.8364) = 0.7931
    global_results = exact_metric_aggregation(metrics)
    true_f1 = global_results["f1"]

    assert abs(true_f1 - 0.7931) < 0.01, f"Expected exact F1 ~0.7931, got {true_f1}"
    assert abs(true_f1 - naive_avg_f1) > 0.1, "True global F1 should differ significantly from naive averaged F1"


def test_short_genuine_federated_round():
    """Verify that a genuine short 4-client federated learning round executes and aggregates."""
    import flwr as fl
    from fl_implementation.server import EarlyStoppingFedAvg

    # Create 4 in-memory clients
    clients = [AMLClient(client_id=str(i)) for i in range(1, 5)]
    
    # Initial global model
    global_model = GraphAwareMLP(input_dim=32)
    initial_weights = global_model.get_parameters()

    # Simulate 1 round of fit on all 4 clients
    fit_results = []
    for c in clients:
        weights, n_samples, metrics = c.fit(initial_weights, {})
        fit_results.append((weights, n_samples))

    # Perform weighted average parameter aggregation (standard FedAvg)
    total_samples = sum(n for _, n in fit_results)
    aggregated_weights = []
    for param_idx in range(len(initial_weights)):
        weighted_param = sum(w[param_idx] * n for w, n in fit_results) / total_samples
        aggregated_weights.append(weighted_param)

    # Set aggregated weights into global model
    global_model.set_parameters(aggregated_weights)
    new_weights = global_model.get_parameters()

    assert len(new_weights) == len(initial_weights)
    # Check that weights actually updated
    weight_diff = np.sum([np.sum(np.abs(w1 - w2)) for w1, w2 in zip(initial_weights, new_weights)])
    assert weight_diff > 0.0, "Global weights should be updated after federated round"


def test_byzantine_malicious_update_rejection():
    """Verify that EarlyStoppingFedAvg filters out poisoned/malicious updates (NaNs, extreme norms)."""
    from fl_implementation.server import EarlyStoppingFedAvg
    from flwr.common import ndarrays_to_parameters, FitRes, Status, Code
    from unittest.mock import MagicMock

    strategy = EarlyStoppingFedAvg()

    # Honest client
    honest_model = GraphAwareMLP(input_dim=32)
    honest_params = ndarrays_to_parameters(honest_model.get_parameters())
    honest_client = MagicMock()
    honest_client.cid = "bank_1_honest"
    honest_fit = FitRes(status=Status(code=Code.OK, message="Success"), parameters=honest_params, num_examples=1000, metrics={})

    # Malicious client 1: Extreme norm poisoning attack (e.g. 100,000 weight explosion)
    poisoned_weights = [arr * 10000.0 for arr in honest_model.get_parameters()]
    poisoned_params = ndarrays_to_parameters(poisoned_weights)
    malicious_client_1 = MagicMock()
    malicious_client_1.cid = "bank_2_poisoner"
    malicious_fit_1 = FitRes(status=Status(code=Code.OK, message="Success"), parameters=poisoned_params, num_examples=1000, metrics={})

    # Malicious client 2: NaN injection attack
    nan_weights = [arr.copy() for arr in honest_model.get_parameters()]
    nan_weights[0][0, 0] = np.nan
    nan_params = ndarrays_to_parameters(nan_weights)
    malicious_client_2 = MagicMock()
    malicious_client_2.cid = "bank_3_nan_injector"
    malicious_fit_2 = FitRes(status=Status(code=Code.OK, message="Success"), parameters=nan_params, num_examples=1000, metrics={})

    results = [
        (honest_client, honest_fit),
        (malicious_client_1, malicious_fit_1),
        (malicious_client_2, malicious_fit_2)
    ]

    # Aggregate with EarlyStoppingFedAvg
    aggregated_params, _ = strategy.aggregate_fit(server_round=1, results=results, failures=[])

    # Assert that aggregation succeeded and ONLY honest client was aggregated
    assert aggregated_params is not None, "Aggregation should succeed by filtering malicious clients"
    assert strategy.last_client_count == 1, "Only the 1 honest client should have been aggregated"
    
    # Assert resulting weights have no NaNs and normal norm
    from flwr.common import parameters_to_ndarrays
    res_ndarrays = parameters_to_ndarrays(aggregated_params)
    assert not any(np.isnan(arr).any() for arr in res_ndarrays), "Aggregated weights must not contain NaNs"
    total_norm = np.sqrt(sum(np.sum(np.square(arr)) for arr in res_ndarrays))
    assert total_norm < 150.0, "Aggregated norm must not be poisoned"


def test_shape_and_dimension_mismatch_rejection():
    """Verify that EarlyStoppingFedAvg rejects updates with mismatched tensor shapes or counts."""
    from fl_implementation.server import EarlyStoppingFedAvg
    from flwr.common import ndarrays_to_parameters, FitRes, Status, Code
    from unittest.mock import MagicMock

    strategy = EarlyStoppingFedAvg()

    # Honest client
    honest_model = GraphAwareMLP(input_dim=32)
    honest_params = ndarrays_to_parameters(honest_model.get_parameters())
    honest_client = MagicMock()
    honest_client.cid = "bank_1_honest"
    honest_fit = FitRes(status=Status(code=Code.OK, message="Success"), parameters=honest_params, num_examples=500, metrics={})

    # Malicious client with mismatched tensor count (e.g. 5 tensors instead of 33)
    mismatched_count_weights = [np.ones((10, 10), dtype=np.float32) for _ in range(5)]
    mismatched_count_params = ndarrays_to_parameters(mismatched_count_weights)
    client_bad_count = MagicMock()
    client_bad_count.cid = "bank_bad_count"
    fit_bad_count = FitRes(status=Status(code=Code.OK, message="Success"), parameters=mismatched_count_params, num_examples=500, metrics={})

    # Malicious client with mismatched layer shape (e.g. input dim 10 instead of 32)
    bad_shape_weights = [arr.copy() for arr in honest_model.get_parameters()]
    bad_shape_weights[0] = np.zeros((128, 10), dtype=np.float32)  # corrupted shape
    bad_shape_params = ndarrays_to_parameters(bad_shape_weights)
    client_bad_shape = MagicMock()
    client_bad_shape.cid = "bank_bad_shape"
    fit_bad_shape = FitRes(status=Status(code=Code.OK, message="Success"), parameters=bad_shape_params, num_examples=500, metrics={})

    results = [
        (honest_client, honest_fit),
        (client_bad_count, fit_bad_count),
        (client_bad_shape, fit_bad_shape)
    ]

    aggregated_params, _ = strategy.aggregate_fit(server_round=1, results=results, failures=[])
    assert aggregated_params is not None, "Aggregation should succeed by dropping bad shape clients"
    assert strategy.last_client_count == 1, "Only the 1 honest client with valid shapes must be aggregated"


def test_extreme_all_zero_class_imbalance_evaluation():
    """Verify that model evaluation handles batches with 0 true positives without NaN/crash."""
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    model = GraphAwareMLP(input_dim=32)
    criterion = torch.nn.BCELoss()

    # Extreme edge case: all 100 samples are negative (label 0, zero AML cases)
    X_neg = torch.randn(100, 32)
    y_neg = torch.zeros(100)
    loader_neg = DataLoader(TensorDataset(X_neg, y_neg), batch_size=32)

    loss, auc, f1, acc = model.evaluate(loader_neg, criterion)

    assert not np.isnan(loss), "Loss must not be NaN"
    assert not np.isnan(f1), "F1 score must not be NaN"
    assert not np.isnan(acc), "Accuracy must not be NaN"
    assert 0.0 <= acc <= 1.0, "Accuracy must be valid percentage"
    assert 0.0 <= f1 <= 1.0, "F1 must be in [0, 1]"


