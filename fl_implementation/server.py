import flwr as fl
import sys

from flwr.common import parameters_to_ndarrays
import json
import os
from datetime import datetime, timezone
import torch
# Add project root directory to path for models import
server_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(server_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.mlp import GraphAwareMLP


import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def get_eval_fn(test_csv_path):
    """Factory for centralized server-side evaluation of the global model on holdout test set."""
    if not os.path.exists(test_csv_path):
        return None

    try:
        df = pd.read_csv(test_csv_path)
        X_df = df.drop(columns=['is_laundering', 'sender', 'receiver', 'timestamp', 'nameOrig', 'nameDest', '_pk', '_edge_key'], errors='ignore')
        X_df = X_df.select_dtypes(include=[np.number])
        y = df['is_laundering'].values.astype(np.float32)
        X = X_df.values.astype(np.float32)

        test_loader = DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(y)),
            batch_size=256, shuffle=False
        )
        input_dim = X.shape[1]
    except Exception as e:
        print(f"[WARN] Could not load holdout test set for centralized evaluation: {e}")
        return None

    def evaluate(server_round, parameters, config):
        model = GraphAwareMLP(input_dim=input_dim)
        model.set_parameters(parameters)
        criterion = torch.nn.BCELoss()
        loss, auc, f1, acc = model.evaluate(test_loader, criterion)
        
        # Exact precision and recall on global test set
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                out = model(X_batch).squeeze()
                if out.dim() == 0:
                    out = out.unsqueeze(0)
                y_true.extend(y_batch.numpy())
                y_pred.extend((out.numpy() > 0.5).astype(int))
                
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        return float(loss), {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc": float(auc)
        }

    return evaluate


# Custom strategy subclassing FedAvg for server-side early stopping and exact metric aggregation
class EarlyStoppingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, patience=2, min_delta=0.0002, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.min_delta = min_delta
        self.best_f1 = 0.0
        self.patience_counter = 0

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Byzantine Fault Tolerance: Filter out malicious, corrupted, or poisoned client updates
        clean_results = []
        for client_proxy, fit_res in results:
            # 1. Reject invalid sample counts
            if fit_res.num_examples <= 0:
                print(f"[SECURITY] Dropping update from {client_proxy.cid}: invalid sample count ({fit_res.num_examples})")
                continue

            # 2. Inspect parameter arrays for NaNs, Infs, or extreme norm poisoning
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            
            # Validate parameter count and layer shapes
            expected_shapes = [arr.shape for arr in GraphAwareMLP(input_dim=32).get_parameters()]
            if len(ndarrays) != len(expected_shapes):
                print(f"[SECURITY] Dropping malformed update from {client_proxy.cid}: tensor count mismatch ({len(ndarrays)} != {len(expected_shapes)})")
                continue

            shapes_match = all(arr.shape == exp_shape for arr, exp_shape in zip(ndarrays, expected_shapes))
            if not shapes_match:
                print(f"[SECURITY] Dropping malformed update from {client_proxy.cid}: layer dimension mismatch")
                continue

            has_nan_or_inf = any(np.isnan(arr).any() or np.isinf(arr).any() for arr in ndarrays)
            if has_nan_or_inf:
                print(f"[SECURITY] Dropping malicious update from {client_proxy.cid}: detected NaN/Inf parameter values")
                continue

            # 3. L2 Norm threshold check (Byzantine model poisoning defense)
            total_l2_norm = np.sqrt(sum(np.sum(np.square(arr)) for arr in ndarrays))
            MAX_TOLERABLE_L2_NORM = 150.0  # Normal trained MLP weights have L2 norm < 25.0
            if total_l2_norm > MAX_TOLERABLE_L2_NORM:
                print(f"[SECURITY] Dropping poisoned update from {client_proxy.cid}: extreme L2 norm ({total_l2_norm:.2f} > {MAX_TOLERABLE_L2_NORM})")
                continue

            clean_results.append((client_proxy, fit_res))

        if not clean_results:
            print("[SECURITY] All client updates were rejected as malicious or invalid! Aborting round aggregation.")
            return None, {}

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, clean_results, failures)
        if parameters_aggregated is not None:
            ndarrays = parameters_to_ndarrays(parameters_aggregated)
            self.last_weights = [arr.tolist() for arr in ndarrays]
            self.last_client_count = len(clean_results)
            
            # Serialize global model weights into PyTorch format (model.pt)
            try:
                in_dim = ndarrays[0].shape[1]
                model = GraphAwareMLP(input_dim=in_dim)
                model.set_parameters(ndarrays)
                checkpoint_path = os.path.join(project_root, "fl_implementation", "model.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"[SAVE] PyTorch global model checkpoint saved successfully to {checkpoint_path}")
            except Exception as e:
                print(f"[WARN] Error saving model checkpoint: {e}")
                
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        # Compute exact global metrics from confusion matrix components across clients
        if results:
            client_metrics = [r.metrics for _, r in results if r.metrics]
            total_tp = sum(m.get("tp", 0) for m in client_metrics)
            total_fp = sum(m.get("fp", 0) for m in client_metrics)
            total_fn = sum(m.get("fn", 0) for m in client_metrics)
            total_tn = sum(m.get("tn", 0) for m in client_metrics)
            total_examples = total_tp + total_fp + total_fn + total_tn

            if total_examples > 0:
                acc = (total_tp + total_tn) / total_examples
                prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                
                exact_metrics = {
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1)
                }
                metrics_aggregated = exact_metrics

        # Write aggregated weights and exact global metrics to json
        if hasattr(self, 'last_weights'):
            metrics_dict = metrics_aggregated if metrics_aggregated is not None else {}
            export_data = {
                "round": server_round,
                "weights": self.last_weights,
                "metrics": {
                    "accuracy": metrics_dict.get("accuracy", 0.0),
                    "precision": metrics_dict.get("precision", 0.0),
                    "recall": metrics_dict.get("recall", 0.0),
                    "f1": metrics_dict.get("f1", 0.0)
                },
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "client_count": getattr(self, 'last_client_count', 4)
            }
            
            schemas_dir = os.path.join(project_root, "schemas")
            os.makedirs(schemas_dir, exist_ok=True)
            export_path = os.path.join(schemas_dir, f"global_model_round_{server_round}.json")
            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2)
            print(f"[SAVE] Exported global model for round {server_round} to {export_path}")
        
        if metrics_aggregated is None:
            return loss_aggregated, metrics_aggregated
            
        current_f1 = metrics_aggregated.get("f1", 0.0)
        
        if current_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = current_f1
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        print(f"[METRICS] Early Stopping Status: Patience {self.patience_counter}/{self.patience} | Best F1: {self.best_f1:.4f} | Current F1: {current_f1:.4f}")
        
        if self.patience_counter >= self.patience and server_round > 2:
            print("\n[STOP] EARLY STOPPING TRIGGERED: Global F1-score has saturated.")
            print(f"[RESULT] Final Aggregated F1 Score: {self.best_f1:.4f}")
            print("Stopping federated learning loop successfully...")
            sys.exit(0)
            
        return loss_aggregated, metrics_aggregated


def exact_metric_aggregation(metrics):
    """Exact sample-count-weighted and confusion-matrix metric aggregator."""
    total_tp = sum(m.get("tp", 0) for _, m in metrics)
    total_fp = sum(m.get("fp", 0) for _, m in metrics)
    total_fn = sum(m.get("fn", 0) for _, m in metrics)
    total_tn = sum(m.get("tn", 0) for _, m in metrics)
    total_examples = total_tp + total_fp + total_fn + total_tn

    if total_examples > 0 and (total_tp + total_fp) > 0 and (total_tp + total_fn) > 0:
        accuracy = (total_tp + total_tn) / total_examples
        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        # Fallback to weighted average if confusion matrix was not passed
        total_examples = sum(num_examples for num_examples, _ in metrics)
        aggregated = {}
        for num_examples, m in metrics:
            for k, v in m.items():
                if k not in ("tp", "fp", "fn", "tn"):
                    aggregated[k] = aggregated.get(k, 0) + v * num_examples
        for k in aggregated:
            aggregated[k] /= max(1, total_examples)
        accuracy = aggregated.get("accuracy", 0.0)
        precision = aggregated.get("precision", 0.0)
        recall = aggregated.get("recall", 0.0)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n[METRICS] EXACT GLOBAL ROUND SUMMARY")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("-" * 30)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VFI Federated Learning Server")
    parser.add_argument("--rounds", type=int, default=int(os.getenv("FL_ROUNDS", "10")), help="Number of FL rounds")
    parser.add_argument("--min-clients", type=int, default=int(os.getenv("MIN_CLIENTS", "4")), help="Min clients required")
    parser.add_argument("--port", type=int, default=8080, help="FL server port")
    args = parser.parse_args()

    print(f"Starting FL Server on port {args.port} (target rounds: {args.rounds}, min clients: {args.min_clients})...")
    
    test_csv_path = os.path.join(project_root, "fl_implementation", "data", "global_test.csv")
    eval_fn = get_eval_fn(test_csv_path)

    strategy = EarlyStoppingFedAvg(
        patience=2,
        min_delta=0.0002,
        evaluate_fn=eval_fn,
        evaluate_metrics_aggregation_fn=exact_metric_aggregation,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
    )

    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )