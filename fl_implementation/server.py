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
from evaluation.threshold_metrics import binary_classification_metrics, select_f1_threshold


import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score


def _load_manifest():
    path = os.path.join(project_root, "data", "preprocess_manifest.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def get_eval_fn(test_csv_path, split_name="validation"):
    """
    Load a naturally imbalanced validation or final-test period. The validation
    evaluator may be used for round selection; the final-test evaluator must only
    be called once after the best validation round is selected.
    """
    if not os.path.exists(test_csv_path):
        return None

    try:
        df = pd.read_csv(test_csv_path)
        X_df = df.drop(
            columns=['is_laundering', 'sender', 'receiver', 'timestamp', 'nameOrig',
                     'nameDest', '_pk', '_edge_key'],
            errors='ignore',
        )
        X_df = X_df.select_dtypes(include=[np.number])

        feature_columns = _load_manifest().get("feature_columns")
        if feature_columns and set(feature_columns).issubset(set(X_df.columns)):
            X_df = X_df.reindex(columns=feature_columns, fill_value=0.0)

        y = df['is_laundering'].values.astype(np.float32)
        X = X_df.values.astype(np.float32)

        test_loader = DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(y)),
            batch_size=256, shuffle=False,
        )
        input_dim = X.shape[1]
        n_positive = int((y == 1).sum())
        n_negative = int((y == 0).sum())
    except Exception as e:
        print(f"[WARN] Could not load holdout test set for centralized evaluation: {e}")
        return None

    def evaluate(server_round, parameters, config):
        model = GraphAwareMLP(input_dim=input_dim)
        # Flower may pass either a Parameters object or a list of ndarrays.
        if not isinstance(parameters, (list, tuple)):
            parameters = parameters_to_ndarrays(parameters)
        model.set_parameters(parameters)
        criterion = torch.nn.BCELoss()
        loss, auc, _, _ = model.evaluate(test_loader, criterion)

        # Exact precision/recall/confusion matrix on the global holdout
        model.eval()
        y_true, y_scores = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                out = model(X_batch).squeeze()
                if out.dim() == 0:
                    out = out.unsqueeze(0)
                y_true.extend(y_batch.numpy())
                y_scores.extend(out.numpy())

        y_true = np.asarray(y_true, dtype=np.int32)
        y_scores = np.asarray(y_scores, dtype=np.float64)
        if split_name == "validation":
            # Select the operating threshold on validation only. The final-test
            # evaluator receives this frozen threshold through its config.
            threshold = select_f1_threshold(y_true, y_scores)
        else:
            threshold = float(config.get("decision_threshold", 0.5))
        metrics = binary_classification_metrics(y_true, y_scores, threshold)
        metrics.update({
            "auc": float(auc),
            "average_precision": float(average_precision_score(y_true, y_scores)),
            "split": split_name,
        })
        return float(loss), metrics

    return evaluate


def aggregate_privacy_metrics(client_metrics):
    """Surface the least favorable client accountant value in Flower's round log."""
    rows = [metrics for _, metrics in client_metrics if metrics]
    enabled = [row for row in rows if row.get("dp_enabled") and row.get("dp_epsilon") is not None]
    if not enabled:
        return {"dp_enabled_clients": 0, "dp_rng_mode": "disabled"}
    return {
        "dp_enabled_clients": len(enabled),
        "max_client_epsilon": max(float(row["dp_epsilon"]) for row in enabled),
        "delta": max(float(row.get("dp_delta", 0.0)) for row in enabled),
        "dp_accountant": "rdp",
        "dp_rng_mode": "standard_prng_demo_only",
    }


# Validation-selected FedAvg with one final-test evaluation after training.
class EarlyStoppingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, target_rounds, final_evaluate_fn=None, min_delta=0.0002, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_rounds = target_rounds
        self.final_evaluate_fn = final_evaluate_fn
        self.min_delta = min_delta
        self.best_f1 = 0.0
        self.best_round = None
        self.best_weights = None
        self.best_threshold = 0.5
        self.final_test_metrics = None

    def _schemas_dir(self):
        return os.environ.get("VFI_SCHEMAS_DIR") or os.path.join(project_root, "schemas")

    @staticmethod
    def _output_layer_parameters(parameters):
        model = GraphAwareMLP(input_dim=32)
        model.set_parameters(parameters)
        state = model.state_dict()
        return (
            state["output_layer.0.weight"].detach().cpu().reshape(-1).tolist()
            + state["output_layer.0.bias"].detach().cpu().reshape(-1).tolist()
        )

    def _save_exact_round_updates(self, server_round, results, global_parameters):
        clients = []
        for proxy, fit_result in results:
            client_parameters = parameters_to_ndarrays(fit_result.parameters)
            clients.append({
                "client_id": str(proxy.cid),
                # This is the precise `num_examples` Flower uses for FedAvg.
                "num_examples": int(fit_result.num_examples),
                "output_layer_parameters": self._output_layer_parameters(client_parameters),
            })
        source = {
            "round": int(server_round),
            "run_id": _load_manifest().get("run_id"),
            "aggregation": "Flower FedAvg",
            "clients": clients,
            "global_output_layer_parameters": self._output_layer_parameters(global_parameters),
        }
        updates_dir = os.path.join(self._schemas_dir(), "round_updates")
        os.makedirs(updates_dir, exist_ok=True)
        with open(os.path.join(updates_dir, f"round_{server_round}.json"), "w", encoding="utf-8") as f:
            json.dump(source, f, indent=2)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Basic malformed/non-finite update screening. This is not a Byzantine-
        # robust aggregation guarantee.
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
                print(f"[SCREEN] Dropping non-finite update from {client_proxy.cid}")
                continue

            # Conservative absolute-norm sanity screen; this is not a poisoning
            # defense and must not be presented as Byzantine fault tolerance.
            float_ndarrays = [
                arr for arr in ndarrays
                if np.issubdtype(np.asarray(arr).dtype, np.floating)
            ]
            total_l2_norm = np.sqrt(
                sum(np.sum(np.square(arr, dtype=np.float64)) for arr in float_ndarrays)
            )
            MAX_TOLERABLE_L2_NORM = 500.0  # Allow legitimate trained MLP weights up to 500.0
            if total_l2_norm > MAX_TOLERABLE_L2_NORM:
                print(f"[SCREEN] Dropping out-of-range update from {client_proxy.cid}: norm {total_l2_norm:.2f}")
                continue

            clean_results.append((client_proxy, fit_res))

        if not clean_results:
            print("[SECURITY] All client updates were rejected as malicious or invalid! Aborting round aggregation.")
            return None, {}

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, clean_results, failures)
        if parameters_aggregated is None:
            return None, metrics_aggregated

        ndarrays = parameters_to_ndarrays(parameters_aggregated)
        self.last_weights = [arr.tolist() for arr in ndarrays]
        self.last_client_count = len(clean_results)

        self._save_exact_round_updates(server_round, clean_results, ndarrays)

        # Validation is queried each round for model selection. The final test
        # set is kept out of this loop.
        validation_metrics = None
        if getattr(self, "evaluate_fn", None) is not None:
            try:
                validation_loss, validation_metrics = self.evaluate_fn(server_round, parameters_aggregated, {})
                current_f1 = validation_metrics.get("f1", 0.0)
                current_threshold = validation_metrics.get("decision_threshold", 0.5)
                print(f"[VALIDATION] Round {server_round}: F1={current_f1:.4f}, "
                      f"P={validation_metrics.get('precision', 0.0):.4f}, "
                      f"R={validation_metrics.get('recall', 0.0):.4f}, "
                      f"threshold={current_threshold:.6f}")
                if self.best_weights is None or current_f1 > self.best_f1 + self.min_delta:
                    self.best_f1 = current_f1
                    self.best_round = int(server_round)
                    self.best_threshold = float(current_threshold)
                    self.best_weights = [np.array(arr, copy=True) for arr in ndarrays]
            except Exception as exc:
                print(f"[WARN] Validation evaluation failed: {exc}")
        metrics_dict = validation_metrics or (metrics_aggregated if metrics_aggregated is not None else {})

        export_data = {
            "round": server_round,
            "run_id": _load_manifest().get("run_id"),
            "weights": self.last_weights,
            "metrics": {
                "accuracy": metrics_dict.get("accuracy", 0.0),
                "precision": metrics_dict.get("precision", 0.0),
                "recall": metrics_dict.get("recall", 0.0),
                "f1": metrics_dict.get("f1", 0.0),
                "auc": metrics_dict.get("auc", 0.0),
                "average_precision": metrics_dict.get("average_precision", 0.0),
                "tp": metrics_dict.get("tp"),
                "fp": metrics_dict.get("fp"),
                "fn": metrics_dict.get("fn"),
                "tn": metrics_dict.get("tn"),
                "positives": metrics_dict.get("positives"),
                "negatives": metrics_dict.get("negatives"),
                "support": metrics_dict.get("support"),
                "prevalence": metrics_dict.get("prevalence"),
                "confusion_matrix": metrics_dict.get("confusion_matrix"),
                "decision_threshold": metrics_dict.get("decision_threshold"),
                "threshold_selection": "validation_max_f1",
            },
            "evaluation": metrics_dict.get("split", "validation"),
            "client_fit_summary": metrics_aggregated or {},
            "selected_by_validation": bool(server_round == self.best_round),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "client_count": getattr(self, "last_client_count", 4),
        }

        # Tests may redirect round exports so they never overwrite the canonical
        # leak-free run artifacts in schemas/.
        schemas_dir = self._schemas_dir()
        os.makedirs(schemas_dir, exist_ok=True)
        export_path = os.path.join(schemas_dir, f"global_model_round_{server_round}.json")
        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"[SAVE] Exported global model for round {server_round} to {export_path}")

        # The final test is evaluated exactly once, on the best validation round.
        if server_round >= self.target_rounds and self.best_weights is not None and self.final_evaluate_fn:
            _, self.final_test_metrics = self.final_evaluate_fn(
                self.best_round, self.best_weights,
                {"decision_threshold": self.best_threshold},
            )
            final_record = {
                "run_id": _load_manifest().get("run_id"),
                "selected_round": self.best_round,
                "selection_metric": "validation_f1",
                "best_validation_f1": float(self.best_f1),
                "decision_threshold": float(self.best_threshold),
                "threshold_selection": "validation_max_f1",
                "client_fit_summary": metrics_aggregated or {},
                "evaluation": "final_test_once",
                "metrics": self.final_test_metrics,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            with open(os.path.join(schemas_dir, "final_evaluation.json"), "w", encoding="utf-8") as f:
                json.dump(final_record, f, indent=2)

            final_model = GraphAwareMLP(input_dim=32)
            final_model.set_parameters(self.best_weights)
            checkpoint_path = os.environ.get("VFI_MODEL_PATH") or os.path.join(
                project_root, "fl_implementation", "model.pt"
            )
            torch.save(final_model.state_dict(), checkpoint_path)
            print(
                f"[FINAL TEST] Selected validation round {self.best_round} with "
                f"threshold={self.best_threshold:.6f}; metrics saved once."
            )

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        # Compute exact metric aggregation from client confusion matrices.  These
        # remain useful for monitoring, but they are NOT the reported paper
        # numbers: client test splits are smaller and the server holdout is the
        # canonical evaluation set.
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

                metrics_aggregated = {
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                }

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
    import random

    SEED = int(os.getenv("VFI_SEED", "42"))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    parser = argparse.ArgumentParser(description="VFI Federated Learning Server")
    parser.add_argument("--rounds", type=int, default=int(os.getenv("FL_ROUNDS", "10")), help="Number of FL rounds")
    parser.add_argument("--min-clients", type=int, default=int(os.getenv("MIN_CLIENTS", "4")), help="Min clients required")
    parser.add_argument("--port", type=int, default=8080, help="FL server port")
    parser.add_argument(
        "--validation-only", action="store_true",
        help="Train and select using validation metrics only; do not load or evaluate the final test set.",
    )
    args = parser.parse_args()

    print(f"Starting FL Server on port {args.port} (target rounds: {args.rounds}, min clients: {args.min_clients})...")

    manifest = _load_manifest()
    if manifest.get("pipeline_version") != "causal-v3" or manifest.get("status") != "complete":
        raise RuntimeError(
            "Preprocessing artifacts are missing or stale. Run data/preprocess.py "
            "to create causal-v3 training, validation, and final-test splits first."
        )
    validation_csv_path = os.path.join(project_root, "fl_implementation", "data", "global_validation.csv")
    test_csv_path = os.path.join(project_root, "fl_implementation", "data", "global_test.csv")
    validation_eval_fn = get_eval_fn(validation_csv_path, "validation")
    final_test_eval_fn = None if args.validation_only else get_eval_fn(test_csv_path, "final_test_once")
    if validation_eval_fn is None or (not args.validation_only and final_test_eval_fn is None):
        raise FileNotFoundError("Required validation/final-test CSV is missing; rerun preprocessing.")

    strategy = EarlyStoppingFedAvg(
        target_rounds=args.rounds,
        min_delta=0.0002,
        evaluate_fn=validation_eval_fn,
        final_evaluate_fn=final_test_eval_fn,
        fraction_evaluate=0.0,
        evaluate_metrics_aggregation_fn=exact_metric_aggregation,
        fit_metrics_aggregation_fn=aggregate_privacy_metrics,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=0,
        min_available_clients=args.min_clients,
    )

    if args.validation_only:
        print("[EXPERIMENT] Validation-only mode: the final-test CSV will not be evaluated.")

    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
