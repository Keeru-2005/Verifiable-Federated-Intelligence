import os
import sys
import json
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score

# Add project root directory to path for models import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.mlp import GraphAwareMLP
from evaluation.ablation_logging import append_ablation_result
from evaluation.threshold_metrics import (
    binary_classification_metrics,
    fit_temperature,
    select_smoothed_thresholds,
)

LABEL_COL = "is_laundering"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed_partition_best.csv")
VALIDATION_PATH = os.path.join(PROJECT_ROOT, "fl_implementation", "data", "global_validation.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "fl_implementation", "data", "global_test.csv")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "preprocess_manifest.json")
SCHEMAS_DIR = os.path.join(PROJECT_ROOT, "schemas")
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, "visualizations")

def _feature_frame(df):
    """Extract the numeric feature matrix and never the label."""
    X_df = df.drop(columns=['is_laundering', 'sender', 'receiver', 'timestamp', 'nameOrig', 'nameDest', '_pk', '_edge_key'], errors='ignore')
    return X_df.select_dtypes(include=[np.number])


def load_datasets(train_path=DATA_PATH, validation_path=VALIDATION_PATH):
    """
    Load only training and validation rows for model/threshold selection.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Data file not found at {train_path}")
    if not os.path.exists(validation_path):
        raise FileNotFoundError("global_validation.csv is required.")

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)

    X_train_df = _feature_frame(train_df)
    X_validation_df = _feature_frame(validation_df).reindex(columns=X_train_df.columns, fill_value=0.0)

    X_train = X_train_df.values.astype(np.float32)
    y_train = train_df[LABEL_COL].values.astype(np.float32)
    X_validation = X_validation_df.values.astype(np.float32)
    y_validation = validation_df[LABEL_COL].values.astype(np.float32)

    return (
        X_train, y_train, X_validation, y_validation,
        list(X_train_df.columns), X_train.shape[1],
    )

def evaluate(model, loader, criterion, device, decision_threshold=None,
             calibrated_threshold=None, temperature=None):
    """Evaluate using validation-fitted temperature and thresholds when omitted."""
    model.eval()
    y_true, y_logits = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_true.extend(y_batch.cpu().numpy())
            y_logits.extend(model.forward_logits(X_batch).reshape(-1).cpu().numpy())

    y_true = np.asarray(y_true).astype(int)
    logits = np.asarray(y_logits, dtype=np.float64)
    raw_scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    raw_thresholds = select_smoothed_thresholds(y_true, raw_scores)
    if temperature is None:
        temperature = fit_temperature(logits, y_true)
    calibrated_scores = 1.0 / (
        1.0 + np.exp(-np.clip(logits / temperature, -80.0, 80.0))
    )
    calibrated_thresholds = select_smoothed_thresholds(y_true, calibrated_scores)
    if decision_threshold is None:
        decision_threshold = raw_thresholds["f1_threshold"]
    if calibrated_threshold is None:
        calibrated_threshold = calibrated_thresholds["f1_threshold"]
    metrics = binary_classification_metrics(
        y_true, calibrated_scores, calibrated_threshold
    )
    metrics["decision_threshold"] = float(decision_threshold)
    metrics["calibrated_threshold"] = float(calibrated_threshold)
    metrics["temperature"] = float(temperature)
    metrics["precision_at_least_0_70_threshold"] = calibrated_thresholds.get(
        "precision_threshold"
    )
    metrics["precision_at_least_0_70"] = calibrated_thresholds.get(
        "precision_threshold_precision"
    )
    metrics["recall_at_precision_at_least_0_70"] = calibrated_thresholds.get(
        "precision_threshold_recall"
    )
    metrics["threshold_selection"] = "validation_smoothed_f1_3_point"
    try:
        metrics["auc"] = float(roc_auc_score(y_true, calibrated_scores))
        metrics["average_precision"] = float(average_precision_score(y_true, calibrated_scores))
    except ValueError:
        metrics["auc"] = 0.5
        metrics["average_precision"] = 0.0
    return metrics

def train_centralized(X_train, y_train, X_validation, y_validation, feature_columns,
                      input_dim, epochs=10, batch_size=256, lr=0.001,
                      test_path=TEST_PATH):
    """Select an epoch on validation, then evaluate the selected model once on test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        TensorDataset(torch.tensor(X_validation), torch.tensor(y_validation)),
        batch_size=batch_size, shuffle=False
    )
    model = GraphAwareMLP(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    train_criterion = nn.BCELoss(reduction="none")

    metrics_history = []
    best_validation_ap = -1.0
    best_threshold = 0.5
    best_calibrated_threshold = 0.5
    best_temperature = 1.0
    best_state = None
    best_epoch = None
    best_validation_metrics = None
    print(f"[Centralized] Training on {len(y_train)} natural rows; selecting by validation only.")

    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch).squeeze()
            if out.dim() == 0:
                out = out.unsqueeze(0)
            losses = train_criterion(out, y_batch)
            weights = torch.where(
                y_batch > 0.5,
                torch.full_like(y_batch, 10.0),
                torch.ones_like(y_batch),
            )
            loss = (losses * weights).mean()
            loss.backward()
            optimizer.step()

        metrics = evaluate(model, validation_loader, criterion, device)
        print(f"  Epoch {epoch:02d}/{epochs} | Validation F1: {metrics['f1']:.4f} | "
              f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} | "
              f"AP={metrics['average_precision']:.4f} | threshold={metrics['decision_threshold']:.6f}")
        metrics_history.append({"round": epoch, **metrics})
        if metrics["average_precision"] > best_validation_ap:
            best_validation_ap = metrics["average_precision"]
            best_threshold = metrics["decision_threshold"]
            best_calibrated_threshold = metrics["calibrated_threshold"]
            best_temperature = metrics["temperature"]
            best_epoch = epoch
            best_validation_metrics = dict(metrics)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Centralized baseline did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Final-test file not found at {test_path}")
    test_df = pd.read_csv(test_path)
    X_test_df = _feature_frame(test_df).reindex(columns=feature_columns, fill_value=0.0)
    y_test = test_df[LABEL_COL].to_numpy(dtype=np.float32)
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test_df.to_numpy(dtype=np.float32)),
            torch.tensor(y_test),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    final_test = evaluate(
        model,
        test_loader,
        criterion,
        device,
        decision_threshold=best_threshold,
        calibrated_threshold=best_calibrated_threshold,
        temperature=best_temperature,
    )
    final_test["selected_epoch"] = best_epoch
    final_test["evaluation"] = "final_test_once"
    return metrics_history, final_test, best_validation_metrics

def load_federated_metrics(run_id):
    # Search for global_model_round_*.json
    files = glob.glob(os.path.join(SCHEMAS_DIR, "global_model_round_*.json"))
    rounds_data = []
    
    for file_path in files:
        # Exclude schema templates (e.g. N.json)
        basename = os.path.basename(file_path)
        if "_N.json" in basename or not basename.replace("global_model_round_", "").replace(".json", "").isdigit():
            continue
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                if data.get("run_id") != run_id or data.get("evaluation") != "validation":
                    continue
                m = data["metrics"]
                entry = {
                    "round": int(data["round"]),
                    "accuracy": float(m["accuracy"]),
                    "f1": float(m["f1"]),
                }
                for key in ("precision", "recall", "auc", "average_precision",
                            "decision_threshold", "tp", "fp", "fn", "tn",
                            "positives", "negatives", "support", "confusion_matrix"):
                    if key in m:
                        entry[key] = m[key]
                rounds_data.append(entry)
        except Exception as e:
            print(f"[WARN] Error reading {file_path}: {e}")
            
    # Sort by round number
    rounds_data = sorted(rounds_data, key=lambda x: x["round"])
    return rounds_data

def generate_comparison_plots(centralized_history, federated_history, centralized_test, run_id):
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # Convert lists to DataFrames
    df_central = pd.DataFrame(centralized_history)
    df_central["Type"] = "Centralized Baseline (Pooled)"
    
    # If federated history is empty (e.g. training hasn't run yet), do not invent data.
    if not federated_history:
        print("[WARN] No federated metrics found in schemas/. Skipping the federated series instead of fabricating one.")
        return
        
    df_fed = pd.DataFrame(federated_history)
    df_fed["Type"] = "Federated Learning (Privacy)"
    
    df_all = pd.concat([df_central, df_fed], ignore_index=True)
    
    # Set style
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Title
    fig.suptitle("AML Detection Model Performance: Centralized vs. Federated Learning", fontsize=16, fontweight="bold", y=0.98)
    
    # F1-Score Chart
    sns.lineplot(
        ax=axes[0], data=df_all, x="round", y="f1", hue="Type",
        style="Type", markers=True, dashes=False, linewidth=2.5
    )
    axes[0].set_title("F1-Score Convergence Comparison", fontsize=13, fontweight="semibold")
    axes[0].set_xlabel("Epoch / FL Round", fontsize=11)
    axes[0].set_ylabel("F1-Score", fontsize=11)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(title="Mode")
    
    # Average precision is more informative than accuracy for this highly
    # imbalanced AML dataset, where an all-negative model can exceed 99% accuracy.
    sns.lineplot(
        ax=axes[1], data=df_all, x="round", y="average_precision", hue="Type",
        style="Type", markers=True, dashes=False, linewidth=2.5
    )
    axes[1].set_title("Average Precision Comparison", fontsize=13, fontweight="semibold")
    axes[1].set_xlabel("Epoch / FL Round", fontsize=11)
    axes[1].set_ylabel("Average Precision", fontsize=11)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(title="Mode")
    
    plt.tight_layout()
    plot_path = os.path.join(VISUALIZATION_DIR, "baseline_vs_federated.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[METRICS] Saved performance comparison chart to {plot_path}")
    
    # Save the json metrics comparison for the dashboard API
    results_path = os.path.join(PROJECT_ROOT, "evaluation", "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "run_id": run_id,
            # Compatibility aliases consumed by the dashboard comparison cards.
            "centralized": centralized_history,
            "federated": federated_history,
            "centralized_validation": centralized_history,
            "centralized_final_test_once": centralized_test,
            "federated_validation": federated_history,
        }, f, indent=2)
    print(f"[SAVE] Saved raw comparison results JSON to {results_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Centralized Baseline Training & Comparison")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--sample", type=int, default=None, help="Optional sample size for fast benchmarking")
    args = parser.parse_args()

    # Fixed seed so this baseline is reproducible on CPU.
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=== Centralized Baseline Model Training & Comparison ===")
    try:
        with open(MANIFEST_PATH, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        if manifest.get("pipeline_version") != "causal-v3" or manifest.get("status") != "complete":
            raise RuntimeError("Run causal-v3 preprocessing before creating a baseline.")
        run_id = manifest["run_id"]
        (
            X_train, y_train, X_validation, y_validation,
            feature_columns, input_dim,
        ) = load_datasets()
        if args.sample and len(X_train) > args.sample:
            idx = np.random.choice(len(X_train), args.sample, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
        print(f"Loaded natural train: {X_train.shape[0]} rows, {int(y_train.sum())} positives; "
              f"validation: {X_validation.shape[0]} rows. Final test remains unopened during selection.")
    except Exception as e:
        print(f"[FAIL] Error loading data: {e}")
        sys.exit(1)

    # Train centralized model on the natural training partition only
    centralized_history, centralized_test, centralized_validation = train_centralized(
        X_train, y_train, X_validation, y_validation, feature_columns,
        input_dim, epochs=args.epochs
    )
    append_ablation_result(
        PROJECT_ROOT,
        run_id=f"centralized_{run_id}_{centralized_test['selected_epoch']}",
        experiment_name="centralized_non_dp",
        hyperparameters={
            "data_run_id": run_id,
            "rounds": args.epochs,
            "selected_round": centralized_test["selected_epoch"],
            "loss_type": "bce",
            "pos_weight": 10.0,
            "resampling": "none",
            "dp_enabled": False,
            "max_grad_norm": None,
            "epsilon": 0.0,
            "delta": 0.0,
            "aggregation_strategy": "centralized",
            "temperature": centralized_validation["temperature"],
            "threshold_selection": centralized_validation["threshold_selection"],
        },
        validation_metrics=centralized_validation,
        final_test_metrics=centralized_test,
    )
    
    # Load federated metrics
    federated_history = load_federated_metrics(run_id)
    print(f"Loaded {len(federated_history)} rounds of federated learning metrics.")
    
    # Generate charts
    generate_comparison_plots(centralized_history, federated_history, centralized_test, run_id)
    print("=== Evaluation & Comparison Complete ===")

if __name__ == "__main__":
    main()
