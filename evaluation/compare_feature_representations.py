"""Compare raw causal feature scaling alternatives against the saved PCA folds."""

import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.graph_features import CausalTransactionFeatureExtractor
from data.preprocess import (
    chronological_train_validation_test_split,
    metadata_timestamps_to_seconds,
    split_features_and_labels,
)
from evaluation.run_fl_validation_sweep import PROJECT_VENV_PYTHON

CLIENT = ROOT / "fl_implementation" / "client.py"
FEATURE_REPRESENTATIONS = {
    "standardized_raw": "Train-fitted StandardScaler on the 32 causal features; no PCA.",
    "robust_raw": "Train-fitted RobustScaler on the 32 causal features; no PCA.",
}
SERVER_LAUNCHER = r"""
import json
import os
import random
import sys

import flwr as fl
import numpy as np
import torch

project_root = os.environ["VFI_PROJECT_ROOT"]
sys.path.insert(0, project_root)
from fl_implementation import server

with open(os.environ["VFI_MANIFEST_PATH"], encoding="utf-8") as handle:
    manifest = json.load(handle)
server._load_manifest = lambda: manifest

seed = int(os.environ["VFI_SEED"])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

validation = server.get_eval_fn(os.environ["VFI_VALIDATION_CSV"], "validation")
outer_test = server.get_eval_fn(os.environ["VFI_TEST_CSV"], "final_test_once")
if validation is None or outer_test is None:
    raise FileNotFoundError("The feature-comparison validation or outer-test CSV is missing.")

rounds = int(os.environ["FL_ROUNDS"])
strategy = server.EarlyStoppingFedAvg(
    target_rounds=rounds,
    final_evaluate_fn=outer_test,
    min_delta=0.0002,
    evaluate_fn=validation,
    fraction_evaluate=0.0,
    evaluate_metrics_aggregation_fn=server.exact_metric_aggregation,
    fit_metrics_aggregation_fn=server.aggregate_privacy_metrics,
    min_fit_clients=4,
    min_evaluate_clients=0,
    min_available_clients=4,
)
fl.server.start_server(
    server_address=f"0.0.0.0:{int(os.environ['VFI_PORT'])}",
    config=fl.server.ServerConfig(num_rounds=rounds),
    strategy=strategy,
)
"""


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary_path, path)


def load_raw_transactions(input_path, chunksize=50_000):
    header = pd.read_csv(input_path, nrows=0)
    time_source = next(
        (column for column in ("timestamp", "metadata", "step") if column in header.columns),
        None,
    )
    if time_source is None:
        raise ValueError("Raw data needs a timestamp, metadata datetime, or step column.")

    wanted = [
        time_source, "amount", "nameOrig", "nameDest", "isMoneyLaundering",
        "oldbalanceOrg", "newbalanceOrig", "hour", "day_of_week", "day_of_month", "month",
    ]
    usecols = [column for column in wanted if column in header.columns]
    chunks = []
    if time_source == "metadata":
        for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=chunksize):
            chunk["timestamp"] = metadata_timestamps_to_seconds(chunk.pop("metadata"))
            chunks.append(chunk)
        if not chunks:
            raise ValueError("Raw input contains no transaction rows.")
        frame = pd.concat(chunks, ignore_index=True)
    else:
        frame = pd.read_csv(input_path, usecols=usecols)

    frame.rename(
        columns={
            "nameOrig": "sender",
            "nameDest": "receiver",
            "isMoneyLaundering": "is_laundering",
            **({"step": "timestamp"} if time_source == "step" else {}),
        },
        inplace=True,
    )
    frame.drop(
        columns=[
            column for column in
            ("isFraud", "fraud_probability", "laundering_typology", "isFlaggedFraud")
            if column in frame.columns
        ],
        inplace=True,
    )
    frame["txn_id"] = np.arange(len(frame), dtype=np.int64)
    if "is_laundering" not in frame.columns:
        raise ValueError("Raw input does not contain the laundering target.")
    return frame, time_source


def transform_feature_splits(representation, X_train, X_validation, X_outer_test):
    if representation == "standardized_raw":
        transformer = StandardScaler()
    elif representation == "robust_raw":
        transformer = RobustScaler()
    else:
        raise ValueError(f"Unknown feature representation: {representation}")

    train = transformer.fit_transform(X_train)
    validation = transformer.transform(X_validation)
    outer_test = transformer.transform(X_outer_test)
    return train, validation, outer_test


def make_representation_data(output_dir, representation, feature_names,
                             arrays, labels, raw_counts, timestamp_source):
    output_dir = Path(output_dir)
    data_dir = output_dir / "fl_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_validation, X_outer_test = arrays
    y_train, y_validation, y_outer_test = labels

    train_frame = pd.DataFrame(X_train, columns=feature_names)
    train_frame["is_laundering"] = np.asarray(y_train, dtype=np.int64)
    train_frame = train_frame.sample(frac=1, random_state=42).reset_index(drop=True)
    validation_frame = pd.DataFrame(X_validation, columns=feature_names)
    validation_frame["is_laundering"] = np.asarray(y_validation, dtype=np.int64)
    outer_test_frame = pd.DataFrame(X_outer_test, columns=feature_names)
    outer_test_frame["is_laundering"] = np.asarray(y_outer_test, dtype=np.int64)

    validation_frame.to_csv(data_dir / "global_validation.csv", index=False)
    outer_test_frame.to_csv(data_dir / "global_test.csv", index=False)
    for client_id, shard in enumerate(np.array_split(train_frame, 4), start=1):
        shard.to_csv(data_dir / f"bank_{client_id}.csv", index=False)

    run_id = uuid.uuid4().hex
    manifest = {
        "pipeline_version": "causal-v3",
        "run_id": run_id,
        "status": "complete",
        "split_strategy": "saved_rolling_origin_prefix_whole_timestamp_groups",
        "timestamp_source": timestamp_source,
        "feature_representation": representation,
        "feature_columns": list(feature_names),
        "graph_feature_columns": list(feature_names),
        "raw_rows": int(sum(raw_counts.values())),
        "raw_train_rows": int(raw_counts["train"]),
        "raw_validation_rows": int(raw_counts["validation"]),
        "raw_test_rows": int(raw_counts["outer_test"]),
        "train_positives": int(np.asarray(y_train).sum()),
        "validation_positives": int(np.asarray(y_validation).sum()),
        "test_positives": int(np.asarray(y_outer_test).sum()),
        "resampling": "none",
    }
    manifest_path = output_dir / "preprocess_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return data_dir, manifest_path, manifest


def wait_for_server(process, log_path, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(
                f"Feature-comparison server exited before listening "
                f"(code {process.returncode}):\n{tail}"
            )
        if log_path.exists() and "gRPC server running" in log_path.read_text(
            encoding="utf-8", errors="replace"
        ):
            return
        time.sleep(0.5)
    raise TimeoutError(f"Feature-comparison server did not start within {timeout}s.")


def stop_processes(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def run_federated_trial(args, run_config, representation, fold, data_dir,
                        manifest_path, run_id, port):
    trial_dir = Path(data_dir).parent
    schemas_dir = trial_dir / "schemas"
    schemas_dir.mkdir(parents=True, exist_ok=True)
    launcher_path = trial_dir / "server_launcher.py"
    launcher_path.write_text(SERVER_LAUNCHER, encoding="utf-8")
    server_log_path = schemas_dir / "server.log"
    python_executable = args.python_executable

    env = os.environ.copy()
    env.update({
        "VFI_PROJECT_ROOT": str(ROOT),
        "VFI_MANIFEST_PATH": str(manifest_path),
        "VFI_VALIDATION_CSV": str(data_dir / "global_validation.csv"),
        "VFI_TEST_CSV": str(data_dir / "global_test.csv"),
        "VFI_SCHEMAS_DIR": str(schemas_dir),
        "VFI_MODEL_PATH": str(trial_dir / "model.pt"),
        "VFI_PRIVACY_STATE_DIR": str(trial_dir / "privacy"),
        "VFI_SEED": str(run_config["seed"]),
        "VFI_PORT": str(port),
        "FL_ROUNDS": str(run_config["rounds"]),
        "POSITIVE_LOSS_WEIGHT": str(run_config["positive_loss_weight"]),
        "ENABLE_DP": "true",
        "DP_NOISE_MULTIPLIER": str(run_config["noise_multiplier"]),
        "CLIENT_LEARNING_RATE": str(run_config["learning_rate"]),
        "LOCAL_EPOCHS": str(run_config["local_epochs"]),
        "FEDPROX_MU": str(run_config["fedprox_mu"]),
        "DP_ACCOUNTANT_TAG": f"features_{representation}_fold{fold}_{run_id[:8]}",
        "SERVER_ADDRESS": f"localhost:{port}",
    })
    # Keep the original project importable when the launcher itself is in temp.
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    processes = []
    handles = []
    try:
        server_log = open(server_log_path, "w", encoding="utf-8")
        handles.append(server_log)
        server = subprocess.Popen(
            [python_executable, str(launcher_path)],
            cwd=ROOT,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        processes.append(server)
        wait_for_server(server, server_log_path)

        for client_id in range(1, 5):
            client_log = open(schemas_dir / f"client_{client_id}.log", "w", encoding="utf-8")
            handles.append(client_log)
            client_env = {
                **env,
                "CLIENT_ID": str(client_id),
                "DATA_FILE": str(data_dir / f"bank_{client_id}.csv"),
            }
            client = subprocess.Popen(
                [python_executable, str(CLIENT), "--cid", str(client_id)],
                cwd=ROOT,
                env=client_env,
                stdout=client_log,
                stderr=subprocess.STDOUT,
            )
            processes.append(client)

        server_code = server.wait(timeout=4 * 60 * 60)
        if server_code != 0:
            tail = server_log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
            raise RuntimeError(f"Flower server exited with code {server_code}:\n{tail}")
        client_codes = [client.wait(timeout=120) for client in processes[1:]]
        if any(code != 0 for code in client_codes):
            raise RuntimeError(f"Flower clients exited unsuccessfully: {client_codes}.")
    except Exception:
        stop_processes(processes)
        raise
    finally:
        for handle in handles:
            handle.close()

    evaluation = read_json(schemas_dir / "final_evaluation.json")
    if evaluation.get("run_id") != run_id or evaluation.get("evaluation") != "final_test_once":
        raise RuntimeError("The trial did not produce its expected isolated outer evaluation.")
    return evaluation


def build_feature_splits(frame, cutoff, validation_size, outer_test_size):
    prefix = frame.loc[frame["timestamp"] <= cutoff].copy()
    train_raw, validation_raw, outer_test_raw = chronological_train_validation_test_split(
        prefix,
        validation_size=validation_size,
        test_size=outer_test_size,
    )

    extractor = CausalTransactionFeatureExtractor()
    train_features = extractor.fit_transform(train_raw)
    evaluation_extractor = copy.deepcopy(extractor)
    validation_features = evaluation_extractor.transform(
        validation_raw.drop(columns=["is_laundering"])
    )
    outer_test_features = evaluation_extractor.transform(
        outer_test_raw.drop(columns=["is_laundering"])
    )
    X_train, y_train = split_features_and_labels(train_features, train_raw)
    X_validation, y_validation = split_features_and_labels(
        validation_features, validation_raw
    )
    X_outer_test, y_outer_test = split_features_and_labels(
        outer_test_features, outer_test_raw
    )
    X_validation = X_validation.reindex(columns=X_train.columns, fill_value=0)
    X_outer_test = X_outer_test.reindex(columns=X_train.columns, fill_value=0)

    raw_counts = {
        "train": len(train_raw),
        "validation": len(validation_raw),
        "outer_test": len(outer_test_raw),
    }
    labels = (y_train, y_validation, y_outer_test)
    return X_train, X_validation, X_outer_test, labels, raw_counts


def confusion_summary(folds):
    totals = {
        key: sum(int(row["metrics"][key]) for row in folds)
        for key in ("tp", "fp", "fn", "tn")
    }
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**totals, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=ROOT / "evaluation" / "rolling_origin_cv_20261117T172715Z.json",
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--python-executable",
        default=str(PROJECT_VENV_PYTHON if PROJECT_VENV_PYTHON.exists() else Path(sys.executable)),
    )
    parser.add_argument("--port", type=int, default=8180)
    args = parser.parse_args()

    if not args.baseline_report.is_file():
        raise FileNotFoundError(f"Saved PCA rolling-origin baseline not found: {args.baseline_report}")
    baseline = read_json(args.baseline_report)
    run_config = baseline["model_configuration"]
    if (
        not baseline.get("original_final_test_excluded")
        or not run_config.get("dp_enabled")
        or run_config["noise_multiplier"] != 0.5
        or run_config["positive_loss_weight"] != 10.0
    ):
        raise ValueError("Baseline report does not match the required pre-final-test DP protocol.")
    input_path = args.input or Path(baseline["raw_input"])
    if not input_path.is_file():
        raise FileNotFoundError(f"Raw transaction data not found: {input_path}")

    current_manifest = read_json(ROOT / "data" / "preprocess_manifest.json")
    expected_features = current_manifest["graph_feature_columns"]
    if len(expected_features) != 32 or current_manifest["pca_components"] != 32:
        raise ValueError("This comparison expects the saved 32-feature, 32-component pipeline.")

    raw_frame, timestamp_source = load_raw_transactions(input_path)
    if len(raw_frame) != current_manifest["raw_rows"]:
        raise ValueError("Raw input row count does not match the current preprocessing manifest.")
    if timestamp_source != current_manifest["timestamp_source"]:
        raise ValueError("Raw timestamp source differs from the current preprocessing manifest.")

    output_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = ROOT / "evaluation" / f"feature_representation_comparison_{output_stamp}.json"
    report = {
        "status": "in_progress",
        "evaluation": "paired_rolling_origin_feature_representation_comparison",
        "baseline_report": str(args.baseline_report.resolve()),
        "raw_input": str(input_path.resolve()),
        "original_final_test_excluded": True,
        "fresh_blind_holdout": False,
        "note": (
            "Each alternative uses the same saved rolling-origin windows and fixed DP-FL "
            "configuration as the PCA baseline. Transformations are fitted on that fold's "
            "training rows only; thresholds and rounds are selected on its inner validation. "
            "The historical windows are development data, not a fresh blind estimate."
        ),
        "model_configuration": run_config,
        "representations": {
            "pca32_baseline": {
                "description": (
                    "Saved baseline: StandardScaler, full 32-component PCA, then a "
                    "training-fitted post-PCA StandardScaler."
                ),
                "folds": [],
            }
        },
        "pca_diagnostic": {
            "input_feature_count": len(expected_features),
            "pca_component_count": current_manifest["pca_components"],
            "interpretation": (
                "PCA retains all 32 input dimensions; it is a rotation/scaling "
                "representation, not dimensionality reduction."
            ),
        },
    }
    baseline_by_fold = {int(row["fold"]): row for row in baseline["folds"]}
    for fold_id in sorted(baseline_by_fold):
        row = baseline_by_fold[fold_id]
        report["representations"]["pca32_baseline"]["folds"].append({
            "fold": fold_id,
            "prefix_end_fraction": row["prefix_end_fraction"],
            "train_rows": row["train_rows"],
            "inner_validation_rows": row["inner_validation_rows"],
            "outer_test_rows": row["outer_test_rows"],
            "inner_validation_positives": row["inner_validation_positives"],
            "outer_test_positives": row["outer_test_positives"],
            "selected_round": row["selected_round"],
            "best_inner_validation_f1": row["best_inner_validation_f1"],
            "decision_threshold": row["decision_threshold"],
            "metrics": row["metrics"],
        })
    write_json_atomic(output_path, report)

    for representation, description in FEATURE_REPRESENTATIONS.items():
        report["representations"][representation] = {
            "description": description,
            "folds": [],
        }
    representation_items = list(FEATURE_REPRESENTATIONS.items())
    for fold_offset, fold_id in enumerate(sorted(baseline_by_fold)):
        baseline_fold = baseline_by_fold[fold_id]
        cutoff = baseline_fold["prefix_max_timestamp"]
        print(f"[PREPROCESS] Building causal feature matrices for fold {fold_id}.", flush=True)
        X_train, X_validation, X_outer_test, labels, raw_counts = build_feature_splits(
            raw_frame,
            cutoff,
            validation_size=0.1,
            outer_test_size=0.1,
        )
        expected_counts = {
            "train": baseline_fold["train_rows"],
            "validation": baseline_fold["inner_validation_rows"],
            "outer_test": baseline_fold["outer_test_rows"],
        }
        if raw_counts != expected_counts:
            raise ValueError(
                f"Fold {fold_id} split counts differ from the saved PCA baseline: "
                f"{raw_counts} != {expected_counts}"
            )
        if list(X_train.columns) != expected_features:
            raise ValueError("Reconstructed causal feature columns differ from the saved manifest.")
        if (
            int(np.asarray(labels[1]).sum()) != baseline_fold["inner_validation_positives"]
            or int(np.asarray(labels[2]).sum()) != baseline_fold["outer_test_positives"]
        ):
            raise ValueError(f"Fold {fold_id} class counts differ from the saved PCA baseline.")

        for representation_offset, representation in enumerate(FEATURE_REPRESENTATIONS):
            transformed = transform_feature_splits(
                representation,
                X_train,
                X_validation,
                X_outer_test,
            )
            raw_feature_names = list(X_train.columns)
            fold_temp = tempfile.TemporaryDirectory(
                prefix=f"vfi_features_{representation}_fold{fold_id}_"
            )
            with fold_temp as temp_dir:
                data_dir, manifest_path, manifest = make_representation_data(
                    temp_dir,
                    representation,
                    raw_feature_names,
                    transformed,
                    labels,
                    raw_counts,
                    timestamp_source,
                )
                evaluation = run_federated_trial(
                    args,
                    run_config,
                    representation,
                    fold_id,
                    data_dir,
                    manifest_path,
                    manifest["run_id"],
                    args.port + fold_offset * len(representation_items) + representation_offset,
                )
                metrics = evaluation["metrics"]
                baseline_metrics = baseline_fold["metrics"]
                result = {
                    "fold": fold_id,
                    "prefix_end_fraction": baseline_fold["prefix_end_fraction"],
                    "train_rows": raw_counts["train"],
                    "inner_validation_rows": raw_counts["validation"],
                    "outer_test_rows": raw_counts["outer_test"],
                    "inner_validation_positives": int(np.asarray(labels[1]).sum()),
                    "outer_test_positives": int(np.asarray(labels[2]).sum()),
                    "selected_round": evaluation["selected_round"],
                    "best_inner_validation_f1": evaluation["best_validation_f1"],
                    "decision_threshold": evaluation["decision_threshold"],
                    "max_client_epsilon": evaluation.get(
                        "client_fit_summary", {}
                    ).get("max_client_epsilon"),
                    "metrics": metrics,
                    "delta_f1_vs_pca": float(metrics["f1"] - baseline_metrics["f1"]),
                    "delta_average_precision_vs_pca": float(
                        metrics["average_precision"] - baseline_metrics["average_precision"]
                    ),
                }
                report["representations"][representation]["folds"].append(result)
                print(
                    f"[{representation} fold {fold_id}] "
                    f"outer F1={metrics['f1']:.4f}, "
                    f"AP={metrics['average_precision']:.4f}, "
                    f"delta F1={result['delta_f1_vs_pca']:+.4f}",
                    flush=True,
                )
            write_json_atomic(output_path, report)

    for representation, summary in report["representations"].items():
        folds = summary["folds"]
        f1_values = [float(row["metrics"]["f1"]) for row in folds]
        ap_values = [float(row["metrics"]["average_precision"]) for row in folds]
        summary["mean_outer_f1"] = statistics.mean(f1_values)
        summary["sample_sd_outer_f1"] = statistics.stdev(f1_values) if len(f1_values) > 1 else 0.0
        summary["mean_outer_average_precision"] = statistics.mean(ap_values)
        summary["pooled_outer_confusion_metrics"] = confusion_summary(folds)
        if representation != "pca32_baseline":
            summary["mean_delta_f1_vs_pca"] = statistics.mean(
                row["delta_f1_vs_pca"] for row in folds
            )
            summary["mean_delta_average_precision_vs_pca"] = statistics.mean(
                row["delta_average_precision_vs_pca"] for row in folds
            )

    report["status"] = "complete"
    write_json_atomic(output_path, report)
    print(f"[SAVE] Feature-representation comparison: {output_path}", flush=True)
    for representation, summary in report["representations"].items():
        print(
            f"[SUMMARY] {representation}: mean outer F1={summary['mean_outer_f1']:.4f}, "
            f"mean AP={summary['mean_outer_average_precision']:.4f}",
            flush=True,
        )
    for representation in FEATURE_REPRESENTATIONS:
        summary = report["representations"][representation]
        print(
            f"[DELTA] {representation} vs PCA32: mean F1 "
            f"{summary['mean_delta_f1_vs_pca']:+.4f}, mean AP "
            f"{summary['mean_delta_average_precision_vs_pca']:+.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
