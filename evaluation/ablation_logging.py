"""Append experiment results while keeping validation and final-test metrics separate."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


METRIC_FIELDS = (
    "accuracy", "precision", "recall", "f1", "auc", "average_precision",
    "tp", "fp", "fn", "tn", "decision_threshold", "calibrated_threshold",
    "temperature", "support", "positives", "negatives", "confusion_matrix",
)


def _metric_record(metrics):
    metrics = metrics or {}
    return {field: metrics.get(field) for field in METRIC_FIELDS}


def _git_commit(root):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _working_tree_dirty(root):
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def _write_summary(path, records):
    rows = [
        "# Ablation Results",
        "",
        "Validation metrics are used for tuning and selection. Final-test metrics are recorded only for explicitly finalized candidates.",
        "",
        "## Completed validation-only comparison",
        "",
        "| Run ID | Candidate | Selected round | Validation F1 | Validation PR-AUC (AP) | Epsilon |",
        "|---|---|---:|---:|---:|---:|",
    ]
    validation_runs = [
        record for record in records
        if record.get("status") == "completed"
        and record.get("validation_metrics", {}).get("f1") is not None
        and record.get("validation_metrics", {}).get("average_precision") is not None
        and record.get("final_test_metrics", {}).get("f1") is None
    ]
    for record in validation_runs:
        validation = record["validation_metrics"]
        hyperparameters = record["hyperparameters"]
        rows.append(
            "| {run_id} | {candidate} | {round} | {f1} | {ap} | {epsilon} |".format(
                run_id=record["run_id"],
                candidate=_candidate_label(hyperparameters),
                round=hyperparameters.get("selected_round", "—"),
                f1=_format(validation["f1"]),
                ap=_format(validation["average_precision"]),
                epsilon=_format(hyperparameters.get("epsilon")),
            )
        )
    rows.extend([
        "",
        "## Finalized candidate comparison",
        "",
        "| Run ID | Candidate | Validation F1 | Validation PR-AUC | Final F1 | Final PR-AUC | Final ROC-AUC | Precision | Recall | TP | FP | FN | TN | Raw threshold | Calibrated threshold | Epsilon | Delta | Accounting caveat |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    finalized_runs = [
        record for record in records
        if record.get("status") == "completed"
        and record.get("final_test_metrics", {}).get("f1") is not None
    ]
    for record in finalized_runs:
        validation = record["validation_metrics"]
        final = record["final_test_metrics"]
        hyperparameters = record["hyperparameters"]
        rows.append(
            "| {run_id} | {candidate} | {val_f1} | {val_ap} | {f1} | {ap} | {auc} | "
            "{precision} | {recall} | {tp} | {fp} | {fn} | {tn} | {raw_threshold} | "
            "{calibrated_threshold} | {epsilon} | {delta} | {caveat} |".format(
                run_id=record["run_id"],
                candidate=_candidate_label(hyperparameters),
                val_f1=_format(validation.get("f1")),
                val_ap=_format(validation.get("average_precision")),
                f1=_format(final.get("f1")),
                ap=_format(final.get("average_precision")),
                auc=_format(final.get("auc")),
                precision=_format(final.get("precision")),
                recall=_format(final.get("recall")),
                tp=final.get("tp", "—"),
                fp=final.get("fp", "—"),
                fn=final.get("fn", "—"),
                tn=final.get("tn", "—"),
                raw_threshold=_format(final.get("decision_threshold")),
                calibrated_threshold=_format(final.get("calibrated_threshold")),
                epsilon=_format(hyperparameters.get("epsilon")),
                delta=_format(hyperparameters.get("delta")),
                caveat=hyperparameters.get("dp_accounting_validity", "unspecified"),
            )
        )
    rows.extend([
        "",
        "## Full run history",
        "",
        "| Run ID | Status | Experiment | Selected round | Validation F1 | Validation AP | Final-test F1 | Final-test AP | Epsilon |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for record in records:
        validation = record["validation_metrics"]
        final = record["final_test_metrics"]
        epsilon = record["hyperparameters"].get("epsilon")
        rows.append(
            "| {run_id} | {status} | {experiment} | {round} | {val_f1} | {val_ap} | "
            "{test_f1} | {test_ap} | {epsilon} |".format(
                run_id=record["run_id"],
                status=record["status"],
                experiment=record["experiment_name"],
                round=record["hyperparameters"].get("selected_round", "—"),
                val_f1=_format(validation["f1"]),
                val_ap=_format(validation["average_precision"]),
                test_f1=_format(final["f1"]),
                test_ap=_format(final["average_precision"]),
                epsilon=_format(epsilon),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _candidate_label(hyperparameters):
    loss_type = hyperparameters.get("loss_type", "bce")
    resampling = hyperparameters.get("resampling", "none")
    if loss_type == "focal":
        return "Focal (alpha={0}, gamma={1})".format(
            _format(hyperparameters.get("focal_alpha")),
            _format(hyperparameters.get("focal_gamma")),
        )
    if resampling == "smote_tomek":
        return "BCE + SMOTE-Tomek ({0})".format(
            _format(hyperparameters.get("smote_strategy"))
        )
    try:
        positive_weight = float(hyperparameters.get("pos_weight", 1.0))
    except (TypeError, ValueError):
        positive_weight = 1.0
    if positive_weight != 1.0:
        return "Cost-sensitive BCE (pos_weight={0})".format(_format(positive_weight))
    return "Baseline BCE"


def _format(value):
    return "—" if value is None else f"{float(value):.6g}"


def append_ablation_result(root, *, run_id, experiment_name, hyperparameters,
                           validation_metrics, final_test_metrics=None,
                           timestamp=None, status="completed", error=None):
    """Append one completed screening/finalization record and refresh its summary."""
    root = Path(root)
    results_path = root / "evaluation" / "ablation_results.json"
    summary_path = root / "evaluation" / "ablation_summary.md"
    if results_path.exists():
        with open(results_path, encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"{results_path} must contain a JSON array.")
    else:
        records = []

    record = {
        "run_id": str(run_id),
        "experiment_name": str(experiment_name),
        "status": str(status),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": _git_commit(root),
        "working_tree_dirty": _working_tree_dirty(root),
        "error": error,
        "hyperparameters": dict(hyperparameters),
        "validation_metrics": _metric_record(validation_metrics),
        "final_test_metrics": _metric_record(final_test_metrics),
    }
    records.append(record)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = results_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(results_path)
    _write_summary(summary_path, records)
    return record


def refresh_ablation_summary(root):
    """Rebuild the Markdown view from the append-only JSON result records."""
    root = Path(root)
    results_path = root / "evaluation" / "ablation_results.json"
    summary_path = root / "evaluation" / "ablation_summary.md"
    with open(results_path, encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{results_path} must contain a JSON array.")
    _write_summary(summary_path, records)
