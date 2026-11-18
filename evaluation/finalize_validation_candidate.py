"""Evaluate one frozen validation-selected checkpoint on final_test_once."""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ablation_logging import append_ablation_result
from fl_implementation.server import get_eval_fn
from models.mlp import GraphAwareMLP


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _matches(candidate, loss_type, positive_weight, resampling, smote_strategy):
    config = candidate
    if config["loss_type"] != loss_type or config["resampling"] != resampling:
        return False
    if float(config["positive_loss_weight"]) != positive_weight:
        return False
    return (
        resampling != "smote_tomek"
        or float(config["smote_strategy"]) == smote_strategy
    )


def finalize_candidate(sweep_file, loss_type, positive_weight, resampling,
                       smote_strategy, output_dir):
    sweep_file = Path(sweep_file).resolve()
    report = _read_json(sweep_file)
    manifest = _read_json(ROOT / "data" / "preprocess_manifest.json")
    if report.get("run_id") != manifest.get("run_id"):
        raise ValueError("Sweep and preprocessing manifest run IDs do not match.")

    matches = [
        candidate for candidate in report.get("candidates", [])
        if _matches(
            candidate, loss_type, positive_weight, resampling, smote_strategy
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one validation-selected candidate, found {len(matches)}."
        )
    candidate = matches[0]
    validation = candidate["best_validation"]
    selected_round = int(validation["round"])
    artifact_dir = Path(candidate["artifact_dir"])
    if not artifact_dir.is_absolute():
        artifact_dir = (ROOT / artifact_dir).resolve()
    expected_sweep_dir = f"{manifest['run_id']}_{report['sweep_id']}"
    if artifact_dir.parent.name != expected_sweep_dir:
        raise ValueError("Selected candidate artifacts do not belong to this sweep.")
    round_record = _read_json(artifact_dir / f"global_model_round_{selected_round}.json")
    candidate_tag = artifact_dir.name
    checkpoint = artifact_dir / "best_model.pt"
    round_update = artifact_dir / "round_updates" / f"round_{selected_round}.json"
    if not checkpoint.is_file() or not round_update.is_file():
        raise FileNotFoundError(
            f"Selected checkpoint or exact round updates are missing: {checkpoint}, {round_update}"
        )

    if round_record.get("evaluation") != "validation":
        raise ValueError("Selected round artifact is not marked as validation.")
    if round_record.get("run_id") != manifest["run_id"]:
        raise ValueError("Selected round artifact belongs to a different data run.")
    if not round_record.get("selected_by_validation"):
        raise ValueError("Selected round is not the server's validation-selected checkpoint.")
    for metric in (
        "f1", "average_precision", "auc", "decision_threshold",
        "calibrated_threshold", "temperature", "tp", "fp", "fn", "tn",
    ):
        if round_record["metrics"].get(metric) != validation.get(metric):
            raise ValueError(
                f"Selected round and sweep report disagree on validation {metric}."
            )
    source_validation_run_id = f"{report['sweep_id']}_{candidate_tag}"
    if round_update.exists():
        update_record = _read_json(round_update)
        if update_record.get("round") != selected_round:
            raise ValueError("Selected round update artifact has the wrong round number.")
        if len(update_record.get("global_output_layer_parameters", [])) != 33:
            raise ValueError("Selected round update does not contain 33 output parameters.")

    config = candidate
    model = GraphAwareMLP(
        input_dim=32,
        dropout=float(config["model_dropout"]),
    )
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    round_model = GraphAwareMLP(
        input_dim=32,
        dropout=float(config["model_dropout"]),
    )
    round_model.set_parameters(
        [np.asarray(parameters) for parameters in round_record["weights"]]
    )
    if any(
        not torch.equal(model_state, round_state)
        for model_state, round_state in zip(
            model.state_dict().values(), round_model.state_dict().values()
        )
    ):
        raise ValueError(
            "Saved best_model.pt does not match the server-selected validation round."
        )

    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(ROOT / "evaluation"):
        raise ValueError("Finalized candidate outputs must be under evaluation/.")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing a second final-test evaluation for this candidate: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    shutil.copy2(checkpoint, output_dir / "best_model.pt")
    shutil.copy2(round_update, output_dir / f"round_{selected_round}.json")
    shutil.copy2(
        artifact_dir / f"global_model_round_{selected_round}.json",
        output_dir / f"validation_round_{selected_round}.json",
    )

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    attempt_path = output_dir / "final_test_attempt.json"
    _write_json(attempt_path, {
        "status": "in_progress",
        "started_at": timestamp,
        "source_validation_run_id": source_validation_run_id,
        "selected_round": selected_round,
        "evaluation": "final_test_once",
    })

    final_eval = get_eval_fn(
        str(ROOT / "fl_implementation" / "data" / "global_test.csv"),
        split_name="final_test_once",
    )
    try:
        _, final_metrics = final_eval(
            selected_round,
            model.get_parameters(),
            {
                "decision_threshold": validation["decision_threshold"],
                "calibrated_threshold": validation["calibrated_threshold"],
                "temperature": validation["temperature"],
            },
        )
    except Exception as exc:
        _write_json(attempt_path, {
            "status": "failed",
            "started_at": timestamp,
            "failed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_validation_run_id": source_validation_run_id,
            "selected_round": selected_round,
            "evaluation": "final_test_once",
            "error": str(exc),
        })
        raise

    expected_support = int(manifest["raw_test_rows"])
    if int(final_metrics["support"]) != expected_support:
        _write_json(attempt_path, {
            "status": "failed",
            "started_at": timestamp,
            "failed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_validation_run_id": source_validation_run_id,
            "selected_round": selected_round,
            "evaluation": "final_test_once",
            "error": (
                f"Expected {expected_support} final-test rows, "
                f"observed {final_metrics['support']}."
            ),
        })
        raise ValueError(
            f"Expected {expected_support} final-test rows, got {final_metrics['support']}."
        )

    client_summary = validation.get("client_fit_summary", {})
    hyperparameters = {
        "data_run_id": manifest["run_id"],
        "source_validation_run_id": source_validation_run_id,
        "rounds": report["screening_rounds"],
        "selected_round": selected_round,
        "loss_type": config["loss_type"],
        "pos_weight": config["positive_loss_weight"],
        "resampling": config["resampling"],
        "smote_strategy": config["smote_strategy"],
        "dp_enabled": config["dp_enabled"],
        "max_grad_norm": config["max_grad_norm"],
        "noise_multiplier": config["noise_multiplier"],
        "epsilon": client_summary.get("max_client_epsilon"),
        "delta": client_summary.get("delta"),
        "dp_accounting_validity": client_summary.get(
            "dp_accounting_validity"
        ),
        "dp_rng_mode": client_summary.get("dp_rng_mode"),
        "aggregation_strategy": (
            f"FedProx(mu={config['fedprox_mu']})"
            if config["fedprox_mu"] > 0 else "FedAvg"
        ),
        "learning_rate": config["learning_rate"],
        "local_epochs": report["local_epochs"],
        "model_dropout": config["model_dropout"],
        "focal_alpha": config["focal_alpha"],
        "focal_gamma": config["focal_gamma"],
        "temperature": validation["temperature"],
        "threshold_selection": validation["threshold_selection"],
        "final_test_evaluation_calls": 1,
    }
    final_record = {
        "run_id": f"finalized_{source_validation_run_id}",
        "source_validation_run_id": source_validation_run_id,
        "preprocess_run_id": manifest["run_id"],
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selected_round": selected_round,
        "evaluation": "final_test_once",
        "validation_metrics": validation,
        "final_test_metrics": final_metrics,
        "hyperparameters": hyperparameters,
    }
    _write_json(output_dir / "final_evaluation.json", final_record)
    _write_json(attempt_path, {
        "status": "completed",
        "started_at": timestamp,
        "completed_at": final_record["timestamp"],
        "source_validation_run_id": source_validation_run_id,
        "selected_round": selected_round,
        "evaluation": "final_test_once",
        "evaluation_calls": 1,
        "support": final_metrics["support"],
    })
    append_ablation_result(
        ROOT,
        run_id=final_record["run_id"],
        experiment_name=f"finalized_{loss_type}_{resampling}",
        timestamp=final_record["timestamp"],
        hyperparameters=hyperparameters,
        validation_metrics=validation,
        final_test_metrics=final_metrics,
    )
    return final_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-file", required=True)
    parser.add_argument("--loss-type", choices=("bce", "focal"), required=True)
    parser.add_argument("--pos-weight", type=float, required=True)
    parser.add_argument(
        "--resampling", choices=("none", "smote_tomek"), required=True
    )
    parser.add_argument("--smote-strategy", type=float, default=0.05)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    record = finalize_candidate(
        args.sweep_file,
        args.loss_type,
        args.pos_weight,
        args.resampling,
        args.smote_strategy,
        args.output_dir,
    )
    metrics = record["final_test_metrics"]
    print(
        "[FINAL TEST ONCE] {candidate}: F1={f1:.6f}, AP={ap:.6f}, ROC-AUC={auc:.6f}".format(
            candidate=record["run_id"],
            f1=metrics["f1"],
            ap=metrics["average_precision"],
            auc=metrics["auc"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
