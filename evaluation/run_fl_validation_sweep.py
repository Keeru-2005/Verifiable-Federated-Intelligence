"""Validation-only FedAvg configuration sweep for the local VFI proof of concept."""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ablation_logging import append_ablation_result


SERVER = ROOT / "fl_implementation" / "server.py"
CLIENT = ROOT / "fl_implementation" / "client.py"
MANIFEST = ROOT / "data" / "preprocess_manifest.json"
PROJECT_VENV_PYTHON = ROOT / "capstone_env" / "Scripts" / "python.exe"


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def wait_for_server(process, log_path, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Flower server exited before listening (code {process.returncode}).")
        if log_path.exists() and "gRPC server running" in log_path.read_text(
            encoding="utf-8", errors="replace"
        ):
            return
        time.sleep(0.5)
    raise TimeoutError(f"Flower server did not report readiness within {timeout}s.")


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


def run_fl(python_executable, rounds, seed, positive_weight, noise, dp_enabled,
           learning_rate, local_epochs, fedprox_mu, schemas_dir, accountant_tag,
           validation_only, port, data_dir=None, manifest_path=None,
           validation_csv=None, test_csv=None, model_path=None,
           loss_type="bce", resampling="none", max_grad_norm=1.0,
           model_dropout=0.1, smote_strategy=0.05, focal_alpha=0.75,
           focal_gamma=2.0, patience=5):
    env = os.environ.copy()
    env.update({
        "VFI_SEED": str(seed),
        "POSITIVE_LOSS_WEIGHT": str(positive_weight),
        "ENABLE_DP": "true" if dp_enabled else "false",
        "DP_NOISE_MULTIPLIER": str(noise),
        "DP_ACCOUNTANT_TAG": accountant_tag,
        "CLIENT_LEARNING_RATE": str(learning_rate),
        "LOCAL_EPOCHS": str(local_epochs),
        "FEDPROX_MU": str(fedprox_mu),
        "LOSS_TYPE": loss_type,
        "RESAMPLING": resampling,
        "DP_CLIP_NORM": str(max_grad_norm),
        "MODEL_DROPOUT": str(model_dropout),
        "SMOTE_STRATEGY": str(smote_strategy),
        "FOCAL_ALPHA": str(focal_alpha),
        "FOCAL_GAMMA": str(focal_gamma),
        "VFI_SCHEMAS_DIR": str(schemas_dir),
        "SERVER_ADDRESS": f"localhost:{port}",
    })
    if data_dir is not None:
        env["VFI_DATA_DIR"] = str(data_dir)
    if manifest_path is not None:
        env["VFI_MANIFEST_PATH"] = str(manifest_path)
    if validation_csv is not None:
        env["VFI_VALIDATION_CSV"] = str(validation_csv)
    if test_csv is not None:
        env["VFI_TEST_CSV"] = str(test_csv)
    if model_path is not None:
        env["VFI_MODEL_PATH"] = str(model_path)
    if data_dir is not None:
        env["VFI_PRIVACY_STATE_DIR"] = str(Path(schemas_dir) / "privacy")
    server_command = [python_executable, str(SERVER), "--rounds", str(rounds),
                      "--min-clients", "4", "--port", str(port),
                      "--patience", str(patience)]
    if validation_only:
        server_command.append("--validation-only")

    log_dir = Path(schemas_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    handles = []
    try:
        server_log = open(log_dir / "server.log", "w", encoding="utf-8")
        handles.append(server_log)
        server = subprocess.Popen(
            server_command, cwd=ROOT, env=env, stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        processes.append(server)
        wait_for_server(server, log_dir / "server.log")

        for client_id in range(1, 5):
            client_log = open(log_dir / f"client_{client_id}.log", "w", encoding="utf-8")
            handles.append(client_log)
            client_env = {**env, "CLIENT_ID": str(client_id)}
            if data_dir is not None:
                client_env["DATA_FILE"] = str(Path(data_dir) / f"bank_{client_id}.csv")
            client = subprocess.Popen(
                [python_executable, str(CLIENT), "--cid", str(client_id)],
                cwd=ROOT, env=client_env,
                stdout=client_log, stderr=subprocess.STDOUT,
            )
            processes.append(client)

        deadline = time.monotonic() + 4 * 60 * 60
        while time.monotonic() < deadline:
            server_code = server.poll()
            if server_code is not None:
                break
            failed_clients = [
                (index + 1, client.poll())
                for index, client in enumerate(processes[1:])
                if client.poll() not in (None, 0)
            ]
            if failed_clients:
                details = []
                for client_id, exit_code in failed_clients:
                    client_log = log_dir / f"client_{client_id}.log"
                    tail = client_log.read_text(
                        encoding="utf-8", errors="replace"
                    )[-2000:] if client_log.exists() else ""
                    details.append(f"client {client_id} exited {exit_code}:\n{tail}")
                raise RuntimeError(
                    "Federated client process failed before server completion:\n"
                    + "\n".join(details)
                )
            time.sleep(0.5)
        else:
            raise TimeoutError("Federated server exceeded the four-hour run limit.")
        if server_code != 0:
            tail = (log_dir / "server.log").read_text(
                encoding="utf-8", errors="replace"
            )[-4000:]
            raise RuntimeError(
                f"Flower server exited with code {server_code}:\n{tail}"
            )
        client_codes = [client.wait(timeout=120) for client in processes[1:]]
        if any(code != 0 for code in client_codes):
            raise RuntimeError(f"One or more Flower clients exited unsuccessfully: {client_codes}.")
    except Exception:
        stop_processes(processes)
        raise
    finally:
        for handle in handles:
            handle.close()


def read_trial_result(schemas_dir, expected_run_id):
    rounds = []
    for path in Path(schemas_dir).glob("global_model_round_*.json"):
        try:
            data = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("run_id") != expected_run_id or data.get("evaluation") != "validation":
            continue
        metrics = data.get("metrics", {})
        rounds.append({
            "round": int(data["round"]),
            "timestamp": data.get("timestamp"),
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "f1": float(metrics.get("f1", 0.0)),
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "auc": float(metrics.get("auc", 0.0)),
            "average_precision": float(metrics.get("average_precision", 0.0)),
            "decision_threshold": float(metrics.get("decision_threshold", 0.5)),
            "calibrated_threshold": float(metrics.get("calibrated_threshold", 0.5)),
            "temperature": float(metrics.get("temperature", 1.0)),
            "tp": int(metrics.get("tp", 0)),
            "fp": int(metrics.get("fp", 0)),
            "fn": int(metrics.get("fn", 0)),
            "tn": int(metrics.get("tn", 0)),
            "threshold_selection": metrics.get("threshold_selection"),
            "client_fit_summary": data.get("client_fit_summary", {}),
            "selected_by_validation": bool(data.get("selected_by_validation", False)),
        })
    if not rounds:
        raise RuntimeError(f"No validation metrics were exported under {schemas_dir}.")
    selected_rounds = [row for row in rounds if row["selected_by_validation"]]
    best = (
        max(selected_rounds, key=lambda row: row["round"])
        if selected_rounds
        else max(rounds, key=lambda row: row["average_precision"])
    )
    return best, sorted(rounds, key=lambda row: row["round"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=25, help="Maximum validation screening rounds per candidate")
    parser.add_argument("--final-rounds", type=int, default=25, help="Maximum rounds for the selected final model")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience on validation AP")
    parser.add_argument("--seeds", default="42,43", help="Comma-separated reproducibility seeds")
    parser.add_argument("--weights", default="1,10,25,50", help="Comma-separated positive-class loss weights")
    parser.add_argument("--learning-rates", default="0.002",
                        help="Comma-separated client learning rates")
    parser.add_argument("--noise-multipliers", default="0.5", help="Comma-separated DP noise multipliers")
    parser.add_argument("--fedprox-mus", default="0.0,0.01",
                        help="Comma-separated FedProx mu values (0.0 is FedAvg)")
    parser.add_argument("--loss-types", default="bce", help="Comma-separated loss types: bce,focal")
    parser.add_argument("--resampling-methods", default="none",
                        help="Comma-separated client-local samplers: none,smote_tomek")
    parser.add_argument("--max-grad-norms", default="1.0",
                        help="Comma-separated Opacus clipping bounds")
    parser.add_argument("--model-dropouts", default="0.1",
                        help="Comma-separated hidden-layer dropout rates")
    parser.add_argument("--smote-strategy", type=float, default=0.05,
                        help="Client-local SMOTE minority/majority ratio")
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--no-dp-diagnostic", action="store_true",
                        help="Omit the no-DP utility diagnostic candidate.")
    parser.add_argument("--local-epochs", type=int, default=1)
    default_python = PROJECT_VENV_PYTHON if PROJECT_VENV_PYTHON.exists() else Path(sys.executable)
    parser.add_argument("--python-executable", default=str(default_python),
                        help="Python interpreter with Flower, PyTorch, and Opacus installed")
    parser.add_argument("--port", type=int, default=8080,
                        help="Local Flower port (must be unused while the sweep runs)")
    parser.add_argument("--skip-finalize", action="store_true",
                        help="Do not retrain the selected configuration or evaluate final test.")
    args = parser.parse_args()
    if args.rounds < 1 or args.final_rounds < 1 or args.local_epochs < 1 or args.patience < 1:
        parser.error("rounds, patience, and local-epochs must be positive integers")

    manifest = read_json(MANIFEST)
    if manifest.get("pipeline_version") != "causal-v3" or manifest.get("status") != "complete":
        raise RuntimeError("Run completed causal-v3 preprocessing before the sweep.")
    run_id = manifest["run_id"]
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        parser.error("Provide at least one seed.")

    weights = [float(value.strip()) for value in args.weights.split(",") if value.strip()]
    learning_rates = [
        float(value.strip()) for value in args.learning_rates.split(",") if value.strip()
    ]
    noise_values = [float(value.strip()) for value in args.noise_multipliers.split(",") if value.strip()]
    fedprox_values = [float(value.strip()) for value in args.fedprox_mus.split(",") if value.strip()]
    loss_types = [value.strip().lower() for value in args.loss_types.split(",") if value.strip()]
    resampling_methods = [
        value.strip().lower() for value in args.resampling_methods.split(",") if value.strip()
    ]
    clip_norms = [float(value.strip()) for value in args.max_grad_norms.split(",") if value.strip()]
    dropout_values = [
        float(value.strip()) for value in args.model_dropouts.split(",") if value.strip()
    ]
    if not weights or any(value <= 0 for value in weights):
        parser.error("All positive-class loss weights must be positive.")
    if not learning_rates or any(value <= 0 for value in learning_rates):
        parser.error("All client learning rates must be positive.")
    if not noise_values or any(value <= 0 for value in noise_values):
        parser.error("All DP noise multipliers must be positive.")
    if not fedprox_values or any(value < 0 for value in fedprox_values):
        parser.error("FedProx mu values must be non-negative.")
    if not loss_types or any(value not in ("bce", "focal") for value in loss_types):
        parser.error("Loss types must be bce and/or focal.")
    if not resampling_methods or any(
        value not in ("none", "smote_tomek") for value in resampling_methods
    ):
        parser.error("Resampling methods must be none and/or smote_tomek.")
    if not clip_norms or any(value <= 0 for value in clip_norms):
        parser.error("All max-grad-norm values must be positive.")
    if not dropout_values or any(not 0.0 <= value < 1.0 for value in dropout_values):
        parser.error("Dropout values must be in [0, 1).")
    if not 0.0 < args.smote_strategy < 1.0:
        parser.error("SMOTE strategy must be between 0 and 1.")

    # Keep this grid small and user-adjustable for a local proof of concept.
    # The no-DP condition is diagnostic only and is never eligible for selection.
    configurations = []
    for loss_type in loss_types:
        loss_weights = weights if loss_type == "bce" else [1.0]
        for weight in loss_weights:
            for noise in noise_values:
                for learning_rate in learning_rates:
                    for mu in fedprox_values:
                        for resampling in resampling_methods:
                            for max_grad_norm in clip_norms:
                                for model_dropout in dropout_values:
                                    configurations.append({
                                        "positive_loss_weight": weight,
                                        "noise_multiplier": noise,
                                        "learning_rate": learning_rate,
                                        "fedprox_mu": mu,
                                        "dp_enabled": True,
                                        "loss_type": loss_type,
                                        "resampling": resampling,
                                        "max_grad_norm": max_grad_norm,
                                        "model_dropout": model_dropout,
                                        "smote_strategy": args.smote_strategy,
                                        "focal_alpha": args.focal_alpha,
                                        "focal_gamma": args.focal_gamma,
                                    })
    if not args.no_dp_diagnostic:
        configurations.append({
            "positive_loss_weight": weights[0],
            "noise_multiplier": noise_values[0],
            "learning_rate": learning_rates[0],
            "fedprox_mu": fedprox_values[0],
            "dp_enabled": False,
            "loss_type": "bce",
            "resampling": "none",
            "max_grad_norm": clip_norms[0],
            "model_dropout": dropout_values[0],
            "smote_strategy": args.smote_strategy,
            "focal_alpha": args.focal_alpha,
            "focal_gamma": args.focal_gamma,
        })
    sweep_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    experiment_root = ROOT / "schemas" / "validation_sweeps" / f"{run_id}_{sweep_id}"
    results = []

    for config_index, config in enumerate(configurations, start=1):
        for seed in seeds:
            tag = f"sweep{config_index}_seed{seed}"
            trial_dir = experiment_root / tag
            print(
                f"[SWEEP] {tag}: loss={config['loss_type']}, "
                f"weight={config['positive_loss_weight']}, "
                f"lr={config['learning_rate']}, DP={config['dp_enabled']}, "
                f"sigma={config['noise_multiplier']}, "
                f"rounds={args.rounds} (validation only)", flush=True,
            )
            try:
                run_fl(
                    args.python_executable,
                    args.rounds, seed, config["positive_loss_weight"],
                    config["noise_multiplier"], config["dp_enabled"], config["learning_rate"],
                    args.local_epochs, config["fedprox_mu"], trial_dir,
                    f"{sweep_id}_{tag}",
                    validation_only=True, port=args.port,
                    loss_type=config["loss_type"],
                    resampling=config["resampling"],
                    max_grad_norm=config["max_grad_norm"],
                    model_dropout=config["model_dropout"],
                    smote_strategy=config["smote_strategy"],
                    focal_alpha=config["focal_alpha"],
                    focal_gamma=config["focal_gamma"],
                    patience=args.patience,
                )
            except Exception as exc:
                append_ablation_result(
                    ROOT,
                    run_id=f"{sweep_id}_{tag}",
                    experiment_name=f"validation_{config['loss_type']}_{config['resampling']}",
                    hyperparameters={
                        "data_run_id": run_id,
                        "rounds": args.rounds,
                        "selected_round": None,
                        "loss_type": config["loss_type"],
                        "pos_weight": config["positive_loss_weight"],
                        "resampling": config["resampling"],
                        "dp_enabled": config["dp_enabled"],
                        "max_grad_norm": config["max_grad_norm"],
                        "epsilon": None,
                        "delta": 1e-5 if config["dp_enabled"] else 0.0,
                        "dp_accounting_validity": (
                            "not_formally_valid_for_original_records_after_synthetic_resampling"
                            if config["dp_enabled"] and config["resampling"] != "none"
                            else "standard_opacus_accounting"
                            if config["dp_enabled"] else "disabled"
                        ),
                        "aggregation_strategy": (
                            f"FedProx(mu={config['fedprox_mu']})"
                            if config["fedprox_mu"] > 0 else "FedAvg"
                        ),
                        "noise_multiplier": config["noise_multiplier"],
                    },
                    validation_metrics=None,
                    status="failed",
                    error=str(exc),
                )
                raise
            best, all_rounds = read_trial_result(trial_dir, run_id)
            client_summary = best["client_fit_summary"]
            append_ablation_result(
                ROOT,
                run_id=f"{sweep_id}_{tag}",
                experiment_name=f"validation_{config['loss_type']}_{config['resampling']}",
                timestamp=best.get("timestamp"),
                hyperparameters={
                    "data_run_id": run_id,
                    "rounds": args.rounds,
                    "selected_round": best["round"],
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
                    "aggregation_strategy": (
                        f"FedProx(mu={config['fedprox_mu']})"
                        if config["fedprox_mu"] > 0 else "FedAvg"
                    ),
                    "learning_rate": config["learning_rate"],
                    "local_epochs": args.local_epochs,
                    "model_dropout": config["model_dropout"],
                    "focal_alpha": config["focal_alpha"],
                    "focal_gamma": config["focal_gamma"],
                    "temperature": best["temperature"],
                    "threshold_selection": best["threshold_selection"],
                },
                validation_metrics=best,
            )
            results.append({
                **config,
                "seed": seed,
                "best_validation": best,
                "round_metrics": all_rounds,
                "artifact_dir": str(trial_dir),
            })
            print(
                f"[SWEEP RESULT] {tag}: best validation F1={best['f1']:.4f}, "
                f"P={best['precision']:.4f}, R={best['recall']:.4f}, "
                f"AP={best['average_precision']:.4f}", flush=True,
            )

    eligible = [row for row in results if row["dp_enabled"]]
    means = {}
    for row in eligible:
        key = (
            row["positive_loss_weight"], row["noise_multiplier"],
            row["learning_rate"], row["fedprox_mu"],
            row["loss_type"], row["resampling"], row["max_grad_norm"],
            row["model_dropout"], row["smote_strategy"],
        )
        means.setdefault(key, []).append(row["best_validation"]["average_precision"])
    selected_key = max(means, key=lambda key: sum(means[key]) / len(means[key]))
    selected = {
        "positive_loss_weight": selected_key[0],
        "noise_multiplier": selected_key[1],
        "learning_rate": selected_key[2],
        "fedprox_mu": selected_key[3],
        "loss_type": selected_key[4],
        "resampling": selected_key[5],
        "max_grad_norm": selected_key[6],
        "model_dropout": selected_key[7],
        "smote_strategy": selected_key[8],
        "dp_enabled": True,
        "mean_best_validation_average_precision": (
            sum(means[selected_key]) / len(means[selected_key])
        ),
        "seeds": seeds,
    }

    final_record = None
    if not args.skip_finalize:
        final_tag = f"selected_{selected_key[4]}_{selected_key[5]}_seed{seeds[0]}"
        print(
            f"[FINALIZE] Retraining selected DP configuration with {args.final_rounds} rounds; "
            "final test will be evaluated once after validation selection.", flush=True,
        )
        run_fl(
            args.python_executable,
            args.final_rounds, seeds[0], selected_key[0], selected_key[1], True,
            selected_key[2], args.local_epochs, selected_key[3], ROOT / "schemas",
            f"{sweep_id}_{final_tag}",
            validation_only=False,
            port=args.port,
            loss_type=selected_key[4],
            resampling=selected_key[5],
            max_grad_norm=selected_key[6],
            model_dropout=selected_key[7],
            smote_strategy=selected_key[8],
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            patience=args.patience,
        )
        final_record = read_json(ROOT / "schemas" / "final_evaluation.json")
        if final_record.get("run_id") != run_id:
            raise RuntimeError("Final evaluation run_id does not match the current preprocessing run.")
        final_hyperparameters = {
            "data_run_id": run_id,
            "rounds": args.final_rounds,
            "selected_round": final_record["selected_round"],
            "loss_type": selected_key[4],
            "pos_weight": selected_key[0],
            "resampling": selected_key[5],
            "smote_strategy": selected_key[8],
            "dp_enabled": True,
            "max_grad_norm": selected_key[6],
            "noise_multiplier": selected_key[1],
            "epsilon": final_record.get("client_fit_summary", {}).get("max_client_epsilon"),
            "delta": final_record.get("client_fit_summary", {}).get("delta"),
            "dp_accounting_validity": final_record.get(
                "client_fit_summary", {}
            ).get("dp_accounting_validity"),
            "aggregation_strategy": (
                f"FedProx(mu={selected_key[3]})" if selected_key[3] > 0 else "FedAvg"
            ),
            "learning_rate": selected_key[2],
            "local_epochs": args.local_epochs,
            "model_dropout": selected_key[7],
            "focal_alpha": args.focal_alpha,
            "focal_gamma": args.focal_gamma,
            "temperature": final_record.get("temperature"),
            "threshold_selection": final_record.get("threshold_selection"),
        }
        append_ablation_result(
            ROOT,
            run_id=f"{sweep_id}_{final_tag}",
            experiment_name=f"finalized_{selected_key[4]}_{selected_key[5]}",
            timestamp=final_record.get("timestamp"),
            hyperparameters=final_hyperparameters,
            validation_metrics=final_record.get("validation_metrics"),
            final_test_metrics=final_record.get("metrics"),
        )

    report = {
        "sweep_id": sweep_id,
        "run_id": run_id,
        "selection": "mean_best_validation_average_precision_across_seeds",
        "screening_rounds": args.rounds,
        "final_rounds": args.final_rounds,
        "local_epochs": args.local_epochs,
        "candidates": results,
        "selected_dp_configuration": selected,
        "no_dp_diagnostic": [row for row in results if not row["dp_enabled"]],
        "final_test_once": final_record,
        "note": "The no-DP condition is a diagnostic only. All candidate selection used validation data; final test metrics were read only after choosing the DP configuration.",
    }
    output_path = ROOT / "evaluation" / f"fl_validation_sweep_{sweep_id}.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[SAVE] Sweep report: {output_path}", flush=True)
    print(
        f"[SELECTED] DP weight={selected_key[0]}, sigma={selected_key[1]}, "
        f"lr={selected_key[2]}",
        flush=True,
    )
    if final_record:
        metrics = final_record["metrics"]
        print(
            f"[FINAL TEST] F1={metrics['f1']:.4f}, P={metrics['precision']:.4f}, "
            f"R={metrics['recall']:.4f}, AP={metrics.get('average_precision', 0.0):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
