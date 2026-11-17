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
           validation_only, port):
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
        "VFI_SCHEMAS_DIR": str(schemas_dir),
    })
    server_command = [python_executable, str(SERVER), "--rounds", str(rounds),
                      "--min-clients", "4", "--port", str(port)]
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
            client = subprocess.Popen(
                [python_executable, str(CLIENT), "--cid", str(client_id)],
                cwd=ROOT, env={**env, "CLIENT_ID": str(client_id)},
                stdout=client_log, stderr=subprocess.STDOUT,
            )
            processes.append(client)

        server_code = server.wait(timeout=4 * 60 * 60)
        if server_code != 0:
            raise RuntimeError(f"Flower server exited with code {server_code}.")
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
            "f1": float(metrics.get("f1", 0.0)),
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "average_precision": float(metrics.get("average_precision", 0.0)),
            "decision_threshold": float(metrics.get("decision_threshold", 0.5)),
            "tp": int(metrics.get("tp", 0)),
            "fp": int(metrics.get("fp", 0)),
            "fn": int(metrics.get("fn", 0)),
            "client_fit_summary": data.get("client_fit_summary", {}),
        })
    if not rounds:
        raise RuntimeError(f"No validation metrics were exported under {schemas_dir}.")
    return max(rounds, key=lambda row: row["f1"]), sorted(rounds, key=lambda row: row["round"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5, help="Validation screening rounds per candidate")
    parser.add_argument("--final-rounds", type=int, default=10, help="Rounds for the selected final model")
    parser.add_argument("--seeds", default="42,43", help="Comma-separated reproducibility seeds")
    parser.add_argument("--weights", default="10,25,50", help="Comma-separated positive-class loss weights")
    parser.add_argument("--learning-rates", default="0.002",
                        help="Comma-separated client learning rates")
    parser.add_argument("--noise-multipliers", default="0.5", help="Comma-separated DP noise multipliers")
    parser.add_argument("--fedprox-mus", default="0.0,0.01",
                        help="Comma-separated FedProx mu values (0.0 is FedAvg)")
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
    if args.rounds < 1 or args.final_rounds < 1 or args.local_epochs < 1:
        parser.error("rounds and local-epochs must be positive integers")

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
    if not weights or any(value <= 0 for value in weights):
        parser.error("All positive-class loss weights must be positive.")
    if not learning_rates or any(value <= 0 for value in learning_rates):
        parser.error("All client learning rates must be positive.")
    if not noise_values or any(value <= 0 for value in noise_values):
        parser.error("All DP noise multipliers must be positive.")
    if not fedprox_values or any(value < 0 for value in fedprox_values):
        parser.error("FedProx mu values must be non-negative.")

    # Keep this grid small and user-adjustable for a local proof of concept.
    # The no-DP condition is diagnostic only and is never eligible for selection.
    configurations = [
        {"positive_loss_weight": weight, "noise_multiplier": noise,
         "learning_rate": learning_rate, "fedprox_mu": mu, "dp_enabled": True}
        for weight in weights for noise in noise_values
        for learning_rate in learning_rates for mu in fedprox_values
    ]
    if not args.no_dp_diagnostic:
        configurations.append({
            "positive_loss_weight": weights[0],
            "noise_multiplier": noise_values[0],
            "learning_rate": learning_rates[0],
            "fedprox_mu": fedprox_values[0],
            "dp_enabled": False,
        })
    sweep_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    experiment_root = ROOT / "schemas" / "validation_sweeps" / f"{run_id}_{sweep_id}"
    results = []

    for config_index, config in enumerate(configurations, start=1):
        for seed in seeds:
            tag = f"sweep{config_index}_seed{seed}"
            trial_dir = experiment_root / tag
            print(
                f"[SWEEP] {tag}: weight={config['positive_loss_weight']}, "
                f"lr={config['learning_rate']}, DP={config['dp_enabled']}, "
                f"sigma={config['noise_multiplier']}, "
                f"rounds={args.rounds} (validation only)", flush=True,
            )
            run_fl(
                args.python_executable,
                args.rounds, seed, config["positive_loss_weight"],
                config["noise_multiplier"], config["dp_enabled"], config["learning_rate"],
                args.local_epochs, config["fedprox_mu"], trial_dir,
                f"{sweep_id}_{tag}",
                validation_only=True, port=args.port,
            )
            best, all_rounds = read_trial_result(trial_dir, run_id)
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
        )
        means.setdefault(key, []).append(row["best_validation"]["f1"])
    selected_key = max(means, key=lambda key: sum(means[key]) / len(means[key]))
    selected = {
        "positive_loss_weight": selected_key[0],
        "noise_multiplier": selected_key[1],
        "learning_rate": selected_key[2],
        "fedprox_mu": selected_key[3],
        "dp_enabled": True,
        "mean_best_validation_f1": sum(means[selected_key]) / len(means[selected_key]),
        "seeds": seeds,
    }

    final_record = None
    if not args.skip_finalize:
        final_tag = (
            f"selected_w{selected_key[0]}_sigma{selected_key[1]}_"
            f"lr{selected_key[2]}_seed{seeds[0]}"
        )
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
        )
        final_record = read_json(ROOT / "schemas" / "final_evaluation.json")
        if final_record.get("run_id") != run_id:
            raise RuntimeError("Final evaluation run_id does not match the current preprocessing run.")

    report = {
        "sweep_id": sweep_id,
        "run_id": run_id,
        "selection": "mean_best_validation_f1_across_seeds",
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
