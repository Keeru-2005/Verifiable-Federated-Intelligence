"""
End-to-End Pipeline & Failure Handling Test Suite
Validates the complete 8-stage pipeline from FL training to ZK proving and blockchain auditability,
plus failure-handling on invalid proofs, malformed artifacts, and disconnected clients.
"""
import os
import sys
import json
import shutil
import pytest
import subprocess
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.mlp import GraphAwareMLP
from fl_implementation.client import AMLClient


# These artifacts are canonical repository outputs; the E2E test generates its
# own fixtures but must not leave fabricated metrics or proofs behind.
_PRESERVED_ARTIFACTS = [
    "schemas/proof.json",
    "schemas/global_model_round_1.json",
    "schemas/e2e_test_round_1.json",
    "schemas/final_evaluation.json",
    "schemas/round_updates/round_1.json",
    "schemas/round_updates/circuit_round_1.json",
    "zkp/circuit/input.json",
    "clients.json",
    "fl_implementation/model.pt",
    "blockchain/deployed_addresses.json",
]


@pytest.fixture(scope="module", autouse=True)
def preserve_repo_artifacts():
    """Back up canonical artifacts and restore them after the E2E tests."""
    backups = {}
    missing = []
    for rel in _PRESERVED_ARTIFACTS:
        path = os.path.join(PROJECT_ROOT, rel)
        if os.path.exists(path):
            with open(path, "rb") as f:
                backups[path] = f.read()
        else:
            missing.append(path)
    yield
    for path, data in backups.items():
        with open(path, "wb") as f:
            f.write(data)
    for path in missing:
        if os.path.exists(path):
            os.remove(path)


def test_e2e_full_pipeline_run(causal_client_data):
    """
    Test a complete successful end-to-end run:
    1. 4 Bank clients train for 1 round
    2. Weights are aggregated into global model checkpoint (model.pt)
    3. Final layer parameters (33 params) are extracted into clients.json
    4. Circuit inputs are generated into input.json
    5. Real zk-SNARK proof is generated via SnarkJS
    6. Local cryptographic verification passes
    7. Hardhat verifies proof on-chain and writes ledger record
    """
    print("\n--- [E2E] Step 1: 4-Client Federated Training Round ---")
    clients = [AMLClient(client_id=str(i)) for i in range(1, 5)]
    global_model = GraphAwareMLP(input_dim=32)
    initial_weights = global_model.get_parameters()

    fit_results = []
    for c in clients:
        weights, n_samples, metrics = c.fit(initial_weights, {})
        fit_results.append((weights, n_samples))

    # FedAvg parameter aggregation
    total_samples = sum(n for _, n in fit_results)
    aggregated = []
    for idx in range(len(initial_weights)):
        agg_tensor = sum(w[idx] * n for w, n in fit_results) / total_samples
        aggregated.append(agg_tensor)

    global_model.set_parameters(aggregated)
    model_pt_path = os.path.join(PROJECT_ROOT, "fl_implementation", "model.pt")
    torch.save(global_model.state_dict(), model_pt_path)
    assert os.path.exists(model_pt_path), "model.pt must be created"

    # Export round 1 json
    schemas_dir = os.path.join(PROJECT_ROOT, "schemas")
    os.makedirs(schemas_dir, exist_ok=True)
    round_json_path = os.path.join(schemas_dir, "e2e_test_round_1.json")
    run_id = "isolated-e2e-test-run"
    with open(round_json_path, "w") as f:
        json.dump({
            "round": 1,
            "run_id": run_id,
            "weights": [arr.tolist() for arr in aggregated],
            "metrics": {"accuracy": 0.985, "precision": 0.98, "recall": 0.97, "f1": 0.975},
            "evaluation": "e2e_test_fixture_not_a_benchmark",
            "client_count": 4
        }, f, indent=2)
    assert os.path.exists(round_json_path)

    def output_layer_parameters(parameters):
        model = GraphAwareMLP(input_dim=32)
        model.set_parameters(parameters)
        state = model.state_dict()
        return (
            state["output_layer.0.weight"].detach().cpu().reshape(-1).tolist()
            + state["output_layer.0.bias"].detach().cpu().reshape(-1).tolist()
        )

    updates_dir = os.path.join(schemas_dir, "round_updates")
    os.makedirs(updates_dir, exist_ok=True)
    round_updates = {
        "round": 1,
        "run_id": run_id,
        "aggregation": "Flower FedAvg",
        "clients": [
            {
                "client_id": str(index + 1),
                "num_examples": count,
                "output_layer_parameters": output_layer_parameters(weights),
            }
            for index, (weights, count) in enumerate(fit_results)
        ],
        "global_output_layer_parameters": output_layer_parameters(aggregated),
    }
    round_updates_path = os.path.join(updates_dir, "round_1.json")
    with open(round_updates_path, "w", encoding="utf-8") as update_file:
        json.dump(round_updates, update_file, indent=2)

    manifest_path = os.path.join(str(causal_client_data.parent), "preprocess_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump({
            "pipeline_version": "causal-v3",
            "status": "complete",
            "run_id": run_id,
        }, manifest_file)

    final_evaluation_path = os.path.join(schemas_dir, "final_evaluation.json")
    with open(final_evaluation_path, "w", encoding="utf-8") as evaluation_file:
        json.dump({
            "selected_round": 1,
            "evaluation": "final_test_once",
            "run_id": run_id,
            "metrics": {"accuracy": 0.985, "precision": 0.98, "recall": 0.97, "f1": 0.975},
        }, evaluation_file)

    print("--- [E2E] Step 2: Weight Extraction into clients.json ---")
    extract_cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "fl_implementation", "extract_client_weights.py"),
        "1",
    ]
    extract_env = dict(os.environ, VFI_MANIFEST_PATH=manifest_path)
    res = subprocess.run(
        extract_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=extract_env,
    )
    assert res.returncode == 0, f"extract_client_weights failed: {res.stderr}"
    clients_json_path = os.path.join(updates_dir, "circuit_round_1.json")
    assert os.path.exists(clients_json_path)

    with open(clients_json_path, encoding="utf-8") as f:
        cdata = json.load(f)
    assert len(cdata["clientWeights"]) == 4
    assert len(cdata["clientWeights"][0]) == 33, "Each client must expose 33 output-layer parameters"
    assert len(cdata["clientCounts"]) == 4

    print("--- [E2E] Step 3: Quantize & Generate Circuit Input ---")
    input_json_path = os.path.join(str(causal_client_data.parent), "input.json")
    gen_input_cmd = [
        "node", os.path.join(PROJECT_ROOT, "zkp", "circuit", "generate_circuit_input.js"),
        clients_json_path, "1", input_json_path,
    ]
    res = subprocess.run(gen_input_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, f"generate_circuit_input.js failed: {res.stderr}"
    assert os.path.exists(input_json_path)

    print("--- [E2E] Step 4: Real zk-SNARK Proof Generation ---")
    gen_proof_cmd = ["node", os.path.join(PROJECT_ROOT, "zkp", "generate_proof.js"), input_json_path, "1"]
    res = subprocess.run(gen_proof_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, f"generate_proof.js failed: {res.stderr}"
    proof_json_path = os.path.join(schemas_dir, "proof.json")
    assert os.path.exists(proof_json_path)

    print("--- [E2E] Step 5: Cryptographic Pairing Verification ---")
    verify_cmd = ["node", os.path.join(PROJECT_ROOT, "zkp", "verify_proof.js")]
    res = subprocess.run(verify_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, f"verify_proof.js failed: {res.stderr}"
    assert "SUCCESS: Cryptographic proof is valid!" in res.stdout

    print("--- [E2E] Step 6: Blockchain Smart Contract Submission ---")
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    assert npx, "npx is required for the on-chain E2E step"
    submit_cmd = [npx, "hardhat", "run", "scripts/submit_proof.js"]
    submit_env = dict(os.environ, MODEL_ROUND_PATH=round_json_path)
    res = subprocess.run(submit_cmd, cwd=os.path.join(PROJECT_ROOT, "blockchain"), capture_output=True, text=True, encoding="utf-8", errors="replace", env=submit_env)
    assert res.returncode == 0, f"submit_proof.js failed: {res.stderr}"
    assert "Verification Successful!" in res.stdout or "Transaction confirmed" in res.stdout
    print("[SUCCESS] Full End-to-End Pipeline Executed and Verified Successfully!")


def test_invalid_proof_rejection():
    """Verify that an altered proof is rejected by verify_proof.js with exit code 1."""
    proof_path = os.path.join(PROJECT_ROOT, "schemas", "proof.json")
    with open(proof_path) as f:
        proof_obj = json.load(f)

    corrupted = dict(proof_obj)
    corrupted["proof"] = dict(proof_obj["proof"])
    corrupted["proof"]["pi_a"] = ["999999999999999999999999", "8888888888888888888888", "1"]

    temp_path = os.path.join(PROJECT_ROOT, "schemas", "proof_corrupted_temp.json")
    with open(temp_path, "w") as f:
        json.dump(corrupted, f)

    try:
        # Run verification on corrupted proof
        vkey_path = os.path.join(PROJECT_ROOT, "zkp", "circuit", "verification_key.json")
        verify_cmd = [
            "node", "-e", f"""
            const snarkjs = require("snarkjs");
            const fs = require("fs");
            const vKey = JSON.parse(fs.readFileSync("{vkey_path}"));
            const proofData = JSON.parse(fs.readFileSync("{temp_path}"));
            snarkjs.groth16.verify(vKey, proofData.publicSignals, proofData.proof)
                .then(valid => process.exit(valid ? 0 : 1))
                .catch(() => process.exit(1));
            """
        ]
        res = subprocess.run(verify_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert res.returncode != 0, "Corrupted proof was not rejected!"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_missing_input_error_handling():
    """Verify that missing input file causes generate_proof.js to fail cleanly with actionable message."""
    cmd = ["node", os.path.join(PROJECT_ROOT, "zkp", "generate_proof.js"), "non_existent_file.json"]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode != 0, "generate_proof should fail when input file is missing"
    assert "not found" in (res.stdout + res.stderr).lower()


def test_disconnected_client_quorum_enforcement():
    """Verify that Flower FL server strictly enforces 4-client quorum when a client drops out."""
    stress_test_path = os.path.join(PROJECT_ROOT, "evaluation", "stress_test.py")
    res = subprocess.run([sys.executable, stress_test_path], cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, f"stress_test failed: {res.stderr}"
    assert "STRESS TEST PASSED" in res.stdout
