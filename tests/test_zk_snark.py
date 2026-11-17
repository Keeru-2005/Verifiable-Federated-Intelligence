"""
Zero-Knowledge Proof (ZK-SNARK) Test Suite
Validates circuit artifacts, quantization rules, proof generation, pairing verification, and tamper rejection.
"""
import os
import sys
import json
import pytest
import subprocess
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZKP_DIR = os.path.join(PROJECT_ROOT, "zkp")
CIRCUIT_DIR = os.path.join(ZKP_DIR, "circuit")
SCHEMAS_DIR = os.path.join(PROJECT_ROOT, "schemas")


def _schemas_dir():
    """Directory the proof scripts should use (tests redirect it to a temp dir)."""
    return os.environ.get("VFI_SCHEMAS_DIR", SCHEMAS_DIR)


@pytest.fixture(scope="module", autouse=True)
def isolate_proof_exports(tmp_path_factory):
    """Keep proof-generation tests from overwriting the canonical proof.

    Groth16 proofs are randomized, so regenerating them in place changes the
    documented artifact even when the public signals are identical. Redirect the
    scripts (via their environment overrides) to a temporary schema directory.
    """
    temp_dir = tmp_path_factory.mktemp("vfi_test_schemas")
    previous_schemas = os.environ.get("VFI_SCHEMAS_DIR")
    previous_proof = os.environ.get("VFI_PROOF_PATH")
    os.environ["VFI_SCHEMAS_DIR"] = str(temp_dir)
    os.environ["VFI_PROOF_PATH"] = str(temp_dir / "proof.json")
    yield
    if previous_schemas is None:
        os.environ.pop("VFI_SCHEMAS_DIR", None)
    else:
        os.environ["VFI_SCHEMAS_DIR"] = previous_schemas
    if previous_proof is None:
        os.environ.pop("VFI_PROOF_PATH", None)
    else:
        os.environ["VFI_PROOF_PATH"] = previous_proof


def test_circuit_artifacts_exist():
    """Verify that Circom compilation artifacts, keys, and verifier contracts exist."""
    r1cs_file = os.path.join(CIRCUIT_DIR, "aml_verify.r1cs")
    wasm_file = os.path.join(CIRCUIT_DIR, "aml_verify_js", "aml_verify.wasm")
    zkey_file = os.path.join(CIRCUIT_DIR, "aml_verify_final.zkey")
    vkey_file = os.path.join(CIRCUIT_DIR, "verification_key.json")
    sol_verifier = os.path.join(PROJECT_ROOT, "blockchain", "contracts", "Groth16Verifier.sol")

    assert os.path.exists(r1cs_file), f"R1CS missing at {r1cs_file}"
    assert os.path.exists(wasm_file), f"WASM binary missing at {wasm_file}"
    assert os.path.exists(zkey_file), f"Proving key missing at {zkey_file}"
    assert os.path.exists(vkey_file), f"Verification key missing at {vkey_file}"
    assert os.path.exists(sol_verifier), f"Solidity verifier missing at {sol_verifier}"


def test_quantization_consistency():
    """Verify fixed-point quantization encoding (scale 10^6, offset 2^32)."""
    SCALE = 1_000_000
    OFFSET = 2 ** 32

    def encode(val):
        scaled = round(val * SCALE)
        return str(scaled + OFFSET)

    def decode(enc_str):
        val = int(enc_str)
        return (val - OFFSET) / SCALE

    test_weights = [-0.034521, 0.0, 0.045612, -1.5, 2.345678]
    for orig in test_weights:
        encoded = encode(orig)
        assert int(encoded) >= 0, "Encoded fixed-point weight must be non-negative"
        recovered = decode(encoded)
        assert abs(orig - recovered) < 1e-5, f"Recovery precision lost: orig={orig}, recovered={recovered}"


def test_witness_and_proof_generation(tmp_path):
    """Execute real Groth16 witness and proof generation via generate_proof.js."""
    script_path = os.path.join(ZKP_DIR, "generate_proof.js")
    source_input_path = os.path.join(CIRCUIT_DIR, "input.json")
    input_path = tmp_path / "input.json"
    shutil.copyfile(source_input_path, input_path)
    with open(input_path, encoding="utf-8") as input_file:
        input_data = json.load(input_file)
    round_number = int(input_data["roundNumber"])
    with open(tmp_path / "context.json", "w", encoding="utf-8") as context_file:
        json.dump({
            "round": round_number,
            "totalSamples": input_data["totalSamples"],
            "run_id": "isolated-zk-test-run",
        }, context_file)

    res = subprocess.run(
        ["node", script_path, str(input_path), str(round_number)],
        cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert res.returncode == 0, f"generate_proof.js failed: {res.stderr}\n{res.stdout}"
    assert "Cryptographic Groth16 verification PASSED!" in res.stdout
    assert os.path.exists(os.path.join(_schemas_dir(), "proof.json"))


def test_genuine_proof_verification_passes():
    """Execute verify_proof.js and assert cryptographic pairing check passes."""
    script_path = os.path.join(ZKP_DIR, "verify_proof.js")
    res = subprocess.run(["node", script_path], cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, f"verify_proof.js failed: {res.stderr}"
    assert "SUCCESS: Cryptographic proof is valid!" in res.stdout


def test_tampered_public_input_rejection():
    """Verify that tampering with any public input causes verification to fail."""
    import tempfile
    proof_path = os.path.join(_schemas_dir(), "proof.json")
    vkey_path = os.path.join(CIRCUIT_DIR, "verification_key.json")

    with open(proof_path) as f:
        data = json.load(f)

    # Tamper commitment
    tampered = dict(data)
    tampered["publicSignals"] = ["99999999999999999999999999999999999999", data["publicSignals"][1], data["publicSignals"][2]]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(tampered, tf)
        temp_proof = tf.name

    try:
        # Run node verification script with tampered proof
        verify_cmd = [
            "node", "-e", f"""
            const snarkjs = require("snarkjs");
            const fs = require("fs");
            const vKey = JSON.parse(fs.readFileSync("{vkey_path}"));
            const proofData = JSON.parse(fs.readFileSync("{temp_proof}"));
            snarkjs.groth16.verify(vKey, proofData.publicSignals, proofData.proof)
                .then(valid => process.exit(valid ? 0 : 1))
                .catch(() => process.exit(1));
            """
        ]
        res = subprocess.run(verify_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert res.returncode != 0, "Tampered public input should have been rejected by pairing verification!"
    finally:
        if os.path.exists(temp_proof):
            os.remove(temp_proof)


def test_tampered_proof_points_rejection():
    """Verify that corrupting proof points pi_a or pi_c causes verification to fail."""
    import tempfile
    proof_path = os.path.join(_schemas_dir(), "proof.json")
    vkey_path = os.path.join(CIRCUIT_DIR, "verification_key.json")

    with open(proof_path) as f:
        data = json.load(f)

    # Corrupt pi_a
    tampered = dict(data)
    tampered["proof"]["pi_a"][0] = "123456789"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(tampered, tf)
        temp_proof = tf.name

    try:
        verify_cmd = [
            "node", "-e", f"""
            const snarkjs = require("snarkjs");
            const fs = require("fs");
            const vKey = JSON.parse(fs.readFileSync("{vkey_path}"));
            const proofData = JSON.parse(fs.readFileSync("{temp_proof}"));
            snarkjs.groth16.verify(vKey, proofData.publicSignals, proofData.proof)
                .then(valid => process.exit(valid ? 0 : 1))
                .catch(() => process.exit(1));
            """
        ]
        res = subprocess.run(verify_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert res.returncode != 0, "Corrupted proof points should have been rejected by pairing verification!"
    finally:
        if os.path.exists(temp_proof):
            os.remove(temp_proof)


def test_quantization_boundary_extreme_values():
    """Verify that quantization logic handles negative values, zero, and high precision safely."""
    SCALE = 1_000_000
    OFFSET = 2**32
    BN254_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617

    test_weights = [-15.87654321, 0.0, 15.87654321, -0.000001, 100.0]
    for w in test_weights:
        quantized = int(round(w * SCALE)) + OFFSET
        assert quantized >= 0, f"Quantized weight {quantized} must be non-negative"
        assert quantized < BN254_PRIME, f"Quantized weight {quantized} must be within BN254 field"

    # Test that extreme negative within normal range fits safely
    extreme_neg = -1000.0
    q_neg = int(round(extreme_neg * SCALE)) + OFFSET
    assert q_neg > 0, "Large negative weight should remain positive after offset addition"


def test_quantization_error_on_missing_clients_file():
    """Verify that generate_circuit_input.js exits cleanly with code 1 if clients file does not exist."""
    gen_script = os.path.join(CIRCUIT_DIR, "generate_circuit_input.js")
    res = subprocess.run(["node", gen_script, "non_existent_clients_file.json", "1"], cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode != 0, "Should return non-zero exit code on non-existent file"

