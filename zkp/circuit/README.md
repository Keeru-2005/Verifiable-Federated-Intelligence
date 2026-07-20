# ZK Circuit Compilation & Trusted Setup Instructions

This directory is designated for the zero-knowledge circuit definition (`aml_verify.circom`) and compilation outputs. 

## 🛠️ Tasks for Likith & Keerthana

### 1. Circuit Scaffolding (`aml_verify.circom`)
* Define the input shapes and aggregation logic. Keep constraint sizes minimized to reduce proving overhead.

### 2. Compilation and Trusted Setup
Once the circuit is ready, execute the following commands to generate the verification key and Prover/Verifier keys:

```bash
# Compile the circuit
circom aml_verify.circom --r1cs --wasm --sym

# Start a new powers of tau ceremony (BN254 curve)
snarkjs powersoftau new bn128 12 pot12_0000.ptau -v
snarkjs powersoftau contribute pot12_0000.ptau pot12_0001.ptau --name="First Contribution" -v

# Prepare phase 2
snarkjs powersoftau prepare phase2 pot12_0001.ptau pot12_final.ptau -v

# Generate verification keys
snarkjs groth16 setup aml_verify.r1cs pot12_final.ptau aml_verify_0000.zkey
snarkjs zkey contribute aml_verify_0000.zkey aml_verify_final.zkey --name="Verifier Contribution" -v

# Export the Verification Key (Mandatory)
snarkjs zkey export verificationkey aml_verify_final.zkey verification_key.json
```

> [!IMPORTANT]
> The exported `verification_key.json` MUST be saved directly to this directory (`zkp/circuit/verification_key.json`). Once saved, the local verifier utility `node zkp/verify_proof.js` will automatically switch from structural validation to cryptographic verification.

### 3. Proof Generation (`generate_proof.js`)
* Read the target round weights from `schemas/global_model_round_N.json`.
* Execute proof generation via `snarkjs groth16 fullprove` using the compiled `.wasm` file and `aml_verify_final.zkey`.
* Export the output to `schemas/proof.json` matching the frozen proof schema structure.
