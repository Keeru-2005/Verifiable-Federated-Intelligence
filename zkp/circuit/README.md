# VFI aggregation proof circuit

## Statement proved

The checked-in `aml_verify.circom` circuit proves a sample-count-weighted integer-average relation for the configured 33 quantized output-layer parameters and produces a Poseidon commitment. The circuit inputs bind the public commitment, round number, and total sample count. The selected output layer is the classifier's `Linear(32, 1)` weight vector plus bias.

The circuit does **not** prove that client training was performed correctly, that the reported client counts or data are truthful, that the complete model was correctly aggregated, or that the model has a stated accuracy. The Solidity contract stores metrics and a model hash as metadata; they are not all constrained by this circuit. This is an aggregation-integrity proof of limited scope, not a proof of model correctness.

The input-generation code uses fixed-point quantization and a positive offset to encode signed floating-point weights into nonnegative field elements. This encoding and its bounds are implementation assumptions; do not describe the circuit as proving the original floating-point computation.

## Build artifacts

The repository includes compiled circuit artifacts for its local demo. To rebuild after changing `aml_verify.circom`, use the compatible Circom and SnarkJS versions recorded by the project, regenerate the R1CS/WASM, run the trusted-setup procedure for the new circuit, export the verification key and verifier contract, and rerun proof-generation and tampering tests. The resulting verification key must match the exact circuit version used to generate the proof.

Typical compile command from this directory:

```bash
circom aml_verify.circom --r1cs --wasm --sym -l ../../node_modules
```

Proof generation and verification are orchestrated from the project scripts; consult `../generate_proof.js`, `../verify_proof.js`, and `../../blockchain/scripts/submit_proof.js`. Preserve the circuit hash, verification-key hash, proof input, public signals, and run record together.

## Trusted setup

Groth16 uses setup material tied to the circuit. Any circuit edit requires new setup artifacts and a newly compiled verifier. The repository setup files are suitable for reproducing the prototype only; do not claim a production ceremony or audited setup unless that process and its participants are separately documented.

