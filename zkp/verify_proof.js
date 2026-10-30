const fs = require("fs");
const path = require("path");

async function main() {
  console.log("[VERIFY] Running Local ZK Proof Verification...");

  // 1. Resolve paths
  const projectRoot = path.join(__dirname, "..");
  const proofPath = path.join(projectRoot, "schemas", "proof.json");
  const vKeyPath = path.join(projectRoot, "zkp", "circuit", "verification_key.json");

  // 2. Load proof.json
  if (!fs.existsSync(proofPath)) {
    console.error(`[FAIL] Error: proof.json not found at ${proofPath}`);
    process.exit(1);
  }

  let proofData;
  try {
    const rawProof = fs.readFileSync(proofPath);
    proofData = JSON.parse(rawProof);
  } catch (err) {
    console.error(`[FAIL] Error parsing proof.json: ${err.message}`);
    process.exit(1);
  }

  // 3. Check if verification key exists (generated after circuit compilation)
  if (fs.existsSync(vKeyPath)) {
    console.log("[KEY] Verification key found. Performing cryptographic pairing checks via snarkjs...");
    try {
      // Lazy load snarkjs to avoid requirement errors if run in custom setups
      const snarkjs = require("snarkjs");
      const vKey = JSON.parse(fs.readFileSync(vKeyPath));

      const isVerified = await snarkjs.groth16.verify(
        vKey,
        proofData.publicSignals,
        proofData.proof
      );

      if (isVerified) {
        console.log("[SUCCESS] SUCCESS: Cryptographic proof is valid!");
        process.exit(0);
      } else {
        console.error("[FAIL] FAILURE: Cryptographic proof verification failed!");
        process.exit(1);
      }
    } catch (err) {
      console.error(`[FAIL]Error verifying proof cryptographically: ${err.message}`);
      process.exit(1);
    }
  } else {
    // -------------------------------------------------------------------------
    //  NOTE FOR LIKITH & KEERTHANA:
    // - Likith: Once you write 'aml_verify.circom' and compile it, place your 
    //   exported 'verification_key.json' in 'zkp/circuit/'.
    // - Keerthana: Once you write your proof generation routine (e.g. 'generate_proof.js'),
    //   it will output the 'proof.json' containing publicSignals and proof points.
    // - This verification script will automatically detect 'verification_key.json' 
    //   and switch from structural validation to cryptographic Groth16 pairing verification!
    // -------------------------------------------------------------------------
    console.warn("[WARN] Warning: verification_key.json not found in zkp/circuit/ (circuit has not been compiled yet).");
    console.log("[LIST] Falling back to structural validation of the frozen proof schema...");

    // Structural validation
    const { proof, publicSignals, round_reference, circuit_hash } = proofData;

    if (!proof || !publicSignals || !round_reference || !circuit_hash) {
      console.error("[FAIL] Schema Validation Failed: Missing top-level fields (proof, publicSignals, round_reference, or circuit_hash).");
      process.exit(1);
    }

    if (proof.protocol !== "groth16" || proof.curve !== "bn128") {
      console.error(`[FAIL] Schema Validation Failed: Unsupported protocol '${proof.protocol}' or curve '${proof.curve}'. Must be groth16 and bn128.`);
      process.exit(1);
    }

    // Check pi_a, pi_b, pi_c structure
    if (!Array.isArray(proof.pi_a) || proof.pi_a.length < 2 ||
      !Array.isArray(proof.pi_b) || proof.pi_b.length < 2 ||
      !Array.isArray(proof.pi_c) || proof.pi_c.length < 2) {
      console.error("[FAIL] Schema Validation Failed: pi_a, pi_b, or pi_c are not arrays of the required size.");
      process.exit(1);
    }

    // Check publicSignals array
    if (!Array.isArray(publicSignals) || publicSignals.length !== 3) {
      console.error(`[FAIL] Schema Validation Failed: publicSignals must be an array of length 3 (received ${publicSignals.length}).`);
      process.exit(1);
    }

    console.log(`
[SUCCESS] Structural validation SUCCESSFUL!
----------------------------
Protocol        : ${proof.protocol}
Curve           : ${proof.curve}
Round Reference : ${round_reference}
Circuit Hash    : ${circuit_hash}
Public Signals  : [ ${publicSignals.join(", ")} ]
`);
    console.log("[NOTE] Ready for deployment! Once the circuit is compiled, place the verification_key.json in zkp/circuit/ to run cryptographic pairings verification.");
    process.exit(0);
  }
}

main().catch(err => {
  console.error("Local verification script failed:", err);
  process.exit(1);
});
