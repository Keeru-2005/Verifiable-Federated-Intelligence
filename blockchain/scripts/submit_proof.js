const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("=== Submitting zk-SNARK Proof to AMLVerifier Contract ===");

  // 1. Read the deployed addresses
  const deployedAddressesPath = path.join(__dirname, "../deployed_addresses.json");
  if (!fs.existsSync(deployedAddressesPath)) {
    throw new Error("deployed_addresses.json not found! Run deploy.js first.");
  }
  const deployedAddresses = JSON.parse(fs.readFileSync(deployedAddressesPath));
  console.log(`AMLVerifier contract address: ${deployedAddresses.AMLVerifier}`);

  // 2. Read proof.json
  const proofPath = path.join(__dirname, "../../schemas/proof.json");
  if (!fs.existsSync(proofPath)) {
    throw new Error("proof.json not found in schemas/ folder!");
  }
  const proofData = JSON.parse(fs.readFileSync(proofPath));
  const roundNumber = proofData.round_reference;
  if (!Number.isInteger(Number(roundNumber)) || proofData.publicSignals?.length !== 3) {
    throw new Error("Proof metadata is missing a valid round or the expected three public signals.");
  }
  if (String(proofData.publicSignals[1]) !== String(roundNumber)) {
    throw new Error("Proof public round number does not match round_reference.");
  }
  if (String(proofData.publicSignals[2]) !== String(proofData.total_samples)) {
    throw new Error("Proof public sample count does not match proof metadata.");
  }
  console.log(`Proof loaded. Target Round Reference: ${roundNumber}`);

  // 3. Read corresponding global_model_round_N.json metrics.
  //    MODEL_ROUND_PATH lets tests submit a fixture without overwriting the
  //    canonical leak-free round exports in schemas/.
  const modelPath = process.env.MODEL_ROUND_PATH
    ? path.resolve(process.env.MODEL_ROUND_PATH)
    : path.join(__dirname, `../../schemas/global_model_round_${roundNumber}.json`);
  if (!fs.existsSync(modelPath)) {
    throw new Error(`global_model_round_${roundNumber}.json not found in schemas/ folder!`);
  }
  const modelData = JSON.parse(fs.readFileSync(modelPath));
  if (Number(modelData.round) !== Number(roundNumber)) {
    throw new Error("Proof round does not match the selected global model artifact.");
  }
  const finalEvaluationPath = path.join(__dirname, "../../schemas/final_evaluation.json");
  if (!fs.existsSync(finalEvaluationPath)) {
    throw new Error("final_evaluation.json is required before recording test metrics on-chain.");
  }
  const finalEvaluation = JSON.parse(fs.readFileSync(finalEvaluationPath));
  if (Number(finalEvaluation.selected_round) !== Number(roundNumber) ||
      finalEvaluation.evaluation !== "final_test_once" ||
      finalEvaluation.run_id !== proofData.run_id ||
      modelData.run_id !== proofData.run_id) {
    throw new Error("Proof round does not match the single final-test evaluation record.");
  }
  const finalMetrics = finalEvaluation.metrics;
  console.log(`Model metadata loaded for round ${roundNumber}.`);

  // 4. Connect to the AMLVerifier contract (auto-deploying if current network has no bytecode)
  let contractAddress = deployedAddresses.AMLVerifier;
  const code = await ethers.provider.getCode(contractAddress);
  if (code === "0x" || code === "0x0") {
    console.log("[NOTE] Contract not found at address on this network. Deploying fresh contracts...");
    const Groth16VerifierFactory = await ethers.getContractFactory("Groth16Verifier");
    const verifierInst = await Groth16VerifierFactory.deploy();
    await verifierInst.waitForDeployment();
    const verifierAddr = await verifierInst.getAddress();

    const AMLVerifierFactory = await ethers.getContractFactory("AMLVerifier");
    const amlInst = await AMLVerifierFactory.deploy(verifierAddr);
    await amlInst.waitForDeployment();
    contractAddress = await amlInst.getAddress();
    console.log(`Deployed Groth16Verifier to ${verifierAddr} and AMLVerifier to ${contractAddress}`);

    // Update deployed_addresses.json
    deployedAddresses.Groth16Verifier = verifierAddr;
    deployedAddresses.AMLVerifier = contractAddress;
    fs.writeFileSync(deployedAddressesPath, JSON.stringify(deployedAddresses, null, 2));
  }

  const AMLVerifier = await ethers.getContractFactory("AMLVerifier");
  const amlVerifier = AMLVerifier.attach(contractAddress);

  // 5. Parse and format parameters
  const a = proofData.proof.pi_a.slice(0, 2);
  const b = [
    [proofData.proof.pi_b[0][1], proofData.proof.pi_b[0][0]],
    [proofData.proof.pi_b[1][1], proofData.proof.pi_b[1][0]]
  ];
  const c = proofData.proof.pi_c.slice(0, 2);
  const input = proofData.publicSignals;
  console.log(`Proved output-layer commitment: ${input[0]}`);
  console.log(`Proved Flower aggregation sample count: ${input[2]}`);

  // Compute a simple hash for model weights to represent on-chain
  const modelWeightsStr = JSON.stringify(modelData.weights);
  const modelWeightsHash = ethers.keccak256(ethers.toUtf8Bytes(modelWeightsStr));
  console.log(`Computed Model Weights Hash: ${modelWeightsHash}`);

  // Scale metrics (e.g. 0.9542 -> 9542)
  const scale = 10000;
  const accuracy = Math.round(finalMetrics.accuracy * scale);
  const precision = Math.round(finalMetrics.precision * scale);
  const recall = Math.round(finalMetrics.recall * scale);
  const f1 = Math.round(finalMetrics.f1 * scale);
  const clientCount = modelData.client_count;
  const circuitHash = proofData.circuit_hash;

  console.log(`Formatted metrics to submit:
  - Accuracy : ${accuracy / 100}% (${accuracy})
  - Precision: ${precision / 100}% (${precision})
  - Recall   : ${recall / 100}% (${recall})
  - F1-Score : ${f1 / 100}% (${f1})
  - Clients  : ${clientCount}
  - Circuit  : ${circuitHash}`);

  // 6. Submit the proof transaction
  console.log("Sending verifyAndRecordRound transaction...");
  const tx = await amlVerifier.verifyAndRecordRound(
    a,
    b,
    c,
    input,
    roundNumber,
    modelWeightsHash,
    accuracy,
    precision,
    recall,
    f1,
    clientCount,
    circuitHash
  );

  console.log(`Transaction sent! Hash: ${tx.hash}`);
  const receipt = await tx.wait();
  console.log(`Transaction confirmed in block: ${receipt.blockNumber}`);
  console.log(`Gas used for on-chain proof verification + round recording: ${receipt.gasUsed.toString()}`);

  // Retrieve verification result from contract
  const roundDetails = await amlVerifier.rounds(roundNumber);
  console.log(`
[COMPLETE] Verification Successful!
On-Chain Round Log Details:
----------------------------
Round Index  : ${roundDetails.roundNumber.toString()}
Weights Hash : ${roundDetails.modelWeightsHash}
Accuracy     : ${(Number(roundDetails.accuracy) / 100).toFixed(2)}%
F1-Score     : ${(Number(roundDetails.f1) / 100).toFixed(2)}%
Client Count : ${roundDetails.clientCount.toString()}
Record Time  : ${new Date(Number(roundDetails.timestamp) * 1000).toLocaleString()}
Submitter    : ${roundDetails.submitter}
`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Submission failed:", error);
    process.exit(1);
  });
