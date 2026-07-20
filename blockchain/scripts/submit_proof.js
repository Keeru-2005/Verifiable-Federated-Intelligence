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
  console.log(`Proof loaded. Target Round Reference: ${roundNumber}`);

  // 3. Read corresponding global_model_round_N.json metrics
  const modelPath = path.join(__dirname, `../../schemas/global_model_round_${roundNumber}.json`);
  if (!fs.existsSync(modelPath)) {
    throw new Error(`global_model_round_${roundNumber}.json not found in schemas/ folder!`);
  }
  const modelData = JSON.parse(fs.readFileSync(modelPath));
  console.log(`Model metadata loaded for round ${roundNumber}.`);

  // 4. Connect to the AMLVerifier contract
  const AMLVerifier = await ethers.getContractFactory("AMLVerifier");
  const amlVerifier = AMLVerifier.attach(deployedAddresses.AMLVerifier);

  // 5. Parse and format parameters
  const a = proofData.proof.pi_a.slice(0, 2);
  const b = [
    proofData.proof.pi_b[0],
    proofData.proof.pi_b[1]
  ];
  const c = proofData.proof.pi_c.slice(0, 2);
  const input = proofData.publicSignals;

  // Compute a simple hash for model weights to represent on-chain
  const modelWeightsStr = JSON.stringify(modelData.weights);
  const modelWeightsHash = ethers.keccak256(ethers.toUtf8Bytes(modelWeightsStr));
  console.log(`Computed Model Weights Hash: ${modelWeightsHash}`);

  // Scale metrics (e.g. 0.9542 -> 9542)
  const scale = 10000;
  const accuracy = Math.round(modelData.metrics.accuracy * scale);
  const precision = Math.round(modelData.metrics.precision * scale);
  const recall = Math.round(modelData.metrics.recall * scale);
  const f1 = Math.round(modelData.metrics.f1 * scale);
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

  // Retrieve verification result from contract
  const roundDetails = await amlVerifier.rounds(roundNumber);
  console.log(`
🎉 Verification Successful!
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
