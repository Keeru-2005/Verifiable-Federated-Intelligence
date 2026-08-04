const express = require("express");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const { ethers } = require("ethers");

const app = express();
const PORT = process.env.PORT || 3000;

const projectRoot = path.join(__dirname, "..");
const schemasDir = path.join(projectRoot, "schemas");
const visualizationsDir = path.join(projectRoot, "visualizations");
const blockchainDir = path.join(projectRoot, "blockchain");
const evaluationDir = path.join(projectRoot, "evaluation");

// In-memory pipeline status
let pipelineStatus = {
  status: "idle", // idle, running, success, error
  step: "Ready to start",
  logs: [],
  targetRound: 1
};

app.use(express.json());
app.use(express.static(__dirname));
app.use("/visualizations", express.static(visualizationsDir));

// Endpoint to get pipeline status
app.get("/api/status", (req, res) => {
  res.json(pipelineStatus);
});

// Endpoint to trigger the pipeline run
app.post("/api/run", (req, res) => {
  if (pipelineStatus.status === "running") {
    return res.status(400).json({ error: "Pipeline is already running." });
  }

  pipelineStatus.status = "running";
  pipelineStatus.step = "Starting pipeline coordination...";
  pipelineStatus.logs = ["🚀 Pipeline launch initiated."];
  
  // First run the baseline comparison training to make sure we have comparison data and checkpoints
  pipelineStatus.step = "Running centralized baseline comparison training...";
  pipelineStatus.logs.push("🏃 Training baseline model on pooled dataset...");
  
  const pyProcess = spawn("python", ["evaluation/compare_baseline.py"], { cwd: projectRoot });
  
  pyProcess.stdout.on("data", (data) => {
    const lines = data.toString().split("\n");
    lines.forEach(line => {
      if (line.trim()) {
        pipelineStatus.logs.push(`[Python] ${line.trim()}`);
      }
    });
  });

  pyProcess.stderr.on("data", (data) => {
    pipelineStatus.logs.push(`[Python Error] ${data.toString().trim()}`);
  });

  pyProcess.on("close", (code) => {
    if (code !== 0) {
      pipelineStatus.status = "error";
      pipelineStatus.step = "Centralized baseline training failed.";
      pipelineStatus.logs.push(`❌ Centralized baseline script exited with code ${code}.`);
      return;
    }
    
    // Now trigger the coordinate.js script to run the ledger verification
    pipelineStatus.step = "Running ZK Proof & Ledger coordination script...";
    pipelineStatus.logs.push("⛓️ Starting smart contract coordination layer...");
    
    const coordProcess = spawn("npx", ["hardhat", "run", "scripts/coordinate.js", "--network", "localhost"], { cwd: blockchainDir });
    
    coordProcess.stdout.on("data", (data) => {
      const lines = data.toString().split("\n");
      lines.forEach(line => {
        if (line.trim()) {
          pipelineStatus.logs.push(line.trim());
          if (line.includes("Identified latest aggregated FL round")) {
            const match = line.match(/Round (\d+)/);
            if (match) pipelineStatus.targetRound = parseInt(match[1]);
          }
          if (line.includes("⚙️ Extracting")) pipelineStatus.step = "Extracting client weights...";
          if (line.includes("⚙️ Generating ZK")) pipelineStatus.step = "Generating ZK inputs...";
          if (line.includes("⚙️ Compiling proof")) pipelineStatus.step = "Compiling ZK Proof...";
          if (line.includes("🏃 Submitting proof")) pipelineStatus.step = "Submitting proof to Polygon Amoy Ledger...";
          if (line.includes("🔗 Ledger Confirmation")) pipelineStatus.step = "Verifying read-back confirmation...";
        }
      });
    });

    coordProcess.stderr.on("data", (data) => {
      pipelineStatus.logs.push(`[Hardhat Error] ${data.toString().trim()}`);
    });

    coordProcess.on("close", (coordCode) => {
      if (coordCode === 0) {
        pipelineStatus.status = "success";
        pipelineStatus.step = "E2E Pipeline Executed Successfully!";
        pipelineStatus.logs.push("🎉 Loop closed! Verification saved on blockchain and read back successfully.");
      } else {
        pipelineStatus.status = "error";
        pipelineStatus.step = "Ledger submission failed.";
        pipelineStatus.logs.push(`❌ Hardhat script exited with code ${coordCode}.`);
      }
    });
  });

  res.json({ message: "Pipeline started." });
});

// Endpoint to get all round details from schema folder
app.get("/api/rounds", (req, res) => {
  try {
    const files = fs.readdirSync(schemasDir);
    const roundFiles = files.filter(f => f.startsWith("global_model_round_") && f.endsWith(".json") && !f.includes("_N.json"));
    
    const rounds = roundFiles.map(f => {
      const filePath = path.join(schemasDir, f);
      const data = JSON.parse(fs.readFileSync(filePath, "utf8"));
      return {
        round: data.round,
        metrics: data.metrics,
        client_count: data.client_count || 4,
        timestamp: data.timestamp
      };
    });
    
    rounds.sort((a, b) => a.round - b.round);
    res.json(rounds);
  } catch (err) {
    res.status(500).json({ error: "Failed to read rounds: " + err.message });
  }
});

// Endpoint to get comparison results
app.get("/api/baseline", (req, res) => {
  const comparisonPath = path.join(evaluationDir, "comparison_results.json");
  if (fs.existsSync(comparisonPath)) {
    try {
      const data = JSON.parse(fs.readFileSync(comparisonPath, "utf8"));
      res.json(data);
    } catch (err) {
      res.status(500).json({ error: "Failed to read comparison: " + err.message });
    }
  } else {
    res.json({ centralized: [], federated: [] });
  }
});

// Endpoint to read round details from blockchain ledger
app.get("/api/blockchain", async (req, res) => {
  const deployedAddressesPath = path.join(blockchainDir, "deployed_addresses.json");
  if (!fs.existsSync(deployedAddressesPath)) {
    return res.status(404).json({ error: "Smart contracts not deployed. Run pipeline first." });
  }

  try {
    const deployedAddresses = JSON.parse(fs.readFileSync(deployedAddressesPath, "utf8"));
    const amlVerifierAddress = deployedAddresses.AMLVerifier;
    
    // Load ABI from hardhat artifacts
    const artifactPath = path.join(blockchainDir, "artifacts/contracts/AMLVerifier.sol/AMLVerifier.json");
    if (!fs.existsSync(artifactPath)) {
      return res.status(404).json({ error: "Contract artifacts not compiled." });
    }
    
    const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
    const abi = artifact.abi;
    
    // Connect to hardhat local node
    const provider = new ethers.JsonRpcProvider("http://127.0.0.1:8545");
    const contract = new ethers.Contract(amlVerifierAddress, abi, provider);
    
    // Retrieve registered round numbers
    const roundNumbers = await contract.getRoundNumbers();
    const rounds = [];
    
    for (const roundNum of roundNumbers) {
      const details = await contract.rounds(roundNum);
      rounds.push({
        roundNumber: details.roundNumber.toString(),
        modelWeightsHash: details.modelWeightsHash,
        accuracy: (Number(details.accuracy) / 100).toFixed(2),
        precision: (Number(details.precision) / 100).toFixed(2),
        recall: (Number(details.recall) / 100).toFixed(2),
        f1: (Number(details.f1) / 100).toFixed(2),
        clientCount: details.clientCount.toString(),
        timestamp: new Date(Number(details.timestamp) * 1000).toISOString(),
        circuitHash: details.circuitHash,
        submitter: details.submitter
      });
    }
    
    res.json({
      address: amlVerifierAddress,
      rounds: rounds
    });
  } catch (err) {
    console.error("❌ /api/blockchain Error details:", err);
    res.status(500).json({ error: "Failed to fetch ledger: " + err.message });
  }
});

app.listen(PORT, () => {
  console.log(`🌍 Dashboard Express server running at http://localhost:${PORT}`);
});
