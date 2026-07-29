const express = require('express');
const cors = require('cors');
const { ethers } = require('ethers');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3001;

// Minimal ABI for the required functions
const AMLVerifierABI = [
  "function getRoundNumbers() view returns (uint256[])",
  "function getRoundsCount() view returns (uint256)",
  "function owner() view returns (address)",
  "function getLedgerEntry(uint256 roundNumber) view returns (tuple(uint256 roundNumber, string modelWeightsHash, uint256 accuracy, uint256 precision, uint256 recall, uint256 f1, uint256 clientCount, uint256 timestamp, string circuitHash, address submitter))",
  "function verifyOnly(uint256[2] a, uint256[2][2] b, uint256[2] c, uint256[3] input) view returns (bool)",
  "event RoundVerified(uint256 indexed roundNumber, string modelWeightsHash, uint256 accuracy, uint256 precision, uint256 recall, uint256 f1, uint256 clientCount, uint256 timestamp, string circuitHash, address indexed submitter)"
];

let provider;
let contract;

async function setup() {
  const rpcUrl = process.env.RPC_URL || "http://127.0.0.1:8545";
  provider = new ethers.JsonRpcProvider(rpcUrl);

  const deployedAddressesPath = path.join(__dirname, 'deployed_addresses.json');
  let contractAddress;
  try {
    const data = fs.readFileSync(deployedAddressesPath, 'utf8');
    const addresses = JSON.parse(data);
    contractAddress = addresses.AMLVerifier || addresses.amlVerifier;
    if (!contractAddress) throw new Error("AMLVerifier address not found in JSON");
  } catch (error) {
    console.error("Error reading deployed_addresses.json:", error);
    process.exit(1);
  }

  contract = new ethers.Contract(contractAddress, AMLVerifierABI, provider);
  console.log(`Connected to contract at ${contractAddress}`);
}

setup();

// Helper to convert struct array to object
function parseRound(round) {
  return {
    roundNumber: round.roundNumber.toString(),
    modelWeightsHash: round.modelWeightsHash,
    accuracy: round.accuracy.toString(),
    precision: round.precision.toString(),
    recall: round.recall.toString(),
    f1: round.f1.toString(),
    clientCount: round.clientCount.toString(),
    timestamp: round.timestamp.toString(),
    circuitHash: round.circuitHash,
    submitter: round.submitter
  };
}

app.get('/api/rounds', async (req, res) => {
  try {
    const rounds = await contract.getRoundNumbers();
    res.json({ rounds: rounds.map(r => r.toString()) });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/rounds/:id', async (req, res) => {
  try {
    const roundNumber = req.params.id;
    const round = await contract.getLedgerEntry(roundNumber);
    res.json({ round: parseRound(round) });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/ledger', async (req, res) => {
  try {
    const roundNumbers = await contract.getRoundNumbers();
    const ledger = [];
    for (const id of roundNumbers) {
      const round = await contract.getLedgerEntry(id);
      ledger.push(parseRound(round));
    }
    res.json({ ledger });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/status', async (req, res) => {
  try {
    const address = await contract.getAddress();
    const owner = await contract.owner();
    const count = await contract.getRoundsCount();
    res.json({ address, owner, roundsCount: count.toString() });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/events', async (req, res) => {
  try {
    const filter = contract.filters.RoundVerified();
    const events = await contract.queryFilter(filter);
    const parsedEvents = events.map(e => ({
      transactionHash: e.transactionHash,
      blockNumber: e.blockNumber,
      args: {
        roundNumber: e.args[0].toString(),
        modelWeightsHash: e.args[1],
        accuracy: e.args[2].toString(),
        precision: e.args[3].toString(),
        recall: e.args[4].toString(),
        f1: e.args[5].toString(),
        clientCount: e.args[6].toString(),
        timestamp: e.args[7].toString(),
        circuitHash: e.args[8],
        submitter: e.args[9]
      }
    }));
    res.json({ events: parsedEvents });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/proof/verify', async (req, res) => {
  try {
    const { a, b, c, input } = req.body;
    if (!a || !b || !c || !input) {
      return res.status(400).json({ error: "Missing proof elements" });
    }
    const isValid = await contract.verifyOnly(a, b, c, input);
    res.json({ isValid });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Blockchain API server running on port ${PORT}`);
});
