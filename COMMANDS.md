# VFI — Master Commands Reference Guide

This document contains every command required to set up, preprocess data, train models, verify zero-knowledge proofs on the blockchain, run the enterprise web UI, and execute automated test suites.

---

## 📋 Table of Contents
1. [Environment & Installation](#1-environment--installation)
2. [End-to-End System Execution (Order-Wise)](#2-end-to-end-system-execution-order-wise)
3. [Automated Testing Suite (45 Tests)](#3-automated-testing-suite-45-tests)
4. [Alternative Quick Demos](#4-alternative-quick-demos)
5. [Useful Utility Commands](#5-useful-utility-commands)

---

## 1. Environment & Installation

```bash
# Clone the repository
git clone https://github.com/ananyalakshmi9/Verifiable-Federated-Intelligence.git
cd Verifiable-Federated-Intelligence

# Install Python requirements (PyTorch, Flower, NetworkX, Scikit-learn, etc.)
pip install -r requirements.txt

# Install root Node.js dependencies
npm install

# Install blockchain Hardhat dependencies
cd blockchain && npm install && cd ..

# Install dashboard dependencies
cd dashboard && npm install && cd ..
```

---

## 2. End-to-End System Execution (Order-Wise)

Follow these steps sequentially to run the full pipeline from raw data to web application:

### Step 1: Preprocess Dataset & Extract Graph Features
Processes `data/amlnet.csv`, calculates PageRank, degree centrality, clustering coefficients, trains the GAT encoder, balances classes via SMOTE (50:50), and partitions data into 4 bank silos:
```bash
# Full dataset run (all 1.09M raw transactions):
python3 data/preprocess.py

# Optional: Fast sample run (~30 seconds for quick testing):
python3 data/preprocess.py --sample-size 20000
```
*Outputs generated:*
- `data/processed_partition_best.csv` (264,532 balanced training rows)
- `fl_implementation/data/bank_1.csv` through `bank_4.csv` (66,133 rows per bank)
- `fl_implementation/data/global_test.csv` (33,172 holdout test transactions)

---

### Step 2: Federated Learning Training (Flower)

#### **Terminal 1: Start the Flower Aggregation Server**
```bash
python3 fl_implementation/server.py --rounds 10 --min-clients 4
```
*(Leave this running — it listens on port 8080)*

#### **Terminal 2: Launch the 4 Bank Clients**
Open a new terminal tab and run:
```bash
python3 fl_implementation/client.py --cid 1 & python3 fl_implementation/client.py --cid 2 & python3 fl_implementation/client.py --cid 3 & python3 fl_implementation/client.py --cid 4 &
```
*(Trains across 10 rounds using DP-SGD privacy, reaches 99.56% F1, and exports models to `schemas/global_model_round_*.json`)*

---

### Step 3: Start Local Blockchain Node

#### **Terminal 3: Start Hardhat Node**
```bash
cd blockchain && npx hardhat node
```
*(Leave this running — provides the local EVM JSON-RPC ledger at http://127.0.0.1:8545)*

---

### Step 4: Deploy Smart Contracts & Record ZK Proof

In **Terminal 2** (project root):

#### 4A. Deploy AML Verifier Contracts
```bash
npm run blockchain:deploy
```
*Deploys `Groth16Verifier.sol` and `AMLVerifier.sol` to the local network.*

#### 4B. Generate Groth16 Proof & Submit On-Chain
```bash
npm run blockchain:submit
```
*Quantizes weights, generates BN254 Groth16 proof, and verifies + records round on-chain in Block #3.*

---

### Step 5: Launch the Enterprise Web UI

In **Terminal 2**:
```bash
npm run dashboard
```

Open your browser to:
- **Bank Operator Console:** [http://localhost:3000](http://localhost:3000)
- **Regulatory Audit Portal:** [http://localhost:3000/regulator.html](http://localhost:3000/regulator.html)

---

## 3. Automated Testing Suite (45 Tests)

The repository includes a comprehensive 45-test suite covering data preprocessing, PyTorch models, DP-SGD privacy, Circom ZK circuits, Hardhat smart contracts, and backend REST APIs.

### Run ALL 45 Tests in One Command
```bash
npm test
```

### Run Individual Test Suites
```bash
# 1. Python Pytest Suite (27 tests: preprocessing, FL model, DP-SGD, ZK circuits)
pytest tests/ -v
# or
npm run test:python

# 2. Hardhat Solidity Contract Suite (11 tests: pairing precompile, replay protection, ZK verification)
cd blockchain && npx hardhat test
# or
npm run test:contracts

# 3. Backend Express API Suite (7 tests: REST endpoints, live blockchain RPC, concurrency)
node tests/test_backend_api.js
# or
npm run test:api
```

---

## 4. Alternative Quick Demos

### Automated Terminal Showcase Demo
Runs data verification, model aggregation, ZK witness generation, proof creation, and off-chain/on-chain checks in a single automated script:
```bash
npm run demo
# or
node demo/run_demo.js
```

---

## 5. Useful Utility Commands

```bash
# Verify on-chain ledger records via CLI:
cd blockchain && npx hardhat test

# Check dashboard backend health:
curl http://localhost:3000/api/status
curl http://localhost:3000/api/rounds
curl http://localhost:3000/api/blockchain

# Verify Circom inputs:
npm run zk:input

# Generate ZK proof directly:
node zkp/generate_proof.js
```
