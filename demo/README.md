# 🛡️ Verifiable Federated Intelligence (VFI)
## Capstone 80% Milestone Review Demonstration Guide

> **Project Title:** Decentralized, Privacy-Preserving Anti-Money Laundering (AML) Intelligence with Zero-Knowledge Proof Verification & On-Chain Auditability  
> **Target Review:** Capstone 80% Milestone Review (5–7 Minute Evaluation)

---

## 1. What the Demonstration Shows

This demonstration validates the complete, integrated VFI pipeline across all architectural tiers without relying on synthetic mocks:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             END-TO-END PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘
  Transaction Data (amlnet.csv)
        ↓
  Data Preprocessing & SMOTE Balancing (preprocess.py)
        ↓
  Graph Feature Extraction (PageRank, Degree, GAT Embeddings)
        ↓
  4 Decentralized Bank Silos (bank_1.csv to bank_4.csv)
        ↓
  Local Federated Training (GraphAwareMLP in PyTorch)
        ↓
  Flower Aggregation Server (EarlyStoppingFedAvg)
        ↓
  Global Aggregated AML Model (99.67% F1-Score)
        ↓
  Model Weight Export & Finite-Field Quantization
        ↓
  Circom zk-SNARK Circuit & Poseidon Hash Commitment
        ↓
  Dual-Layer ZK Verification (Local Groth16 & EVM On-Chain Smart Contract)
        ↓
  On-Chain Smart Contract Audit Logging (AMLVerifier.sol on Polygon Amoy)
        ↓
  Differential Privacy Bounds ((ε, δ)-DP Calibrated Noise)
        ↓
  Live Dual-Role Web Dashboards (Bank Admin & Regulator Views)
```

---

## 2. Prerequisites & Environment

* **Node.js:** `^18.0` or `^20.0` (Node 20+ compatible)
* **Python:** `^3.10` or `3.12`
* **Core Python Packages:** `torch`, `flwr`, `scikit-learn`, `pandas`, `numpy`
* **Ports Used:**
  * `3000`: Live Web Dashboard (Express)
  * `8080`: Flower FL Aggregator Server (during live FL runs)
  * `8545`: Local Ethereum/Polygon RPC Node (optional fallback)

---

## 3. Quickstart: Running Tomorrow's Demo

### A. Run Master Demonstration (Recommended: ~10 seconds)
From the project root:
```bash
npm run demo
```
*Or directly with Node:*
```bash
node demo/run_demo.js
```
*Or on macOS/Linux:*
```bash
./demo/run_demo.sh
```
*Or on Windows:*
```cmd
demo\run_demo.bat
```

### B. Launch Live Web Dashboard
In a separate terminal window:
```bash
npm run dashboard
```
*Or:*
```bash
cd dashboard && npm start
```
Then open in your browser:
* 🏦 **Bank Admin View:** [http://localhost:3000](http://localhost:3000)
* 📜 **Regulator Audit Ledger:** [http://localhost:3000/regulator.html](http://localhost:3000/regulator.html)

---

## 4. Demonstrating Specific Subsystems

### 1. How to Show Data Preprocessing & 4 Bank Nodes
Point examiners to the data partitioning files:
```bash
ls -lh fl_implementation/data/
```
* **Bank Node 1:** `fl_implementation/data/bank_1.csv` (~180,272 records, 106 MB)
* **Bank Node 2:** `fl_implementation/data/bank_2.csv` (~180,272 records, 106 MB)
* **Bank Node 3:** `fl_implementation/data/bank_3.csv` (~180,271 records, 106 MB)
* **Bank Node 4:** `fl_implementation/data/bank_4.csv` (~180,271 records, 106 MB)
* **Global Holdout Test Set:** `fl_implementation/data/global_test.csv` (~180,272 records)

*Explanation:* Raw transaction data remains strictly inside each bank's local filesystem boundary; only gradients are shared.

### 2. How to Show FL Model Convergence & Metrics
Run the benchmark evaluator to display all 10 federated rounds dynamically:
```bash
npm run metrics
# or: python3 evaluation/benchmark_metrics.py
```
* **Round 1:** Accuracy: `98.03%`, F1-Score: `0.9800`
* **Round 5:** Accuracy: `99.36%`, F1-Score: `0.9936`
* **Round 10 (Converged):** Accuracy: `99.67%`, Precision: `99.48%`, Recall: `99.87%`, F1-Score: `0.9967`

### 3. How to Show ZK-SNARK Weight Quantization & Local Verification
Run the circuit input quantization and verification:
```bash
npm run zk:input
npm run zk:verify
# or: node zkp/verify_proof.js
```
* Shows quantization of 33 neural parameters across 4 clients (721,086 total samples).
* Performs Groth16 structural and pairing validation against the frozen schema:
  * Protocol: `groth16`
  * Curve: `bn128`
  * Circuit Hash: `0x19f12a04781bba1340483dda53dc493243c57b`

### 4. How to Show Blockchain Logging & Smart Contracts
Submit the verified proof and metrics to the smart contract:
```bash
npm run blockchain:submit
# or: cd blockchain && npx hardhat run scripts/submit_proof.js
```
* **Contract:** `blockchain/contracts/AMLVerifier.sol` & `Groth16Verifier.sol`
* **Output:** Displays confirmed block number, transaction hash, submitter address, and on-chain record log.

### 5. How to Run Integration & Stress Tests
Show quorum enforcement and node failure handling:
```bash
npm run test:stress
# or: python3 evaluation/stress_test.py
```
* Kills Bank Node 4 mid-training. The Flower server strictly enforces quorum (`min_available_clients=4`) and halts cleanly rather than aggregating corrupt partial weights.

Show Graph Neural Network integration:
```bash
npm run test:graph
# or: python3 fl_implementation/test_graph_integration.py
```
* Validates GraphEncoder, GAT embeddings, attention edge pruning, and `GraphAwareMLP` tensor compatibility.

---

## 5. Milestone Status: 80% Work Completed

> *"80% of the Verifiable Federated Intelligence (VFI) platform is fully operational, integrating decentralized FL, graph neural networks, zero-knowledge proofs, on-chain smart contract verification, differential privacy, and real-time regulatory compliance."*

| Domain | Completed & Demonstrated (80% Scope) | Remaining Scope (Final 20% Delivery) |
| :--- | :--- | :--- |
| **Data Engineering** | SMOTE class balancing (50:50/30:70), StandardScaler, 32 PCA features, 4 bank stratification | Real-time continuous Kafka streaming ingest |
| **Graph Neural Networks** | PageRank, node degrees, clustering coefficients, and GAT embeddings linked to `GraphAwareMLP` | Dynamic multi-hop cross-bank edge updates |
| **Federated Learning** | 4-Bank decentralized Flower FL, `EarlyStoppingFedAvg`, 10-round convergence (**99.67% F1**, 99.87% recall) | Multi-tier heterogeneous client scheduling |
| **Privacy Preservation** | Complete data locality + Calibrated Differential Privacy (DP-SGD gradient clipping, ε=3.84, δ=1e-5) | Zero-Knowledge Private Information Retrieval (PIR) |
| **Zero-Knowledge Proofs** | Finite-field parameter quantization (`1e6` scale), `aml_verify.circom`, Groth16 proof generation (~3.52s) | GPU/FPGA hardware prover acceleration |
| **Dual ZK Verification** | Local `snarkjs` pairing validation & direct EVM `Groth16Verifier.sol` alt_bn128 pairing check | Recursive SNARK proof batching |
| **Blockchain Auditability** | `AMLVerifier.sol`, Polygon Amoy deployment, immutable round metrics & weight hash logs | Multi-jurisdiction consortium multi-sig contract |
| **Live Dashboards** | Dual-role Bank Admin & Regulator UI, real-time metrics progression, and contract ledger sync | Institutional HSM / Intel SGX key management |

---

## 6. How to Stop & Reset the Demo

* To stop the dashboard server, press `Ctrl + C` in the terminal where it was started.
* To reset temporary test proofs:
  ```bash
  git checkout schemas/proof.json input.json
  ```
