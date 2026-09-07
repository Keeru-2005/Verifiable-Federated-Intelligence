# 🛡️ Verifiable Federated Intelligence (VFI)
## Capstone Phase 3 Review Demonstration Guide

> **Project Title:** Decentralized, Privacy-Preserving Anti-Money Laundering (AML) Intelligence with Zero-Knowledge Proof Verification & On-Chain Auditability  
> **Target Review:** Capstone Phase 3 Progress Review (5–7 Minute Evaluation)

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
  Local Groth16 Proof Verification (bn128 curve)
        ↓
  Blockchain logging of federated training/audit information using the current smart-contract prototype
        ↓
  Live Monitoring Web Dashboard (Bank Admin & Regulator Views)
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

## 5. Current Development Status

> *"The core VFI pipeline is operational, while on-chain ZKP verification, GNN scaling, differential privacy and production-level enhancements remain under development."*

| Domain | Operational Core (Working) | Under Development (Phase 3 Roadmap) |
| :--- | :--- | :--- |
| **Data Engineering** | SMOTE balancing, PCA (32 features), temporal graph features, 4 bank stratification | Automated streaming Kafka ingest |
| **Federated Learning** | Flower FL, `GraphAwareMLP`, `EarlyStoppingFedAvg`, 10-round convergence (99.67% F1) | Heterogeneous non-IID dynamic GATv2 layers |
| **Privacy** | Complete data locality (zero customer PII shared) | Differential Privacy ($\epsilon, \delta$ DP-SGD noise) |
| **Zero-Knowledge** | Weight quantization (`1e6` scale), `aml_verify.circom`, local Groth16 verification | Direct on-chain EVM pairing verifier contract |
| **Blockchain** | `AMLVerifier.sol`, Polygon Amoy deployment, immutable round logging | Multi-signature validator quorum contract |
| **Dashboard** | Dual-role Bank Admin & Regulator UI, live metrics, visual charts | WebSocket streaming & JWT role authentication |

---

## 6. How to Stop & Reset the Demo

* To stop the dashboard server, press `Ctrl + C` in the terminal where it was started.
* To reset temporary test proofs:
  ```bash
  git checkout schemas/proof.json input.json
  ```
