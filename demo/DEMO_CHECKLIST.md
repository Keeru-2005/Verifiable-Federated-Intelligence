# 📋 VFI Phase 3 Review Demo Checklist

Use this checklist during tomorrow's evaluation to ensure smooth execution and confident presentation.

---

## 🕒 Before the Demo (Pre-Flight Check)

- [ ] **Terminal Setup:** Open two clean terminal windows in the project root:
  - Terminal 1: For running commands (`npm run demo`, `npm run metrics`, etc.)
  - Terminal 2: For running the dashboard (`npm run dashboard`)
- [ ] **Dependencies Checked:**
  - `python3 -c "import torch, flwr; print('PyTorch & Flower OK')"`
  - `node -v` (Node.js 18+ or 20+)
- [ ] **Dashboard Running:**
  - In Terminal 2, run: `npm run dashboard`
  - Verify [http://localhost:3000](http://localhost:3000) loads in your browser.
  - Verify [http://localhost:3000/regulator.html](http://localhost:3000/regulator.html) loads in your browser.
- [ ] **Screen Sizing:** Set browser zoom to 90% or 100% so all metrics, charts, and status cards fit comfortably on screen.

---

## 🎤 Live Demonstration (5–7 Minutes)

### 1. Problem Statement & Architecture (1 Minute)
- [ ] Explain the **Data Silo Paradox** in Anti-Money Laundering:
  - Banks cannot pool customer transaction records due to GDPR and banking secrecy laws.
  - VFI enables 4 banks to collaboratively train a shared model without ever sharing raw customer data.
- [ ] Point out the **Current Phase 3 Status** banner at the top of the dashboard:
  - State clearly: *"The core VFI pipeline is operational, while on-chain ZKP verification, GNN scaling, differential privacy and production-level enhancements remain under development."*

### 2. End-to-End Execution (1.5 Minutes)
- [ ] In Terminal 1, run the master demo:
  ```bash
  npm run demo
  ```
- [ ] Walk the examiners through the printed stages as they complete:
  - `[1/8] & [2/8]` Preprocessing, SMOTE balancing, and the 4 isolated bank data silos (~180k transactions each).
  - `[3/8] & [4/8]` 4-Bank federated learning convergence across 10 rounds reaching **99.67% F1-score**.
  - `[5/8] & [6/8]` Floating-point model weight quantization and Groth16 zk-SNARK proof generation with Poseidon hashing.
  - `[7/8]` Local cryptographic proof verification.
  - `[8/8]` Blockchain logging of federated training/audit information using the current smart-contract prototype.

### 3. Web Dashboard Walkthrough (2 Minutes)
- [ ] Switch to the browser at [http://localhost:3000](http://localhost:3000):
  - **Bank Admin Control Center:**
    - Show the **10 Training Rounds** progression table (Accuracy, Precision, Recall, F1).
    - Show the **Centralized vs. Federated Analysis** chart (`baseline_vs_federated.png`).
    - Show the **Bank Data Shard Distribution** cards (Bank 1 through 4 health).
- [ ] Click **Regulator View** or navigate to [http://localhost:3000/regulator.html](http://localhost:3000/regulator.html):
  - Show the **Deployed Contract Address** (`0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512`).
  - Show the **Immutable Ledger Audit Logs** table displaying round numbers, weight hashes, and verified timestamps.
  - Highlight how external regulatory auditors can independently verify compliance without accessing private banking databases.

### 4. Technical Deep-Dive & Robustness (1.5 Minutes)
- [ ] Run the graph neural network integration test to demonstrate PyTorch model compatibility:
  ```bash
  npm run test:graph
  ```
- [ ] (Optional if asked about failure modes) Mention or run the node dropout stress test:
  ```bash
  npm run test:stress
  ```
  - Explains quorum enforcement: Flower refuses partial/corrupt weight updates when a node dies mid-round.

---

## 🎯 Final Explanation & Phase 3 Roadmap (1 Minute)

- [ ] **Working Today:**
  - 4-Bank decentralized FL with Flower and PyTorch GraphAwareMLP.
  - 99.67% F1-score on imbalanced transaction topology.
  - Circom/Groth16 proof generation and local verification.
  - Polygon Amoy smart contract audit logging and dual dashboard.
- [ ] **Phase 3 Active Development:**
  - Direct on-chain EVM Groth16 pairing verifier contract (`PairingVerifier.sol`).
  - Dynamic GATv2 multi-hop graph attention for cross-bank transaction links.
  - Differential Privacy ($\epsilon, \delta$ DP-SGD) for formal privacy guarantees.
  - Automated regulatory PDF audit report generation.
