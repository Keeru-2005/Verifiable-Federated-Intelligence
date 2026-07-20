# Verifiable Federated Intelligence

**A Privacy-Preserving Framework for Decentralized AML Detection via Structural Embeddings and zk-SNARKs**

This framework pioneers a trustless environment where disjointed financial networks collaboratively train an Anti-Money Laundering (AML) schema without surrendering topological data autonomy. Every iteration transmitted is authenticated via zk-SNARKs mathematically verified immutably on the Polygon Amoy Testnet.

## 👥 Authors
- A R Keerthana 
- Ananya Lakshmi 
- Kavan Reddy 
- Likhith Kumar 

## 🏛️ System Architecture

```text
[Bank Node A: Docker]  [Bank Node B: Docker]  [Bank Node C: Docker]  [Bank Node D: Docker]
         |                      |                      |                      |
         └──── Local Weights ───┴──── Local Weights ───┴──── Local Weights ───┘
                                          │
                          ┌───────────────▼───────────────────┐
                          │  COORDINATION LAYER (Federated Hub)│
                          │   Flower Server: FedAvg Orchestrator│
                          │   + MLP Model Architecture         │
                          └───────────────┬───────────────────┘
                                          │ Aggregated State
                          ┌───────────────▼───────────────────┐
                          │  VERIFICATION LAYER (zk-SNARK)    │
                          │   Circom Circuit Compiler          │
                          │   → snarkjs: Proof Generation      │
                          └───────────────┬───────────────────┘
                                          │ Proof + Weights
                          ┌───────────────▼───────────────────┐
                          │  SETTLEMENT LAYER (Blockchain)     │
                          │   Polygon Amoy Smart Contract      │
                          │   → Global AML Ledger              │
                          └───────────────────────────────────┘
```

## 📋 Prerequisites
- **Python:** `3.10+`
- **Docker:** `24.0+`
- **Node.js:** `18.0+`
- **SNARK Tools:** `circom`, `snarkjs`
- **Blockchain:** Metamask + Polygon Amoy MATIC funds

## 🚀 Step-by-Step Setup

1. **Clone & Environment Setup:**
   ```bash
   git clone <repo_url>
   cd capstone-aml-federated
   pip install -r requirements.txt
   cp .env.example .env # Configure your private keys and limits here
   ```

2. **Dataset Federation (Phase 1):**
   ```bash
   python data/download_dataset.py
   python data/preprocess.py
   python fl_implementation/split_into_banks.py
   ```

3. **Federated Orchestration (Phase 2):**
   ```bash
   # Run via Docker
   docker-compose up --build
   ```

4. **Cryptographic Proofs & Amoy Settlement:**
   ```bash
   cd zkp/circuit && ./compile_circuit.sh
   # Deploy
   cd ../../blockchain && npx hardhat run scripts/deploy.js --network amoy
   # Verify on-chain
   python pipeline/run_full_pipeline.py
   ```

## ⚠️ Known Limitations
- ZKP generation restricts neural dimensions to PCA-condensed vectors.
- Amoy deployment subject to faucet constraints regarding MATIC throughput. 

## 📖 References
Incorporates topological ML paradigms adapted from *AMLNet* architecture methodologies. Constraints implemented in compliance with standardized mathematical privacy frameworks.
