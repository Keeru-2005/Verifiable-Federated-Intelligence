# 🧠 Algorithms & Pseudocode: Verifiable Federated Intelligence (VFI)
## Ready-to-Use Slide Content for Review Presentation

---

### 📑 SLIDE 1: Core Algorithmic Architecture
**Slide Title:** Core Algorithmic Framework: Privacy-Preserving Verifiable AML

#### Bullet Points for Slide:
* **Hybrid Topological Learning:** Combines **Graph Attention Networks (GAT)** for transaction network structure with **GraphAwareMLP** for tabular risk scoring.
* **Decentralized Coordination:** **EarlyStoppingFedAvg** — Parameter aggregation weighted by shard size with patience-based saturation convergence.
* **Calibrated Privacy Guarantee:** **Differentially Private SGD (DP-SGD)** with bounded gradient $L_2$ norm clipping ($C=1.0$) and calibrated Gaussian noise injection ($\sigma=0.5$).
* **Cryptographic Soundness:** **Poseidon Hash Commitment** & **Groth16 zk-SNARK** proving that aggregated weights match verifiable client updates without exposing parameters.
* **On-Chain EVM Verification:** Bilinear pairing check on the `alt_bn128` elliptic curve executed inside Solidity (`AMLVerifier.sol` & `Groth16Verifier.sol`).

---

### 📑 SLIDE 2: Mathematical Formulations
**Slide Title:** Mathematical Foundations

#### 1. Graph Attention Mechanism (Local Node Representation)
$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}h_i \parallel \mathbf{W}h_j]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}h_i \parallel \mathbf{W}h_k]\right)\right)}$$

#### 2. Federated Aggregation with DP-SGD
$$\bar{g}_k = \frac{g_k}{\max\left(1, \frac{\|g_k\|_2}{C}\right)}, \quad \tilde{g}_k = \bar{g}_k + \mathcal{N}\left(0, \sigma^2 C^2 \mathbf{I}\right)$$
$$W_{t+1} = \sum_{k=1}^{K} \frac{n_k}{N} \left(W_t - \eta \tilde{g}_k\right)$$

#### 3. Finite-Field Parameter Quantization
$$q_i = \left(\lfloor w_i \times 10^6 \rfloor + 2^{32}\right) \pmod p \quad \text{where } p = 21888242871839275222246405745257275088548364400416034343698204186575808495617$$

#### 4. Groth16 On-Chain Bilinear Pairing Check
$$e(A, B) = e(\alpha, \beta) \cdot e\left(\sum_{i=0}^l x_i \cdot \text{IC}_i, \gamma\right) \cdot e(C, \delta)$$

---

### 📑 SLIDE 3: End-to-End Workflow Pseudocode
**Slide Title:** Algorithmic Workflow: Verifiable Federated Intelligence

```text
Algorithm 1: Verifiable Federated Intelligence (VFI) Master Workflow
---------------------------------------------------------------------------------------------
Input  : Isolated Bank Datasets {D_1, D_2, ..., D_K}, Total Rounds T, Quorum K, Privacy Budget (ε, δ)
Output : Cryptographically Verified Global Model W*, Immutable On-Chain Audit Record Tx_Hash
---------------------------------------------------------------------------------------------
1:  // Phase 1: Local Preprocessing & Topological Graph Encoding
2:  for each Bank Node k ∈ {1, ..., K} in parallel do
3:      D_k_bal ← SMOTE_Oversample(D_k, ratio=0.5)
4:      G_k ← Construct_Temporal_Transaction_Graph(D_k_bal)
5:      H_k ← Extract_Topological_Features(G_k) // PageRank, Degrees, Clustering
6:      Z_k ← GAT_Encoder(G_k, H_k)            // Compute Graph Attention Embeddings
7:      X_k ← Concat(D_k_bal_features, Z_k)
8:  end for
9:
10: // Phase 2: Decentralized Federated Learning with DP-SGD
11: Initialize global weights W_0
12: for round t = 0 to T-1 do
13:     for each Bank Node k ∈ {1, ..., K} in parallel do
14:         g_k ← Compute_Gradients(Loss(GraphAwareMLP(X_k; W_t)))
15:         g_k_clipped ← g_k / max(1, ||g_k||_2 / C)               // DP Gradient Clipping
16:         g_k_private ← g_k_clipped + Normal(0, σ^2 * C^2 * I)     // Gaussian Perturbation
17:         W_t_k ← W_t - η * g_k_private
18:         Send W_t_k, sample_count n_k to FL Server
19:     end for
20:
21:     // Server EarlyStoppingFedAvg Aggregation
22:     W_(t+1) ← ∑_(k=1)^K (n_k / N) * W_t_k
23:     val_F1 ← Evaluate_Global_Model(W_(t+1), Global_Holdout_Set)
24:     if val_F1 saturated for patience P rounds then break
25: end for
26:
27: // Phase 3: ZK Proof Generation & Finite Field Commitment
28: Q(W) ← FixedPointQuantize(W_(t+1), scale=10^6, offset=2^32)
29: Circuit_Hash ← Poseidon_Hash(Q(W))
30: Public_Signals ← [Round_ID, Circuit_Hash, Accuracy, F1]
31: (Proof_π, Public_Signals) ← Groth16_Prove(Circuit_Wasm, ZKey, Q(W), Public_Signals)
32:
33: // Phase 4: Dual Verification & Polygon Amoy Ledger Recording
34: Verify_Local ← Groth16_Verify(VK, Proof_π, Public_Signals)
35: assert Verify_Local == TRUE
36:
37: Tx_Hash ← AMLVerifier.verifyAndRecordRound(
38:               Proof_π.A, Proof_π.B, Proof_π.C, Public_Signals,
39:               Round_ID, Keccak256(W_(t+1)), Accuracy, F1, K, Circuit_Hash
40:           )
41: Broadcast To Web Dashboard (Bank Admin & Regulator Views)
42: return W_(t+1), Tx_Hash
---------------------------------------------------------------------------------------------
```

---

### 📑 SLIDE 4: Subsystem Logical Breakdown
**Slide Title:** 4-Tier Pipeline Architecture

| Subsystem | Input | Primary Algorithm / Method | Output |
| :--- | :--- | :--- | :--- |
| **Tier 1: Data Silos** | Raw transactions (`amlnet.csv`) | SMOTE + StandardScaler + GAT Attention Embeddings | Graph-enriched normalized tensors $X_k$ |
| **Tier 2: Federated Engine** | Bank tensors $X_k$, Global weights $W_t$ | DP-SGD + `EarlyStoppingFedAvg` on Flower | Aggregated weights $W^*$ (99.67% F1) |
| **Tier 3: ZK Prover** | Floating-point weights $W^*$ | Fixed-Point Quantization + Poseidon Hash + Groth16 SNARK | Proof $\pi = (A, B, C)$, Public Signals |
| **Tier 4: Blockchain Ledger** | Proof $\pi$, Public Signals, Metrics | EVM Bilinear Pairing Check (`AMLVerifier.sol`) | Immutable confirmed Block & Tx Hash |

---

### 🎤 Speaker Notes for Slides (What to Say)
1. *"Our core algorithmic pipeline addresses the data silo problem in AML across 4 distinct mathematical phases."*
2. *"First, we convert raw tabular transactions into temporal bipartite graphs and extract topological features using PageRank and Graph Attention Networks (GAT)."*
3. *"Second, decentralized training occurs locally at each bank using Differentially Private SGD (DP-SGD) with gradient clipping ($C=1.0$) and calibrated noise, ensuring formal mathematical privacy."*
4. *"Third, Flower's EarlyStoppingFedAvg aggregates weights until F1 saturation at 99.67%, which are then quantized into Galois finite field elements."*
5. *"Fourth, a Circom circuit computes a Poseidon algebraic hash commitment and generates a Groth16 zk-SNARK proof. This proof is verified on-chain via elliptic curve bilinear pairing checks on Polygon Amoy, providing an immutable audit trail for financial regulators."*
