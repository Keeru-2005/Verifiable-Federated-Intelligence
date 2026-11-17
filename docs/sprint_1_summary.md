# Sprint 1 Executive Summary: 10% Milestone Completion

> **Historical note (November 2026):** this sprint summary describes the original
> preprocessing design, in which SMOTE and scaler/PCA fitting happened before the
> train/test split and before bank sharding. That design leaked holdout information and
> has been replaced. See [`DATA_INTEGRITY.md`](DATA_INTEGRITY.md). The text below is kept
> as a record of the sprint, not as a description of the current pipeline. The bullets
> below are contemporaneous sprint notes, not independently verified measurements or a
> specification of the present implementation. In particular, historical SMOTE, GAT,
> bank-locality, test, and dataset-download statements must not be used as current claims.

This document preserves the team's Sprint 1 summary. The present repository state does not establish that every listed item was completed, tested, or committed on the named branch.

---

### 1. Repository Initialization & DevOps Setup
- **Docker Orchestration (`docker-compose.yml`)**: Engineered a decentralized, multi-container environment capable of flawlessly simulating 4 disjoint, non-communicating bank nodes.
- **Dependency Management (`requirements.txt`)**: Successfully resolved complex `protobuf` versioning conflicts to securely integrate PyTorch ML libraries concurrently alongside `web3` (Ethereum/Polygon) cryptography packages.
- **Data Security (`.gitignore`)**: Hardened the repository to prevent accidental leakage of raw banking data payloads (`amlnet.csv`) and large environment binaries to GitHub.

### 2. Autonomous Data Retrieval Pipeline
- **API Integration (`data/download_dataset.py`)**: Developed an automated data pipeline directly interfacing with the Zenodo REST API. It successfully queries, intercepts, and locally stages the multi-gigabyte **AMLNet_August 2025** dataset payload without human intervention.

### 3. Topological Graph Extraction
- **NetworkX Heuristics (`data/graph_features.py`)**: Traditional tabular ML models consistently fail to identify advanced laundering techniques like *smurfing arrays* or *money mule loops*. We engineered a script that maps transactions into a Directed Graphic topology. It extracts advanced mathematical heuristics (e.g., *PageRank*, *Betweenness Centrality*) that strictly isolate hidden laundering nodes.

### 4. Advanced ML Pre-Processing & Balancing
- **The Imbalance Fix (`data/preprocess.py`)**: Tackled the most severe challenge of AML detection: the extreme >99% / <1% class imbalance (also called the 'Accuracy Paradox'). 
- **SMOTE Interpolation**: Rather than haphazardly duplicating rows, we applied Synthetic Minority Over-sampling Technique (`SMOTE`) to mathematically synthesize new fraud typologies. We artificially padded the illicit behavior to encompass 30% of our matrix, allowing our future neural networks to cleanly recognize laundering patterns.

### 5. zk-SNARK Dimensionality Confinement
- **PCA Bottleneck**: Built a rigid mathematical choke-point utilizing Principal Component Analysis (`PCA`). Cryptographic Zero-Knowledge mathematical circuits have intense limits on dimensional complexity. Our `preprocess.py` pipeline forces the data into exactly **32 numeric columns**.
- **State Preservation**: The `StandardScaler` variances and `PCA` eigenvector matrices are fully serialized to local storage via `joblib`. This ensures that when training occurs later, each distinct bank node applies identical transformations blindly.

### 6. Visual Data Analytics
- Engineered robust visualization outputs mapping the network topologies, Spearman correlations, and class imbalances. All mathematical charts have been structurally routed to a single `visualizations/` directory to facilitate rapid panel grading and auditing.
