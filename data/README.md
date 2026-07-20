# Data Pre-Processing & Graph Extraction Pipeline

This submodule manages the dataset ingestion, graph feature orchestration, and cryptographic dimensionality reduction needed prior to feeding elements to our Federated Network and later the zkLLVM circuit verification node.

## File Map
- `download_dataset.py`: Fetches the `amlnet` structure specifically via the Zenodo API querying ID `16736515`. Includes fallback to generate standardized mock data to prevent build-pipeline failures during network downtime. 
- `graph_features.py`: Derives structural network features (`in_degree`, `out_degree`, `PageRank`, `clustering_coefficient`, `betweenness_centrality`), temporally linking transaction times and standardizing value-weights. Uses `NetworkX`.
- `preprocess.py`: Manages the mathematical payload formatting. Specifically enforces standard scaling, followed by polynomial feature escalation, ultimately condensed by linear `PCA` transformations precisely down to **32 continuous variables**. This discrete fixed target is compulsory to retain zk-SNARK compiler capability and performance consistency within the `Prover`. Output yields structured train/test subsets for nodes `{1, 2, 3, 4}` stratified correctly against the 2% imbalance standard.

## Workflow Runtime Protocol:
```bash
# To run pipeline fully:
python download_dataset.py
python preprocess.py
```
