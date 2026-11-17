# Verifiable Federated Intelligence (VFI)

VFI is a local proof of concept that combines transaction preprocessing, a Flower federated-learning loop over four simulated clients, DP-SGD client training, a narrowly scoped Groth16 aggregation proof, and smart-contract audit logging.

The prototype is not a deployment across real banks. Raw data are centrally preprocessed before the pooled training rows are randomized into four client shards. The current classifier is an MLP over 32 PCA-transformed graph-informed transaction features. The ZK circuit proves a sample-count-weighted relation for 33 quantized output-layer values; it does not prove full-model training or prediction accuracy.

## Current evidence

The original near-99.7% F1 figures came from a leaky pipeline and are invalid as generalization results. The current saved round-10 artifact reports **71.22% validation F1**. The saved final evaluation separately reports **67.88% final-test F1**. These numbers must not be interchanged. The split counts, confusion matrices, threshold, centralized baseline record, and artifact limitations are documented in [`docs/CURRENT_PIPELINE_RESULTS.md`](docs/CURRENT_PIPELINE_RESULTS.md).

The current preprocessing manifest records 1,090,172 rows split chronologically into 763,121 train, 109,017 validation, and 218,034 test rows. Positive counts are 1,226, 217, and 302. Resampling is `none`. See [`data/preprocess_manifest.json`](data/preprocess_manifest.json) and [`docs/DATA_INTEGRITY.md`](docs/DATA_INTEGRITY.md).

## Run the prototype

Install Python dependencies from `requirements.txt`, root Node dependencies with `npm install`, and the blockchain dependencies from `blockchain/` as needed.

```bash
# Preprocess the configured CSV (this overwrites generated split artifacts)
python data/preprocess.py

# Terminal 1: start the Flower server
python fl_implementation/server.py --rounds 10 --min-clients 4

# Terminal 2: start all four simulated clients in separate shells
python fl_implementation/client.py --cid 1
python fl_implementation/client.py --cid 2
python fl_implementation/client.py --cid 3
python fl_implementation/client.py --cid 4
```

The code supports a local blockchain demonstration through Hardhat and the scripts in `blockchain/`. See [`COMMANDS.md`](COMMANDS.md) for ordered commands and [`docs/RUNBOOK.md`](docs/RUNBOOK.md) for the demo workflow. A dataset download path has a mock-data fallback; verify the actual data source and checksum before reporting experimental results.

## Test suite

Run the repository test command with `npm test` or the suites listed in `docs/TEST_RESULTS.md`. That document is a historical test record and includes some statements from an older SMOTE-era implementation. The current no-resampling implementation has not been revalidated as part of the current documentation update; do not use the old pass count as a fresh verification claim.

## Privacy and proof scope

- The client contains Opacus DP-SGD with gradient clipping and Gaussian noise. The saved final evaluation reports maximum client epsilon 11.13 at delta 1e-5 and RNG mode `standard_prng_demo_only`. This is a prototype setting, not production-secure randomness.
- The proof circuit constrains a weighted-average relation and Poseidon commitment for 33 output-layer parameters. It does not establish training correctness, data integrity, model accuracy, or metric truthfulness.
- The ledger records proof status, hashes, and metric metadata. The model hash and metric fields are not all constrained by the ZK proof.
- Because data preparation is pooled and the client partitions are randomized, the prototype does not demonstrate end-to-end institutional data locality or realistic bank-specific non-IID learning.

## Repository map

- `data/` — input handling, causal transaction features, preprocessing manifest.
- `models/` — PyTorch MLP classifier.
- `fl_implementation/` — Flower server and client, simulated data shards.
- `evaluation/` — saved validation sweeps and centralized comparison.
- `zkp/` — Circom circuit and SnarkJS proof tools.
- `blockchain/` — Solidity verifier/ledger, Hardhat scripts, and tests.
- `dashboard/` — local Express API and demonstration UI.
- `docs/` — runbook, integrity notes, project report, and result reconciliation.

