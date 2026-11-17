# VFI Proof-of-Concept Runbook

This guide runs the local prototype. It is not a production deployment guide. Four Flower clients are simulated by splitting centrally preprocessed pooled training rows.

## Environment

Install dependencies using `requirements.txt`, the root `package.json`, and `blockchain/package.json` for the relevant components. Confirm Python, Node, Circom, and SnarkJS versions before recording a run.

## 1. Prepare data

```bash
python data/preprocess.py
```

The script uses the configured CSV and writes the train, validation, test, feature-transform, and client-shard artifacts. It chronologically splits before causal feature extraction and fits numeric transforms on the training split. The current manifest records no resampling. The four client shards are generated from pooled preprocessed training rows. Inspect `data/preprocess_manifest.json`; record the source file checksum, seed, manifest, and a copy of all generated outputs. The downloader can fall back to mock data, so confirm the dataset source.

## 2. Run federated learning

Start the server:

```bash
python fl_implementation/server.py --rounds 10 --min-clients 4
```

Start one client process per terminal:

```bash
python fl_implementation/client.py --cid 1
python fl_implementation/client.py --cid 2
python fl_implementation/client.py --cid 3
python fl_implementation/client.py --cid 4
```

Round artifacts contain validation results. The server's final evaluation record is written separately. For the saved current artifacts, round 10 validation F1 is 71.22%; the current final-test record is 67.88%. Consult `CURRENT_PIPELINE_RESULTS.md`; do not call a validation result final-test performance.

## 3. Run the centralized reference

```bash
python evaluation/compare_baseline.py --epochs 10
```

The centralized model is trained on pooled data, with epoch and threshold chosen using validation. Its saved test result is a reference without federation or DP. Preserve the resulting `evaluation/comparison_results.json` and exact configuration.

## 4. Local proof and ledger demonstration

Start a local EVM node, deploy the contracts, generate/verify the proof, and submit the round using the scripts in `blockchain/`. The proof covers the specified aggregation arithmetic and commitment for the selected output-layer parameters. It does not prove training correctness or predictive performance. Confirm `aml_verify.circom` and `AMLVerifier.sol` before describing this evidence.

## 5. Dashboard and tests

```bash
npm run dashboard
npm test
```

The dashboard is for local demonstration. The checked-in test report contains historical counts and older implementation details; it is not evidence that tests were freshly run on the current tree. Run the tests and record command output before claiming current pass status.

## Result records

- `data/preprocess_manifest.json` — data split and preprocessing metadata.
- `schemas/global_model_round_N.json` — per-round aggregated model and validation results.
- `schemas/final_evaluation.json` — selected checkpoint's final-test result.
- `evaluation/comparison_results.json` — centralized baseline and federated validation history.
- `evaluation/fl_validation_sweep_*.json` — candidate sweep details; check `final_test_once` before quoting any score.

