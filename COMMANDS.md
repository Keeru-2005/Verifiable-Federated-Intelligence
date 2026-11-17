# VFI command guide

These commands run the local prototype. The four clients are randomized shards of centrally preprocessed training data, not independently provisioned banks. Preprocessing overwrites its generated splits and model inputs, so preserve artifacts for any run you need to retain.

## Setup

```bash
pip install -r requirements.txt
npm install
cd blockchain && npm install && cd ..
```

## Preprocessing

```bash
python data/preprocess.py
# Optional quick sample; not directly comparable to the full-data run
python data/preprocess.py --sample-size 20000
```

The current pipeline creates chronological train/validation/test periods, constructs history-based features, fits scaler/PCA transforms on training rows only, applies no resampling, then shuffles the training feature rows into four client shards. Inspect `data/preprocess_manifest.json` after preprocessing. The retriever can fall back to mock data; verify the input source and checksum.

## Federated training

Start the server in one terminal:

```bash
python fl_implementation/server.py --rounds 10 --min-clients 4
```

Start the clients in separate terminals:

```bash
python fl_implementation/client.py --cid 1
python fl_implementation/client.py --cid 2
python fl_implementation/client.py --cid 3
python fl_implementation/client.py --cid 4
```

The server uses validation metrics for round and threshold selection, then writes a separate final-test record. Review `schemas/global_model_round_10.json` and `schemas/final_evaluation.json` together; do not treat a validation score as a final-test score.

## Validation-only model screening

Compare client learning rates across seeds without evaluating or overwriting the final-test record:

```bash
python evaluation/run_fl_validation_sweep.py --rounds 5 --seeds 42,43 --weights 10 --learning-rates 0.0005,0.001,0.002 --noise-multipliers 0.5 --fedprox-mus 0.0 --no-dp-diagnostic --skip-finalize
```

This keeps the current DP noise multiplier and positive-class weight fixed, selects using validation F1, and writes a sweep report under `evaluation/`. Repeatedly consulting the existing test split makes it development evidence rather than a fresh blind result; use rolling-origin validation across time periods when no new holdout data is available.

## Proof and ledger demo

Use the project scripts under `blockchain/` to launch a local Hardhat node, deploy the verifier contracts, generate the proof, and submit a round. The proof is scoped to the 33 quantized output-layer values; it does not prove full-model training or accuracy. Consult `zkp/circuit/aml_verify.circom` before describing what is verified.

## Centralized baseline

```bash
python evaluation/compare_baseline.py --epochs 10
```

This trains a pooled-data model, selects epoch/threshold using validation, and records a final-test baseline. It does not provide the data locality or DP properties of the FL experiment.

## Dashboard and tests

```bash
npm run dashboard
npm test
```

The checked-in test report describes a historical test record and includes old resampling assumptions. The current implementation was not re-tested during this documentation update. See `docs/TEST_RESULTS.md` for the test inventory and caveat, and `docs/CURRENT_PIPELINE_RESULTS.md` for result interpretation.
