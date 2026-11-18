# VFI Proof-of-Concept Runbook

This runbook describes the local prototype. Four Flower clients are randomized shards of centrally preprocessed training rows, not independent banks.

## Environment and preprocessing

Install dependencies from `requirements.txt` and root `package.json`; install `blockchain` dependencies for contract tests. Then run:

```powershell
python data\preprocess.py
```

The preprocessing step generates a chronological split before causal-history features, fits scaler/PCA transforms on pooled training rows only, and writes four client shards from a shared feature basis. The manifest's canonical setting is `resampling: none`. The downloader can fall back to mock data; verify the source and preserve an input checksum before reporting results.

## Leakage-safe diagnostics and validation

Run the training/validation-only audit:

```powershell
python evaluation\diagnose_pipeline.py
```

The four-run ablation commands and their exact flags are maintained in [`COMMANDS.md`](../COMMANDS.md). Use `--skip-finalize` for all tuning and validation sweeps. The pooled scaler/PCA is a shared train-only feature projection for simulated horizontal FL, not institution-local preprocessing.

## Final evaluation protocol

The approved BCE `pos_weight=25` candidate has one valid final-test result in `evaluation/finalized_candidates/bce_pos_weight_25_seed42/`. The approved SMOTE-Tomek attempt used round-8 weights with round-9 thresholds and is invalid. The test data were accessed once for that candidate, so do not rerun it. Test results are never used to change calibration, thresholds, or hyperparameters. See `docs/CURRENT_PIPELINE_RESULTS.md` for exact status and metrics.

For a future, newly approved candidate, the one-shot finalizer requires its validation sweep, loss, weighting/resampling config, and a fresh output directory under `evaluation/`. It validates the server-selected checkpoint and validation artifacts before loading the test CSV. Never change those artifacts after validation or reuse an output directory.

## ZK proof and local ledger

The Circom circuit accepts four client vectors of 33 quantized values and proves the sample-count-weighted aggregation relation plus Poseidon commitment. The proof does not establish full-model correctness, training correctness, data integrity, model accuracy, or metric truthfulness. See `zkp/circuit/README.md` and `COMMANDS.md` for proof commands and artifact paths.

## Dashboard and verification

```powershell
npm run dashboard
npm test
```

`npm test` runs pytest, Hardhat tests, and backend API tests. Groth16 proof generation/verification is a separate suite/command. Record fresh command output in `docs/TEST_RESULTS.md`; do not treat historical test counts as current.

## Canonical artifacts

- `data/preprocess_manifest.json` — split and transformation metadata.
- `evaluation/ablation_results.json` and `evaluation/ablation_summary.md` — separated validation/final-test run records.
- `evaluation/finalized_candidates/` — archived candidate checkpoint, selected-round update, and final-evaluation audit JSON.
- `docs/CURRENT_PIPELINE_RESULTS.md` — authoritative status and metric interpretation.
