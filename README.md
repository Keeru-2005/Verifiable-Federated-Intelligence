# Verifiable Federated Intelligence (VFI)

VFI is a local proof of concept combining chronological transaction preprocessing, Flower federated training across four simulated clients, client DP-SGD, a narrow Groth16 aggregation check, and smart-contract audit logging. Pooled training rows are preprocessed centrally and randomized into client shards; this is not a deployment across independent banks.

The feature pipeline uses a pooled train-only scaler/PCA projection with 32 inputs. The final model layer remains `Linear(32, 1)`, preserving the 33 quantized output parameters consumed by the Groth16 circuit. The proof constrains a sample-count-weighted relation and Poseidon commitment only; it does not prove complete model training or predictive accuracy.

## Current results

The chronological split is 763,121 train rows (1,226 positives), 109,017 validation rows (217 positives), and 218,034 test rows (302 positives). The four-run validation ablation's leading candidate, client-local SMOTE-Tomek, achieved validation F1 **0.751131** and PR-AUC **0.801182** at selected round 8. Its one-time test attempt is invalid because the saved round-8 model was evaluated with round-9 calibration thresholds; do not report or rerun that score.

The valid one-time final-test result is the formal-DP BCE `pos_weight=25` candidate: validation F1 **0.716279** / AP **0.710067**, final-test F1 **0.679137** / AP **0.727406**, ROC-AUC 0.993317, precision 0.600509, recall 0.781457, TP/FP/FN/TN=236/157/66/217,575. It reports epsilon 7.75214 at delta 1e-5 with `standard_prng_demo_only` RNG. This dataset has been consulted during project iteration and is not a fresh blind publication estimate. Full results and limitations are in [`docs/CURRENT_PIPELINE_RESULTS.md`](docs/CURRENT_PIPELINE_RESULTS.md).

## Quick start

Install Python dependencies from `requirements.txt`, root Node dependencies with `npm install`, and blockchain dependencies from `blockchain/` when running the chain tests.

```powershell
python data\preprocess.py
python evaluation\diagnose_pipeline.py
python evaluation\run_fl_validation_sweep.py --help
```

For exact validation sweep, one-time finalization, ZK proof, and test commands, see [`COMMANDS.md`](COMMANDS.md). Inspect `data\preprocess_manifest.json` and confirm the actual dataset source; the download code may fall back to mock data.

## Privacy and interpretation

- The formal-DP candidate's epsilon is reported at delta=1e-5, but RNG mode `standard_prng_demo_only` is demonstrative, not a production-secure randomness claim.
- The SMOTE-Tomek epsilon is nominal only and not a formal guarantee for original records because synthetic samples depend on private examples.
- Four randomized shards and a shared centrally fitted projection do not establish institution-local data handling or realistic bank-specific non-IID behavior.
- The circuit is limited to the weighted output-layer relation and commitment; metric values and model accuracy are not proved.
- Accuracy is misleading under the approximately 0.14% test prevalence; consult confusion counts, precision, recall, F1, and average precision.

## Repository map

- `data/` — source retrieval, causal feature engineering, train-only preprocessing and manifest.
- `models/` — MLP with a fixed 32-to-1 output layer.
- `fl_implementation/` — Flower client/server and training configuration.
- `evaluation/` — diagnostics, validation sweeps, structured results, and candidate archives.
- `zkp/` — Circom circuit, SnarkJS proof generation and verification.
- `blockchain/` — Solidity verifier/ledger and Hardhat tests.
- `dashboard/` — local Express API and demonstration UI.
- `tests/` — Python and backend integration/unit tests.
- `docs/` — data integrity, runbook, diagnostic, optimization, and test records.
