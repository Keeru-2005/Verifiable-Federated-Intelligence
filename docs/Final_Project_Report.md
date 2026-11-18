# Verifiable Federated Intelligence: Implementation Report

## Objective and scope

VFI is a local proof of concept connecting chronological transaction preprocessing, Flower-based federated training, client DP-SGD, a narrow Groth16 aggregation check, and smart-contract audit logging. The four clients are randomized shards of pooled preprocessed rows, not independent banks or financial institutions.

## Data and model

The canonical manifest records 1,090,172 rows: train 763,121 (1,226 positive), validation 109,017 (217), and test 218,034 (302). The split is chronological and made before causal-history feature generation. Scaler, PCA, and post-PCA scaler are fit on pooled training rows only. This is a shared projection across simulated horizontal-FL shards, not institution-local preprocessing.

The classifier is an MLP over 32 PCA-transformed features. Its final layer remains `Linear(32, 1)`, exposing 32 weights plus bias for the existing ZK interface. The circuit proves only a sample-count-weighted relation and Poseidon commitment for these 33 quantized output-layer values; it does not prove complete model training, metric truthfulness, or predictive accuracy.

## Controlled validation results

The seed-42 four-run grid used FedAvg, learning rate 0.002, DP noise multiplier 0.5, max-grad-norm 1.0, dropout 0.1, up to 25 rounds, and validation AP early stopping with patience 5.

| Candidate | Selected round | Validation F1 | Validation AP | Epsilon |
|---|---:|---:|---:|---:|
| Baseline BCE | 10 | 0.713253 | 0.714746 | 7.75214 |
| BCE, `pos_weight=25` | 10 | 0.716279 | 0.710067 | 7.75214 |
| Focal, alpha 0.75 / gamma 2 | 16 | 0.688995 | 0.709957 | 8.90924 |
| BCE + client-local SMOTE-Tomek (0.05) | 8 | **0.751131** | **0.801182** | 7.17958 nominal |

SMOTE-Tomek leads on validation metrics, but epsilon is nominal only and is not a formal guarantee for original records because synthetic rows depend on private examples.

## Final-test result and privacy

The valid one-time final test belongs to BCE with `pos_weight=25`, selected round 10. Validation F1/AP are 0.716279/0.710067; final-test F1/AP/ROC-AUC are 0.679137/0.727406/0.993317. Precision is 0.600509, recall 0.781457, and TP/FP/FN/TN are 236/157/66/217,575. The frozen raw/calibrated thresholds are 0.00252348/0.0310866. Opacus reports epsilon 7.75214 at delta 1e-5; its RNG mode is `standard_prng_demo_only`, not production-secure randomness.

The SMOTE-Tomek candidate's one-time test attempt is invalid: the selected round-8 model was evaluated using round-9 calibration thresholds. It remains audit-only and must not be rerun because the test partition has already been read for that attempt. The canonical final metrics are therefore missing for this validation winner. The physical test period has been consulted during project iteration and is not a new blind paper estimate.

## Diagnostic findings

The diagnostic report records validation logit quantiles of -14.8345/-13.3698/-11.7117/2.8349 (min/median/p95/p99.9) and probability quantiles 3.61e-7/1.56e-6/8.20e-6/0.9445. Sampled positive per-example gradient norms were disproportionately clipped at max-grad-norm 1.0: 69.92% versus 0% of sampled negatives. The audit was limited to one client/checkpoint and is not causal evidence. Historical validation FP/FN at threshold 0.0023786 were 60/63; older FP=156/FN=66 referred to another test record and must not be transferred to the current run.

## Limitations

The input source must be verified because retrieval code can fall back to mock data. Client shards are not institutional partitions; feature preprocessing is centralized. Differential-privacy RNG is demo-only, and resampling weakens the formal interpretation of its epsilon. The circuit's scope is limited to output-layer aggregation arithmetic and commitment. The currently used test split is development evidence; a publication-quality estimate requires a newly reserved later-period holdout.

Authoritative run artifacts and exact result tables are in `docs/CURRENT_PIPELINE_RESULTS.md`, `evaluation/ablation_results.json`, and `evaluation/ablation_summary.md`.
