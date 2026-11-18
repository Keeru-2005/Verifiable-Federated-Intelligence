# Current VFI Pipeline Results

**Status:** Four controlled validation-only runs completed; two candidates were approved for one final-test evaluation each. The formal-DP `pos_weight=25` final result is valid. The SMOTE-Tomek attempt used thresholds from a different checkpoint and is invalid; its test data were accessed once, so it must not be rerun or reported as a final score.

## Data protocol

The canonical manifest (`data/preprocess_manifest.json`, run `abc4b575614845cdb99a25f3f91aa979`) records a chronological whole-timestamp split:

| Split | Rows | Positive | Negative |
|---|---:|---:|---:|
| Train | 763,121 | 1,226 | 761,895 |
| Validation | 109,017 | 217 | 108,800 |
| Final test | 218,034 | 302 | 217,732 |
| Total | 1,090,172 | 1,745 | 1,088,427 |

The scaler, PCA, and post-PCA scaler are fit on pooled training rows only, then retained as a **shared train-only feature projection** for simulated horizontal FL. Four clients are randomized shards of centrally preprocessed training rows, not independent banks. The canonical manifest states `resampling: none`; the SMOTE-Tomek ablation is an isolated client-training experiment and does not change validation or test rows.

## Validation-only ablation

All four runs use seed 42, FedAvg, learning rate 0.002, DP noise multiplier 0.5, max-grad-norm 1.0, dropout 0.1, up to 25 rounds, and validation average-precision early stopping with patience 5. Selection is from the latest server-selected checkpoint, not merely the raw AP maximum.

| Candidate | Selected round | Validation F1 | Precision | Recall | PR-AUC (AP) | ROC-AUC | TP / FP / FN / TN | Raw threshold | Calibrated threshold | Epsilon |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| Baseline BCE | 10 | 0.713253 | 0.747475 | 0.682028 | 0.714746 | 0.982489 | 148 / 50 / 69 / 108,750 | 0.00547861 | 0.0457910 | 7.75214 |
| BCE, `pos_weight=25` | 10 | 0.716279 | 0.723005 | 0.709677 | 0.710067 | 0.981860 | 154 / 59 / 63 / 108,741 | 0.00252348 | 0.0310866 | 7.75214 |
| Focal (`alpha=0.75`, `gamma=2`) | 16 | 0.688995 | 0.716418 | 0.663594 | 0.709957 | 0.984576 | 144 / 57 / 73 / 108,743 | 0.2221460 | 0.1203540 | 8.90924 |
| BCE + client-local SMOTE-Tomek (`sampling_strategy=0.05`) | 8 | 0.751131 | 0.737778 | 0.764977 | 0.801182 | 0.991807 | 166 / 59 / 51 / 108,741 | 0.9993300 | 0.9882523 | 7.17958 nominal |

The JSON run record (`evaluation/ablation_results.json`) and per-run sweep JSON are authoritative for complete hyperparameters and metric precision.

## Final-test evaluations

The test set was accessed once for each approved candidate. Do not tune from these results. The SMOTE-Tomek attempt is invalid due to a round-8 model paired with round-9 calibration thresholds. Its returned metrics are retained only in the candidate archive as `invalid_attempt_metrics`; canonical `final_test_metrics` is null. Do not repeat that evaluation.

| Candidate | Validation F1 | Validation AP | Final-test F1 | Final-test precision | Final-test recall | Final-test AP | Final-test ROC-AUC | TP / FP / FN / TN | Threshold (raw; calibrated) | Epsilon / delta |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| BCE, `pos_weight=25`, round 10 | 0.716279 | 0.710067 | 0.679137 | 0.600509 | 0.781457 | 0.727406 | 0.993317 | 236 / 157 / 66 / 217,575 | 0.00252348; 0.0310866 | 7.75214 / 1e-5 |
| BCE + SMOTE-Tomek, round 8 | 0.751131 | 0.801182 | **Invalid; not reportable** | — | — | — | — | — | round-9 thresholds were used with round-8 weights | 7.17958 nominal only |

For SMOTE-Tomek, the single attempted test read yielded metrics for the mismatched model/threshold pair, but those values are invalid for the selected candidate. The record is preserved for audit and is deliberately omitted from the valid metrics table.

The formal-DP run uses Opacus accounting at `delta=1e-5`, with `dp_rng_mode=standard_prng_demo_only`. The RNG mode is demo-only and not a production-secure randomness claim. SMOTE-Tomek epsilon is nominal because synthetic records depend on private examples; it is not a formal DP guarantee for original records.

## Historical records and scope

Earlier saved schemas report baseline validation F1 0.714617 and final-test F1 0.680115. The older `0.7122` / `0.6788` pair came from another historical record. The four-run grid and approved evaluations above are distinct records; never combine their metrics or thresholds. The original near-99.7% F1 results are invalid due to leakage.

The ZK circuit constrains a sample-count-weighted relation and Poseidon commitment for 33 quantized output-layer values (32 weights and bias from `Linear(32, 1)`). It does not prove full-model training, data integrity, accuracy, or metric truthfulness. The current final-test split has guided project iteration and is development evidence, not a fresh blind estimate; future publication-quality evaluation requires a new frozen chronological holdout.

The archived round-8 update for the validation winner has a successfully generated and independently verified Groth16 proof; the full Python, Hardhat, and backend suites also pass. See `docs/TEST_RESULTS.md` for exact verification scope and counts.

See `evaluation/ablation_summary.md`, `docs/DIAGNOSTIC_REPORT.md`, and `docs/ABLATION_AND_OPTIMIZATION_REPORT.md` for detailed run and diagnostic records.
