# Ablation Results

Validation metrics are used for tuning and selection. Final-test metrics are recorded only for explicitly finalized candidates.

## Completed validation-only comparison

| Run ID | Candidate | Selected round | Validation F1 | Validation PR-AUC (AP) | Epsilon |
|---|---|---:|---:|---:|---:|
| 20261118T081221Z_36933195_sweep1_seed42 | Baseline BCE | 10 | 0.713253 | 0.714746 | 7.75214 |
| 20261118T083749Z_696d18be_sweep1_seed42 | Cost-sensitive BCE (pos_weight=25) | 10 | 0.716279 | 0.710067 | 7.75214 |
| 20261118T085713Z_2f2b17e4_sweep1_seed42 | Focal (alpha=0.75, gamma=2) | 16 | 0.688995 | 0.709957 | 8.90924 |
| 20261118T092303Z_2ece36a5_sweep1_seed42 | BCE + SMOTE-Tomek (0.05) | 8 | 0.751131 | 0.801182 | 7.17958 |

## Final-test evaluation status

| Run ID | Candidate | Validation F1 | Validation PR-AUC | Final accuracy | Final F1 | Final PR-AUC | Final ROC-AUC | Precision | Recall | TP | FP | FN | TN | Raw threshold | Calibrated threshold | Epsilon | Delta | Accounting caveat |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| finalized_20261118T083749Z_696d18be_sweep1_seed42 | Cost-sensitive BCE (pos_weight=25) | 0.716279 | 0.710067 | 0.998977 | 0.679137 | 0.727406 | 0.993317 | 0.600509 | 0.781457 | 236 | 157 | 66 | 217575 | 0.00252348 | 0.0310866 | 7.75214 | 1e-05 | standard_opacus_accounting |
| finalized_20261118T092303Z_2ece36a5_sweep1_seed42 | BCE + client-local SMOTE-Tomek (0.05) | 0.751131 | 0.801182 | invalid | — | — | — | — | — | — | — | — | — | — | — | 7.17958 nominal | 1e-05 | round-8 weights paired with round-9 thresholds; test data read once |

The SMOTE-Tomek candidate's one-time final-test attempt is invalid and excluded from final metrics. It is not to be rerun because the final-test split was already accessed for that attempt. The original attempt values remain only in the candidate archive as audit data.

## Full run history

| Run ID | Status | Experiment | Selected round | Validation F1 | Validation AP | Final-test F1 | Final-test AP | Epsilon |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 20261118T074903Z_59f971a9_sweep1_seed42 | failed | validation_bce_none | None | — | — | — | — | — |
| 20261118T075111Z_2c7bb27e_sweep1_seed42 | failed | validation_bce_none | None | — | — | — | — | — |
| 20261118T075045Z_e2c38cee_sweep1_seed42 | failed | validation_bce_none | None | — | — | — | — | — |
| 20261118T081221Z_36933195_sweep1_seed42 | completed | validation_bce_none | 10 | 0.713253 | 0.714746 | — | — | 7.75214 |
| 20261118T083749Z_696d18be_sweep1_seed42 | completed | validation_bce_none | 10 | 0.716279 | 0.710067 | — | — | 7.75214 |
| 20261118T085713Z_2f2b17e4_sweep1_seed42 | completed | validation_focal_none | 16 | 0.688995 | 0.709957 | — | — | 8.90924 |
| 20261118T092303Z_2ece36a5_sweep1_seed42 | completed | validation_bce_smote_tomek | 8 | 0.751131 | 0.801182 | — | — | 7.17958 |
| finalized_20261118T092303Z_2ece36a5_sweep1_seed42 | invalid_checkpoint_threshold_mismatch | finalized_bce_smote_tomek | 8 | 0.751131 | 0.801182 | — | — | 7.17958 |
| finalized_20261118T083749Z_696d18be_sweep1_seed42 | completed | finalized_bce_none | 10 | 0.716279 | 0.710067 | 0.679137 | 0.727406 | 7.75214 |
