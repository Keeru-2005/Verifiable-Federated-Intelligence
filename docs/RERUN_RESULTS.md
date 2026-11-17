# VFI Run Results and Artifact Provenance

This page distinguishes the current saved causal-v3 outputs from older experiments. It does not claim a fresh run was executed while updating this documentation.

## Current saved run

The preprocessing manifest uses run ID `abc4b575614845cdb99a25f3f91aa979` and records 1,090,172 rows split chronologically into 763,121 training rows, 109,017 validation rows, and 218,034 test rows. Positive counts are 1,226, 217, and 302 respectively. Resampling is recorded as `none`.

### Federated round 10 — validation

`schemas/global_model_round_10.json` explicitly records `evaluation: "validation"` and reports:

| Metric | Validation result |
|---|---:|
| Accuracy | 99.8918% |
| Precision | 75.6477% |
| Recall | 67.2811% |
| F1 | **71.2195%** |
| Average precision | 70.4183% |
| ROC-AUC | 97.4145% |
| Threshold | 0.0038173 |
| Confusion matrix `[[TN,FP],[FN,TP]]` | `[[108753,47],[71,146]]` |
| Support | 109,017 (217 positive, 108,800 negative) |

The threshold was selected on validation. This is a validation result, not a test result.

### Federated selected checkpoint — final test

`schemas/final_evaluation.json` records a final-test evaluation with selected round 10 and threshold 0.0038173:

| Metric | Final-test result |
|---|---:|
| Accuracy | 99.9110% |
| Precision | 67.8808% |
| Recall | 67.8808% |
| F1 | **67.8808%** |
| Average precision | 71.6708% |
| ROC-AUC | 99.3869% |
| Confusion matrix `[[TN,FP],[FN,TP]]` | `[[217635,97],[97,205]]` |
| Support | 218,034 (302 positive, 217,732 negative) |

The current saved final-test record does **not** report 71.22% F1. If another run has produced 71.22% on final test, retain its unmodified raw terminal log and matching result JSON, including its run configuration, before making that claim. Do not relabel validation outputs.

### Centralized comparison

`evaluation/comparison_results.json` records a pooled-data centralized model selected at epoch 4 on validation. Its final-test result is F1 74.0088%, precision 66.4908%, recall 83.4437%, AP 85.0190%, ROC-AUC 99.6682%, and confusion matrix `[[217605,127],[50,252]]` on the same final-test row counts. This baseline does not provide federated data locality or DP. Its JSON uses the same preprocessing run ID, but does not serialize the complete training configuration; verify its settings and provenance before treating it as a definitive paper comparison.

## Other saved experiments

- `evaluation/fl_validation_sweep_20261116T184018Z_2927ee50.json`: one seed, 15-round validation trajectory; best validation F1 73.2861% at round 10; no final-test result (`final_test_once: null`).
- `evaluation/fl_validation_sweep_20261117T031145Z_427ac54b.json`: three-seed screening of a different positive-loss-weight configuration; best validation F1s vary across seeds and it has no final-test result.
- `evaluation/fl_validation_sweep_20261116T171455Z_6e561213.json`: earlier candidate selection and final-test result, with validation F1 73.2861% and final-test F1 70.4648%. This is a different saved run record and should not be merged with the current `schemas/final_evaluation.json`.
- The older full-corpus result F1 0.6169 used a different preprocessing protocol and split. It is historical and not directly comparable to the current 70/10/20 causal-v3 manifest.
- The original near-99.7% F1 values are invalid due to leakage and are not model results.

## Metric interpretation

With test prevalence 0.1385%, always predicting negative yields approximately 99.8615% accuracy. Accuracy is therefore not an adequate standalone metric. Report class counts, precision, recall, F1, average precision, threshold, and confusion matrix. AP and the operating threshold are relevant for rare-event ranking and alert burden.

The same test partition appears in multiple project artifacts and has been used across iterations. Treat its score as development evidence, not a fresh blind final estimate. For a paper, freeze the design using validation and evaluate once on a newly reserved later-period test period.

## Reproduction and integrity

Use the exact current code and dataset input, save the `preprocess_manifest.json`, record the training configuration and seed, and archive all round results plus `final_evaluation.json`. The current artifacts do not establish a complete source-data checksum or all FL hyperparameters. The dataset download script also has a mock-data fallback; verify the actual data source before publishing.

See `docs/CURRENT_PIPELINE_RESULTS.md` for the concise interpretation of current outputs and `docs/DATA_INTEGRITY.md` for the scope and limitations of the privacy and proof claims.
