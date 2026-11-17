# Current VFI Pipeline: Implementation and Results Record

**Status:** documentation of the current saved pipeline artifacts; this file does not claim a fresh run or test execution.

## What the prototype currently does

VFI is a local proof of concept combining chronological transaction preprocessing, a Flower-based federated training loop over four **simulated clients**, validation-based model selection, DP-SGD client training, a narrowly scoped Groth16 aggregation proof, and a smart-contract audit record. The current classifier is an MLP over 32 PCA-transformed, graph-informed transaction features. It is not a GNN classifier operating on four independently held institutional graphs.

The current preprocessor splits the transaction file into chronological train, validation, and test periods before causal history features are extracted. The saved manifest identifies 1,090,172 raw rows: 763,121 train rows (1,226 positives), 109,017 validation rows (217 positives), and 218,034 test rows (302 positives). The reported manifest says resampling is `none`. Scaler and PCA artifacts are fitted on the training split. The four client CSVs are produced by shuffling and equally splitting the pooled training feature table; they are simulated shards, not source-provided bank partitions. See `data/preprocess_manifest.json` and `data/preprocess.py`.

## Federated metrics in the saved artifacts

The checked-in round-10 model record is explicitly labeled `evaluation: "validation"`. It reports **71.22% validation F1**, precision 75.65%, recall 67.28%, average precision 70.42%, and ROC-AUC 97.41%, on 217 positives and 108,800 negatives. The threshold is 0.003817, selected by maximum F1 on the validation split. Its confusion matrix is `[[108753, 47], [71, 146]]` in `[[TN, FP], [FN, TP]]` order.

The current `schemas/final_evaluation.json` separately labels its record `final_test_once`. It reports **67.88% final-test F1**, precision 67.88%, recall 67.88%, average precision 71.67%, and ROC-AUC 99.39%, on 302 positives and 217,732 negatives. Its confusion matrix is `[[217635, 97], [97, 205]]`. The artifact also records best validation F1 as 71.22% and the selected threshold as 0.003817.

Therefore, the 71.22% result is supported as a **validation F1**, not as the final-test F1, by the currently saved round and final-evaluation JSON files. The recorded final-test F1 is 67.88%. A different final-test run with F1 71.22% would need a corresponding raw run artifact before that claim can be used.

## Centralized comparison artifact

`evaluation/comparison_results.json` contains centralized validation history and a centralized final-test record. The latter reports F1 74.01%, precision 66.49%, recall 83.44%, average precision 85.02%, and confusion matrix `[[217605, 127], [50, 252]]` on the same 302-positive, 217,732-negative test partition. It was selected at epoch 4 using validation. This is a pooled-data comparison and has no federated data-locality or DP property. The JSON does not record every training hyperparameter, so treat it as a repository result record and verify its run configuration before using it as a definitive benchmark.

## Privacy and proof scope

Client code contains Opacus DP-SGD with gradient clipping and Gaussian noise. The saved final evaluation reports a maximum client epsilon of 11.13 at delta 1e-5, but marks its RNG as `standard_prng_demo_only`. This is a prototype configuration, not a production-secure randomness claim. The DP configuration (including noise multiplier and accountant scope) should be preserved with any new run.

The Circom circuit proves a sample-count-weighted arithmetic relation for 33 quantized output-layer values and their Poseidon commitment. The contract verifies the proof and records the round. The circuit does not prove local training correctness, full-model aggregation, training-data integrity, model accuracy, or truthfulness of the performance metrics. The model-weight hash and metric fields are ledger metadata, not values constrained by this circuit.

## Interpretation and limits

- The near-99.7% F1 values from the original leaky pipeline are invalid as generalization results and must not be reused.
- Accuracy is not an informative headline under this prevalence: a classifier that predicts every example as negative would score about 99.86% on the current final-test split.
- The saved split is not a new blind evaluation if it has guided repeated project iterations. If no additional data is available, use nested rolling-origin validation on the existing chronological data: select features, thresholds, and settings using only inner windows, then evaluate once on each later outer window and report the aggregate and spread. This estimates temporal generalization more robustly but does not restore blind status to a test split whose results have already influenced development.
- The project does not yet demonstrate end-to-end local feature construction across real bank datasets. Preprocessing is pooled before the four simulated shards are created.
- Current documentation and test-result records include older SMOTE-era and 0.6169-result descriptions. Treat those as historical only where explicitly labeled; do not mix them with the current causal-v3 result.
- Static claims embedded in `dashboard/index.html` and `demo/run_demo.js` still contain older 0.6169/SMOTE descriptions. Those source files were not changed because this task explicitly prohibited code edits; refresh their display text before presenting the dashboard or scripted demo publicly.

## Files of record

- `data/preprocess_manifest.json` — current split, class counts, feature names, and preprocessing controls.
- `schemas/global_model_round_10.json` — round-10 validation metrics and threshold.
- `schemas/final_evaluation.json` — current saved final-test metrics and privacy summary.
- `evaluation/comparison_results.json` — centralized history/final-test comparison and federated validation history.
- `evaluation/fl_validation_sweep_20261116T184018Z_2927ee50.json` — 15-round, one-seed validation trajectory (no final test result).
- `evaluation/fl_validation_sweep_20261117T095900Z_d403a736.json` — five-round, two-seed learning-rate screen at DP noise multiplier 0.5; learning rate 0.002 had the highest mean best validation F1 (60.05%).
- `evaluation/fl_validation_sweep_20261117T120128Z_98a4a232.json` — ten-round, seed-42 validation-only run at learning rate 0.002; F1 71.46%, with no final-test evaluation.

