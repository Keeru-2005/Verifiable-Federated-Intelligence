# Data integrity and scope of the current pipeline

## Historical issue

The original results near 99.7% F1 were produced by an earlier pipeline with leakage paths. The project record says graph features, a GAT encoder, and scaling/PCA were fitted using the full dataset before splitting, and SMOTE was applied before bank sharding. The old figures must not be cited as generalization performance. They are retained only as invalid historical results.

## Current saved preprocessing protocol

The current `causal-v3` manifest records a chronological whole-timestamp train/validation/test split before causal feature extraction. The recorded row counts are:

| Split | Rows | Positive | Negative |
|---|---:|---:|---:|
| Train | 763,121 | 1,226 | 761,895 |
| Validation | 109,017 | 217 | 108,800 |
| Final test | 218,034 | 302 | 217,732 |
| Total | 1,090,172 | 1,745 | 1,088,427 |

The current manifest records `resampling: none`. The data preprocessing code extracts history-based transaction features and fits StandardScaler, PCA, and the post-PCA scaler on the train split. Validation and test rows are transformed without their labels. See `data/preprocess_manifest.json` and `data/preprocess.py`.

The four FL clients are simulated by shuffling the pooled training feature table and splitting it into four shards. Preprocessing is performed centrally before those shards are made. This is **not** a demonstration of four banks independently keeping raw data and constructing their own local features, nor does it model institutional non-IID partitions.

## Current model result records

The round-10 record in `schemas/global_model_round_10.json` is labeled `validation`. It reports F1 0.712195 (71.22%), precision 0.756477 (75.65%), recall 0.672811 (67.28%), AP 0.704183 (70.42%), and ROC-AUC 0.974145 on 217 positive and 108,800 negative validation rows. Its selected threshold is 0.003817 and confusion matrix is `[[108753, 47], [71, 146]]` in `[[TN, FP], [FN, TP]]` order.

The current `schemas/final_evaluation.json` is a separate `final_test_once` record. It reports F1 0.678808 (67.88%), precision 0.678808, recall 0.678808, AP 0.716708, ROC-AUC 0.993869, and confusion matrix `[[217635, 97], [97, 205]]` on 302 positive and 217,732 negative test rows.

Accordingly, **71.22% is the saved validation F1, not the saved final-test F1**. The current final-test artifact records 67.88%. A different run producing 71.22% on the test partition must be accompanied by a corresponding final-test JSON/log before it is described that way. The 73.29% from the 15-round sweep is also a validation result; that sweep contains no final-test evaluation.

The same physical test rows have been used during project iteration. Even though the current server code separates validation from the test evaluator, this dataset should be considered development evidence rather than a blind, one-time external estimate. A final submission should freeze the configuration on validation and evaluate once on a newly reserved later-period holdout.

## Privacy, client, and proof scope

- Client training uses Opacus DP-SGD. The saved final evaluation reports maximum client epsilon 11.13 at delta 1e-5, with RNG mode `standard_prng_demo_only`. This does not support a claim of production-secure random-number generation. Preserve noise multiplier, clip norm, sampling, number of steps, accountant state scope, and seed/configuration per run.
- The circuit proves a sample-count-weighted aggregation relation for 33 quantized output-layer values and a Poseidon commitment. It does not prove full-model aggregation, correct local training, raw-data integrity, model accuracy, or metric truthfulness.
- The blockchain record includes model hashes and performance metrics as metadata; these values are not all constrained by the ZK circuit.
- The dataset retriever may use a mock-data fallback if upstream retrieval fails. Record the exact input source and a checksum for any experiment reported externally.

## Historical documents

Some older documents describe an 80/20 train/holdout protocol, local SMOTE, GAT fitting, 349 test positives, or F1 0.6169. Those refer to older experiment records and must not be mixed with the current manifest and result files. `docs/CURRENT_PIPELINE_RESULTS.md` is the concise current artifact index.
