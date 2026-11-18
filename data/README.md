# Data ingestion and preprocessing

This directory contains the AML transaction input and preprocessing pipeline used by the VFI proof of concept. The current pipeline splits rows chronologically into train, validation, and final-test periods before causal history features are constructed. StandardScaler, PCA, and the post-PCA scaler are fitted on training rows only. The saved current manifest records no resampling (`resampling: none`); class imbalance is handled in client training with positive-class loss weighting.

## Current data protocol

The saved `preprocess_manifest.json` records 1,090,172 raw rows, split into 763,121 training rows (1,226 positive), 109,017 validation rows (217 positive), and 218,034 test rows (302 positive). The test partition contains 217,732 negatives. The four client files are created by shuffling and splitting the pooled training feature table into four simulated shards. They are not institutional bank datasets and do not establish real cross-bank non-IID behavior.

Preprocessing currently occurs centrally before the four client shards are written. Thus, the training loop demonstrates Flower parameter exchange across simulated clients, while the end-to-end data-locality premise (each bank independently deriving features from its own raw records) remains unimplemented.

The source loader (`download_dataset.py`) can fall back to standardized mock data if retrieval fails. Preserve the actual input file and its checksum for any reported run; the manifest alone does not establish which upstream dataset instance was used.

## Current result records

The current four-run validation grid's winner is client-local SMOTE-Tomek (0.05), selected at round 8 with F1 0.751131 and AP 0.801182. Its one-time test attempt is invalid due to a round-8 model/round-9 threshold mismatch and must not be rerun. The valid BCE `pos_weight=25` final-test record has validation F1 0.716279/AP 0.710067 and final-test F1 0.679137/AP 0.727406. See `docs/CURRENT_PIPELINE_RESULTS.md` for complete metrics and the strict validation/test distinction.

## Main files

- `preprocess.py` — chronological split, causal feature construction, train-only transformations, sharding, and manifest creation.
- `graph_features.py` — graph-informed and causal transaction feature construction.
- `temporal_bipartite_graph.py` — temporal bipartite graph utilities.
- `preprocess_manifest.json` — saved split and preprocessing metadata.
- `download_dataset.py` — dataset retrieval with a documented fallback path.

Historical documentation describing full-dataset fitting, SMOTE before sharding, or local-only SMOTE refers to earlier pipeline versions and is not the current protocol. The old near-99.7% F1 values are invalidated because of leakage; do not cite them as current performance.
