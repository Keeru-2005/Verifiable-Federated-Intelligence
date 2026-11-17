# Data ingestion and preprocessing

This directory contains the AML transaction input and preprocessing pipeline used by the VFI proof of concept. The current pipeline splits rows chronologically into train, validation, and final-test periods before causal history features are constructed. StandardScaler, PCA, and the post-PCA scaler are fitted on training rows only. The saved current manifest records no resampling (`resampling: none`); class imbalance is handled in client training with positive-class loss weighting.

## Current data protocol

The saved `preprocess_manifest.json` records 1,090,172 raw rows, split into 763,121 training rows (1,226 positive), 109,017 validation rows (217 positive), and 218,034 test rows (302 positive). The test partition contains 217,732 negatives. The four client files are created by shuffling and splitting the pooled training feature table into four simulated shards. They are not institutional bank datasets and do not establish real cross-bank non-IID behavior.

Preprocessing currently occurs centrally before the four client shards are written. Thus, the training loop demonstrates Flower parameter exchange across simulated clients, while the end-to-end data-locality premise (each bank independently deriving features from its own raw records) remains unimplemented.

The source loader (`download_dataset.py`) can fall back to standardized mock data if retrieval fails. Preserve the actual input file and its checksum for any reported run; the manifest alone does not establish which upstream dataset instance was used.

## Current result records

The round-10 record in `schemas/global_model_round_10.json` is validation-only: F1 71.22%, precision 75.65%, recall 67.28%, AP 70.42%, and ROC-AUC 97.41% on 217 positives / 108,800 negatives. The current `schemas/final_evaluation.json` records final-test F1 67.88%, precision and recall 67.88%, AP 71.67%, and ROC-AUC 99.39% on 302 positives / 217,732 negatives. Do not describe the 71.22% validation value as final-test performance. See `docs/CURRENT_PIPELINE_RESULTS.md` for details and run-artifact distinctions.

## Main files

- `preprocess.py` — chronological split, causal feature construction, train-only transformations, sharding, and manifest creation.
- `graph_features.py` — graph-informed and causal transaction feature construction.
- `temporal_bipartite_graph.py` — temporal bipartite graph utilities.
- `preprocess_manifest.json` — saved split and preprocessing metadata.
- `download_dataset.py` — dataset retrieval with a documented fallback path.

Historical documentation describing full-dataset fitting, SMOTE before sharding, or local-only SMOTE refers to earlier pipeline versions and is not the current protocol. The old near-99.7% F1 values are invalidated because of leakage; do not cite them as current performance.
