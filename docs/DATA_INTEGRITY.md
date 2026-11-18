# Data integrity and experimental scope

## Current chronological split

The canonical manifest (`data/preprocess_manifest.json`) records the following chronological whole-timestamp partition, made before causal history features are generated:

| Split | Rows | Positive | Negative |
|---|---:|---:|---:|
| Train | 763,121 | 1,226 | 761,895 |
| Validation | 109,017 | 217 | 108,800 |
| Final test | 218,034 | 302 | 217,732 |
| Total | 1,090,172 | 1,745 | 1,088,427 |

StandardScaler, PCA, and the post-PCA scaler are fit on pooled training rows only. The resulting feature basis is retained as a **shared train-only projection** across four simulated horizontal-FL shards, which gives clients compatible feature coordinates. Preprocessing is centralized before sharding; the setup is not four independently operated bank pipelines. The manifest's canonical `resampling` value is `none`. The SMOTE-Tomek ablation applies sampling only inside a client's local training shard and not to validation or test rows.

The source loader can fall back to mock data if upstream retrieval fails. The manifest alone does not establish upstream provenance. Preserve the exact input dataset and checksum for any externally reported experiment.

## Leakage controls and limitations

Validation data are used for checkpoint selection, temperature fitting, and threshold selection. They are never joined into training resampling. Test labels are excluded from model selection, calibration, and threshold selection. A final-test run is executed only for an explicitly selected frozen candidate.

Two approved candidates were evaluated once. The formal-DP `pos_weight=25` test result is valid. The SMOTE-Tomek evaluation used round-8 model weights with round-9 calibration thresholds; it is invalid, retained for audit only, and must not be rerun because the test partition was already read for that attempt. The physical test period has also been consulted during broader project iteration, so it is development evidence rather than a new blind estimate.

## Privacy and resampling

The formal-DP candidate reports Opacus epsilon=7.75214 at delta=1e-5 with `dp_rng_mode=standard_prng_demo_only`. This RNG setting is demonstrative and must not be described as production-secure randomness.

For SMOTE-Tomek, synthetic samples are functions of private examples. The reported Opacus epsilon=7.17958 is nominal only and is **not a formal DP guarantee for original records** without an accounting argument that includes the resampling mechanism.

## Proof scope

The Groth16 circuit proves the sample-count-weighted relation and Poseidon commitment for 33 quantized output-layer parameters (32 weights and one bias from `Linear(32, 1)`). It does not prove complete model aggregation, client training correctness, source-data integrity, model accuracy, or metric truthfulness. Contract-side metrics and model hashes are metadata unless separately constrained by the circuit.

## Historical records

Near-99.7% F1 results from the earlier leaky feature/transformation/resampling pipeline are invalid. Older saved schemas and documents may contain historical baseline values; the authoritative current candidate status and metric separation are in `CURRENT_PIPELINE_RESULTS.md` and `evaluation/ablation_summary.md`.
