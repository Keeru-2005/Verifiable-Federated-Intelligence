# VFI project overview and abstract draft

> **Draft status:** this describes a simulated-client proof of concept, not independently operated financial institutions or a production AML deployment. The final-test period has been consulted during project iteration; its score is not a fresh blind estimate.

## Working title

**A Proof of Concept for Auditable Federated AML Classification with a Zero-Knowledge Aggregation Check**

## Abstract draft

This work presents Verifiable Federated Intelligence (VFI), a prototype combining chronological transaction preprocessing, Flower federated model training, client-side differentially private stochastic gradient descent, a zero-knowledge aggregation check, and blockchain audit logging for anti-money-laundering classification. A transaction dataset is split chronologically into training, validation, and test periods, causal history features are generated, and scaler/PCA transforms are fit only on pooled training rows. Four Flower clients are simulated by random partitioning of centrally preprocessed training data; these do not represent independent financial institutions. In a controlled validation ablation, client-local SMOTE-Tomek resampling achieved F1 0.7511 and average precision 0.8012 at the server-selected round 8. Its one-time test attempt is invalid because the saved checkpoint was paired with calibration thresholds from another round and is excluded from final-test claims. The valid BCE model with positive-class weight 25 achieved validation F1 0.7163 and average precision 0.7101, and its one-time final-test record reports F1 0.6791 and average precision 0.7274 on 302 positive and 217,732 negative examples. Its Opacus accountant reports epsilon 7.752 at delta 1e-5 with demonstrative `standard_prng_demo_only` randomness. The Groth16 circuit constrains a sample-count-weighted relation for 33 quantized output-layer parameters and a Poseidon commitment; it does not prove complete model training, accuracy, or metric truthfulness. These results demonstrate a software integration prototype and do not establish production security, real-bank data locality, or generalization to an untouched holdout.

## Keywords

Federated learning; anti-money-laundering classification; transaction features; differential privacy; zero-knowledge proofs; Groth16; smart contracts; software prototype.

## Metric and scope note

Validation metrics and final-test metrics are separate. The SMOTE-Tomek final-test attempt is invalid and must not be repeated because the test partition has already been read for that attempt. The current final-test period is development evidence; a new frozen chronological holdout is needed for an external publication estimate. See `CURRENT_PIPELINE_RESULTS.md` for run records and limitations.
