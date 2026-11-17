# VFI project overview and abstract draft

> **Draft status:** this is a factual project overview, not a submission-ready abstract. The experiment uses simulated clients and the saved final-test result differs from the 71.22% validation result.

## Working title

**A Proof of Concept for Auditable Federated AML Classification with a Zero-Knowledge Aggregation Check**

## Abstract draft

This work presents Verifiable Federated Intelligence (VFI), a software proof of concept connecting transaction preprocessing, federated model training, a zero-knowledge aggregation check, and blockchain audit logging for anti-money-laundering (AML) classification. The current pipeline chronologically divides a transaction dataset into training, validation, and test periods, derives causal transaction-history features, and fits scaling and dimensionality-reduction transforms on training data. A PyTorch multilayer perceptron classifies 32 transformed features. Four Flower clients are simulated by randomly partitioning the centrally preprocessed training rows; they are not independent financial institutions. Client training uses positive-class loss weighting and an Opacus DP-SGD implementation. The server aggregates updates with FedAvg and selects a round and decision threshold using validation metrics.

In the saved round-10 record, validation F1 is 71.22% (precision 75.65%, recall 67.28%, average precision 70.42%) on 217 positive and 108,800 negative examples. The current saved final-test record reports F1 67.88% (precision 67.88%, recall 67.88%, average precision 71.67%) on 302 positive and 217,732 negative examples. The circuit proves a sample-count-weighted arithmetic relation for 33 quantized output-layer parameters and a Poseidon commitment. The contract verifies that proof and records round metadata; it does not prove full-model training, model accuracy, or metric truthfulness. These results demonstrate an integrated local prototype, not a production financial compliance system or an end-to-end deployment across real bank data silos.

## Keywords

Federated learning; anti-money-laundering classification; transaction features; differential privacy; zero-knowledge proofs; Groth16; smart contracts; software prototype.

## Metric and scope note

The old near-99.7% F1 figures are invalid because of target leakage in the earlier pipeline. The 71.22% value is validation F1; do not label it final-test performance. See `CURRENT_PIPELINE_RESULTS.md` for the saved artifacts and required caveats.

