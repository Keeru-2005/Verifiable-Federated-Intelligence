# Verifiable Federated Intelligence: Project Implementation Report

## 1. Objective and scope

VFI is a proof of concept exploring how a federated-learning workflow, differential-privacy training, a zero-knowledge aggregation check, and a blockchain audit record can be connected for an AML classification task. The current experiment uses one centrally processed transaction dataset and four simulated client shards. It does not yet implement independently operated bank pipelines or production-grade privacy/security controls.

## 2. Current implementation

### Data and features

The current preprocessor records a chronological whole-timestamp split into train, validation, and test periods before causal history feature extraction. Scaler, PCA, and post-PCA scaler are fitted on the training split. No oversampling is applied. The four client files are created by shuffling and splitting the pooled training feature table. The current classifier is a PyTorch MLP with 32 PCA-transformed features; transaction-history features inform the inputs, but the classifier is not a GNN doing graph message passing.

The saved manifest records 1,090,172 total rows: 763,121 training (1,226 positive), 109,017 validation (217 positive), and 218,034 test rows (302 positive). The dataset retrieval code has a mock-data fallback, so the exact source and checksum should be retained for a paper experiment.

### Federated training

Four Flower clients train locally from the simulated shards. The client uses positive-class loss weighting and contains Opacus DP-SGD. FedAvg combines the returned updates; the server selects the round and threshold on validation and evaluates a separate final-test file. The saved final evaluation reports epsilon 11.13 at delta 1e-5 and RNG mode `standard_prng_demo_only`; this is not a production-secure randomness claim.

### Proof and ledger

The Circom/Groth16 circuit proves a sample-count-weighted arithmetic relation for 33 quantized output-layer parameters and their Poseidon commitment. A Solidity contract verifies the proof and records round metadata. It does not prove local training correctness, full-model aggregation, data integrity, model accuracy, or the truth of the metric fields. Hashes and metrics stored in the contract are not all constrained by the proof.

## 3. Current saved results

The round-10 model artifact is a **validation** record: F1 71.22%, precision 75.65%, recall 67.28%, AP 70.42%, on 217 positives / 108,800 negatives. Its confusion matrix is `[[108753,47],[71,146]]` and its validation-selected threshold is 0.003817.

The current `schemas/final_evaluation.json` is a separate **final-test** record: F1 67.88%, precision 67.88%, recall 67.88%, AP 71.67%, on 302 positives / 217,732 negatives. Its confusion matrix is `[[217635,97],[97,205]]`.

Therefore, the repository evidence does not support calling 71.22% the final-test F1. The prior near-99.7% F1 values are invalid leakage-era metrics and are not current results. Accuracy is also misleading at the final-test prevalence of 0.1385%; an all-negative classifier would score approximately 99.86% accuracy.

The centralized comparison JSON records a pooled-data final-test F1 of 74.01% (precision 66.49%, recall 83.44%, AP 85.02%). It has no federation or DP property. Verify its exact training configuration before making it a formal benchmark comparison.

See [`CURRENT_PIPELINE_RESULTS.md`](CURRENT_PIPELINE_RESULTS.md) for the result artifacts and scope details.

## 4. What this prototype demonstrates

- A Flower client/server model-update workflow over four simulated data partitions.
- Training-only numeric transform fitting and a chronological evaluation split in the saved manifest.
- A client DP-SGD implementation with a per-run accountant record.
- A working Groth16 circuit for a narrowly stated output-layer aggregation relation.
- Contract-side proof verification and round metadata logging in a local EVM environment.

## 5. Limitations and work needed before submission

- Generate a new chronological final-test period not used in project iteration, then freeze the model, threshold, and configuration before evaluating it once.
- Reconcile the 71.22% validation record with the saved 67.88% final-test record; preserve raw logs and per-run configuration.
- Record input dataset provenance and checksum; verify that no mock fallback data were used.
- Treat four clients as simulations. A cross-institution claim requires institution-aware partitions and local feature construction, or narrower claims that explicitly describe the current centralized preparation.
- Report DP configuration and accountant scope; standard PRNG mode and epsilon must not be presented as production security.
- Keep the ZK statement narrow and explain that model accuracy and metric truthfulness are outside the circuit.
- Re-run and update the test suite against the current no-resampling code; the existing test report is stale in places.

