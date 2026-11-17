# VFI implementation outline

This outline describes the current proof-of-concept implementation, not a protocol for production banks.

## Current flow

```text
Input transaction table
    -> chronological train / validation / test split
    -> causal transaction-history features
    -> fit scaler, PCA and post-PCA scaler on train only
    -> shuffle and divide pooled training features into four simulated clients
    -> local weighted-loss DP-SGD updates using Flower
    -> FedAvg aggregation
    -> select round and decision threshold on validation
    -> evaluate selected checkpoint on saved final-test partition
    -> prove weighted aggregation relation for 33 quantized output-layer values
    -> verify proof and record round metadata in the local EVM ledger
```

The current manifest records `resampling: none`. Client class imbalance is addressed with a positive-class loss weight. The classifier is an MLP over 32 transformed features; it does not perform message passing over a graph during federated classification.

## Simplified algorithm

```text
Input: pooled transaction file D, number of simulated clients K=4
1. Split D chronologically into D_train, D_val, D_test before feature fitting.
2. Fit causal feature construction state and numeric transforms on D_train.
3. Transform D_val and D_test without using their labels for fitting.
4. Shuffle transformed D_train and partition into K client shards.
5. Initialize global model w_0.
6. For each communication round t:
     a. Send w_(t-1) to the simulated clients.
     b. Each client performs local weighted-loss training; DP-SGD is enabled
        by the current client configuration.
     c. Server applies sample-count-weighted FedAvg to returned updates.
     d. Evaluate aggregated model on D_val and record the chosen threshold.
     e. Retain the validation-selected model checkpoint.
7. Evaluate the selected checkpoint once using the configured D_test artifact.
8. Export the selected round's exact client output-layer values/counts.
9. Generate and verify a Groth16 proof of the circuit's stated weighted-average
   relation and Poseidon commitment; record proof and metrics metadata on-chain.
```

## Scope notes

- “Clients” are randomized partitions of pooled data. They do not represent independently sourced bank datasets.
- Preprocessing is centralized; the experiment does not establish raw-data locality from ingestion through feature extraction.
- The proof covers the selected output-layer aggregation relation, not the whole model, training procedure, input truth, or predictive performance.
- Performance metrics are not circuit-constrained. Ledger recording of a metric is not cryptographic proof that the metric is correct.
- See `../docs/CURRENT_PIPELINE_RESULTS.md` for the saved validation and final-test records and their current distinction.

