# VFI local demonstration

The demo scripts exercise the local workflow: inspect preprocessed artifacts, run or read federated round outputs, generate a Groth16 proof for the implemented aggregation statement, and record verification metadata in a local EVM ledger. They demonstrate integration among components; they do not simulate independent real banks or prove end-to-end privacy.

## Current implementation summary

1. A central preprocessing step divides the input chronologically into train, validation, and test periods, extracts causal transaction-history features, and fits feature transforms on training rows.
2. The training feature table is shuffled into four simulated client shards. The current manifest records no resampling; clients apply positive-class loss weighting and DP-SGD.
3. Flower FedAvg aggregates client model updates. A server-side validation evaluator selects a round and decision threshold; a separate saved record contains the final-test evaluation.
4. The ZK circuit constrains a sample-count-weighted arithmetic relation and Poseidon commitment for the 33 quantized output-layer parameters.
5. The contract verifies the proof and records the round plus metric/hash metadata. The circuit does not prove those metrics, the full model, or the local training procedure.

## Current result artifacts

The saved round-10 result is **71.22% validation F1**, with 75.65% precision, 67.28% recall, and 70.42% AP on 217 positives and 108,800 negatives. The current saved final-test record is **67.88% F1**, with 67.88% precision and recall, 71.67% AP, and 302 positives / 217,732 negatives. See [`../docs/CURRENT_PIPELINE_RESULTS.md`](../docs/CURRENT_PIPELINE_RESULTS.md). The previous near-99.7% F1 results are invalidated leakage-era metrics.

## Run

```bash
npm run demo
# or
node demo/run_demo.js
```

See [`../COMMANDS.md`](../COMMANDS.md) for the end-to-end setup. The demo may consume saved artifacts rather than perform a fresh training run; check its terminal output and timestamps before describing it as a new experiment.

