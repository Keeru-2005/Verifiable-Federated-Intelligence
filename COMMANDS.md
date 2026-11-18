# VFI command guide

These commands run the local prototype. The four clients are randomized shards of centrally preprocessed training data, not independently provisioned banks. Preprocessing overwrites its generated splits and model inputs, so preserve artifacts for any run you need to retain.

## Setup

```bash
pip install -r requirements.txt
npm install
cd blockchain && npm install && cd ..
```

## Preprocessing

```bash
python data/preprocess.py
# Optional quick sample; not directly comparable to the full-data run
python data/preprocess.py --sample-size 20000
```

The current pipeline creates chronological train/validation/test periods, constructs history-based features, fits scaler/PCA transforms on training rows only, applies no resampling, then shuffles the training feature rows into four client shards. For the prioritized ablations, this pooled train-only scaler/PCA is retained as a shared feature projection, representing simulated horizontal FL; it is not institution-local preprocessing. Inspect `data/preprocess_manifest.json` after preprocessing. The retriever can fall back to mock data; verify the input source and checksum.

## Federated training

For a direct federated run, start the server in one terminal:

```bash
python fl_implementation/server.py --rounds 25 --patience 5 --min-clients 4
```

Start the clients in separate terminals:

```bash
python fl_implementation/client.py --cid 1
python fl_implementation/client.py --cid 2
python fl_implementation/client.py --cid 3
python fl_implementation/client.py --cid 4
```

The server selects the checkpoint by validation average precision, applies validation-fitted temperature scaling and a smoothed F1 threshold, then evaluates the final-test split once. Use `--validation-only` for tuning; it does not open the final-test CSV:

```bash
python fl_implementation/server.py --rounds 25 --patience 5 --min-clients 4 --validation-only
```

### Validation-only ablations

The unified runner starts and stops the server and four clients, supports client-local loss/resampling choices, DP clipping, FedProx, and dropout, and appends completed or failed trial records to `evaluation/ablation_results.json` and `evaluation/ablation_summary.md`.

The following four commands are the focused validation-only grid. Run them sequentially; each gets a unique output directory. All use seed 42, FedAvg, DP noise multiplier 0.5, clip norm 1.0, dropout 0.1, at most 25 rounds, AP early stopping with patience 5, and `--skip-finalize` so none reads `global_test.csv`.

```powershell
# Limit numerical-library worker threads for this memory-constrained host.
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

# Run 1: baseline BCE, no resampling
.\capstone_env\Scripts\python.exe evaluation\run_fl_validation_sweep.py --rounds 25 --final-rounds 25 --patience 5 --seeds 42 --weights 1 --loss-types bce --resampling-methods none --fedprox-mus 0 --learning-rates 0.002 --noise-multipliers 0.5 --local-epochs 1 --max-grad-norms 1 --model-dropouts 0.1 --no-dp-diagnostic --skip-finalize --port 8237

# Run 2: BCEWithLogitsLoss with positive weight 25
.\capstone_env\Scripts\python.exe evaluation\run_fl_validation_sweep.py --rounds 25 --final-rounds 25 --patience 5 --seeds 42 --weights 25 --loss-types bce --resampling-methods none --fedprox-mus 0 --learning-rates 0.002 --noise-multipliers 0.5 --local-epochs 1 --max-grad-norms 1 --model-dropouts 0.1 --no-dp-diagnostic --skip-finalize --port 8237

# Run 3: focal loss (alpha 0.75, gamma 2.0), no resampling
.\capstone_env\Scripts\python.exe evaluation\run_fl_validation_sweep.py --rounds 25 --final-rounds 25 --patience 5 --seeds 42 --weights 1 --loss-types focal --resampling-methods none --fedprox-mus 0 --learning-rates 0.002 --noise-multipliers 0.5 --local-epochs 1 --max-grad-norms 1 --model-dropouts 0.1 --focal-alpha 0.75 --focal-gamma 2 --no-dp-diagnostic --skip-finalize --port 8237

# Run 4: client-local SMOTE-Tomek (minority/majority ratio 0.05)
.\capstone_env\Scripts\python.exe evaluation\run_fl_validation_sweep.py --rounds 25 --final-rounds 25 --patience 5 --seeds 42 --weights 1 --loss-types bce --resampling-methods smote_tomek --smote-strategy 0.05 --fedprox-mus 0 --learning-rates 0.002 --noise-multipliers 0.5 --local-epochs 1 --max-grad-norms 1 --model-dropouts 0.1 --no-dp-diagnostic --skip-finalize --port 8237
```

The SMOTE-Tomek run's Opacus epsilon is nominal and is not a formal privacy guarantee for original records. Compare only validation F1 and average precision before approving a single finalization run.

```powershell
# Example: compare unweighted/cost-sensitive BCE and focal loss; do not finalize or read final test.
python evaluation\run_fl_validation_sweep.py `
  --rounds 25 --final-rounds 25 --patience 5 --seeds 42,43 `
  --weights 1,10,25,50 --loss-types bce,focal `
  --resampling-methods none --max-grad-norms 0.5,1.0,2.0 `
  --fedprox-mus 0,0.01 --skip-finalize
```

Use `--resampling-methods smote_tomek --smote-strategy 0.05` for a separate client-local SMOTE-Tomek trial. Its reported Opacus epsilon is nominal only and is **not** a formal privacy guarantee for original records after synthetic examples are generated.

The runner selects a screening configuration by validation average precision. Omit `--skip-finalize` only when intentionally running the selected configuration through the single `final_test_once` evaluation. That evaluation is not part of candidate tuning.

For a validation-only federated non-DP reference, add `--no-dp-diagnostic --skip-finalize`. This comparison is diagnostic only and is never eligible for DP-candidate selection:

```powershell
python evaluation\run_fl_validation_sweep.py `
  --rounds 25 --seeds 42 --weights 1 --loss-types bce `
  --resampling-methods none --no-dp-diagnostic --skip-finalize
```

The current host may not have enough Windows paging-file capacity for four simultaneous PyTorch client processes; the runner records process failures and aborts promptly instead of waiting for a long server timeout.

### Frozen-candidate final-test evaluation

The approved BCE `pos_weight=25` candidate has already been evaluated once. The SMOTE-Tomek attempt is invalid due to a checkpoint/threshold mismatch and has also consumed its one allowed test read. **Do not rerun either candidate.** For a future explicitly approved candidate, validate the exact server-selected checkpoint first and use a unique output directory:

```powershell
.\capstone_env\Scripts\python.exe evaluation\finalize_validation_candidate.py `
  --sweep-file evaluation\fl_validation_sweep_<run-id>.json `
  --loss-type bce --pos-weight 25 --resampling none `
  --output-dir evaluation\finalized_candidates\<unique-candidate-id>
```

The finalizer refuses mismatched sweep/manifest IDs, non-selected rounds, checkpoint/round-weight mismatches, validation-record inconsistencies, and reused output directories before loading the test CSV. Do not remove or bypass those guards. The command performs one final-test read when accepted.

## Proof and ledger demo

Use the project scripts under `blockchain/` to launch a local Hardhat node, deploy the verifier contracts, generate the proof, and submit a round. The proof is scoped to the 33 quantized output-layer values; it does not prove full-model training or accuracy. Consult `zkp/circuit/aml_verify.circom` before describing what is verified.

To prove the selected validation-winning SMOTE-Tomek round-8 aggregate without touching the final-test split, use the archived four-client update and isolate proof artifacts in that candidate archive:

```powershell
New-Item -ItemType Directory -Force `
  evaluation\finalized_candidates\smote_tomek_005_seed42\proof_stage\round_updates | Out-Null
Copy-Item `
  evaluation\finalized_candidates\smote_tomek_005_seed42\round_8.json `
  evaluation\finalized_candidates\smote_tomek_005_seed42\proof_stage\round_updates\round_8.json
.\capstone_env\Scripts\python.exe fl_implementation\extract_client_weights.py 8 `
  --schemas-dir evaluation\finalized_candidates\smote_tomek_005_seed42\proof_stage
node zkp\circuit\generate_circuit_input.js `
  evaluation\finalized_candidates\smote_tomek_005_seed42\proof_stage\round_updates\circuit_round_8.json 8 `
  evaluation\finalized_candidates\smote_tomek_005_seed42\proof_stage\private_inputs\round_8\input.json
$env:VFI_SCHEMAS_DIR = 'evaluation\finalized_candidates\smote_tomek_005_seed42\proof_stage'
node zkp\generate_proof.js `
  evaluation\finalized_candidates\smote_tomek_005_seed42\proof_stage\private_inputs\round_8\input.json 8
$env:VFI_PROOF_PATH = 'evaluation\finalized_candidates\smote_tomek_005_seed42\proof_stage\proof.json'
node zkp\verify_proof.js
Remove-Item Env:VFI_SCHEMAS_DIR
Remove-Item Env:VFI_PROOF_PATH
```

The extractor verifies four client vectors, 33 parameters per client, client counts, and their weighted aggregate before generating proof input. Do not use the test metric or attempt to repair the invalid SMOTE test result.

## Centralized baseline

```bash
python evaluation/compare_baseline.py --epochs 10
```

This trains a pooled-data model, selects epoch and calibration/threshold on validation, and opens the final-test CSV only after selection. It appends a separately labeled centralized benchmark record. It does not provide data locality or DP properties of the FL experiment.

## Dashboard and tests

```bash
npm run dashboard
```

In PowerShell, add the project virtualenv to `PATH` before invoking the repository test script:

```powershell
$env:PATH = "$PWD\capstone_env\Scripts;$env:PATH"
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
npm test
```

`npm test` runs the full Python suite, Hardhat tests, and backend API tests. Individual suites can be run with `npm run test:python`, `npm run test:contracts`, and `npm run test:api`. Groth16 proof verification is separate; use the commands above. See `docs/TEST_RESULTS.md` for the latest actual verification record, `docs/DIAGNOSTIC_REPORT.md` for the validation-only audit, and `docs/CURRENT_PIPELINE_RESULTS.md` for strict validation/final-test provenance.
