# Test inventory and verified status

**Last verified:** 2026-11-18 16:24 (+05:30), Windows, Python 3.12.6, Node 24.13.0, npm 11.6.2. The full suite was rerun after finalizer and dashboard API changes.

## Full repository test command

The complete `npm test` orchestration passed when the project virtualenv was added to `PATH`:

```powershell
Set-Location 'C:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence\Verifiable-Federated-Intelligence'
$env:PATH = "$PWD\capstone_env\Scripts;$env:PATH"
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
npm test
```

| Suite | Result |
|---|---:|
| Python / pytest | **48 passed**, 4 warnings |
| Hardhat contracts | **11 passing** |
| Backend API | **8 passed, 0 failed** |
| Total | **67 passed**, 0 failures |

The pytest run covered preprocessing, end-to-end pipeline, feature-representation comparison, model/FL behavior, optimization metrics (including rejection of mismatched validation checkpoint/threshold artifacts before test loading), and the isolated ZK-SNARK suite.

## Selected-checkpoint Groth16 proof

A separate proof was generated from the server-selected validation-winning SMOTE-Tomek round-8 archived client update (`evaluation/finalized_candidates/smote_tomek_005_seed42/round_8.json`). The extractor validated 4 clients × 33 parameters and their sample-count-weighted aggregation for 639,989 total samples. Groth16 proof generation passed, its immediate local pairing verification passed, and the separate `zkp/verify_proof.js` verification passed. Proof, input, and context artifacts are isolated in the candidate's `proof_stage/` folder; no test data were accessed.

The proof verifies only the circuit's weighted output-layer relation and commitment. It does not establish accuracy, training correctness, data integrity, or metric truthfulness.

## Coverage and rerun commands

- Python suite: `npm run test:python`
- Hardhat suite: `npm run test:contracts`
- Backend API suite: `npm run test:api`
- Full orchestration: `npm test`
- Selected archived proof flow: see `COMMANDS.md`

The previous historical count of 50 tests is superseded by this fresh run. Passing tests establish the specified software behavior on the current test fixtures; they do not establish production privacy, independent-bank data locality, or generalization to an untouched dataset.
