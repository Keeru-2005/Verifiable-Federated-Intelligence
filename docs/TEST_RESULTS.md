# Test inventory and recorded test status

## Status of this document

This file is an inventory of test coverage and a record of previously reported test execution. The historical report recorded 50 tests passing (32 Python, 11 Hardhat, and 7 API tests). **That is not a fresh verification of the current working tree.** The current documentation update did not run tests.

Several test descriptions in the previous version referred to local SMOTE, while the current preprocessor manifest records `resampling: none` and the current client applies positive-class loss weighting. The test suite and this report should be rerun/reconciled against the current implementation before submission. Do not display the old “50 passed” number as a current test badge or assertion.

## Test areas present in the repository

- Data preprocessing and split/feature integrity: `tests/test_data_preprocessing.py`
- Model and federated client/server behavior: `tests/test_model_fl.py`
- Proof input, generation, and verification: `tests/test_zk_snark.py`
- Pipeline orchestration: `tests/test_e2e_pipeline.py`
- Backend API behavior: `tests/test_backend_api.js`
- Contract and EVM verifier behavior: `blockchain/test/AMLVerifier.test.js`

## Run the tests and preserve evidence

```bash
npm test
```

For mentor or paper review, save the actual command, environment/version details, start/end time, exit code, and complete terminal output. Record which tests were skipped and whether they run against synthetic fixtures or the full pipeline. Update this report only from that output.

## Important distinction

Passing software tests can establish that specified code paths behave as expected on fixtures. They do not establish that AML metrics generalize, that client partitions represent independent institutions, that DP settings are production secure, or that the ZK circuit proves full model training. Current saved performance records and their scope are in `CURRENT_PIPELINE_RESULTS.md`.

