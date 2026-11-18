# Run results and provenance

The current verified run status is maintained in [`CURRENT_PIPELINE_RESULTS.md`](CURRENT_PIPELINE_RESULTS.md) and the canonical structured files `evaluation/ablation_results.json` and `evaluation/ablation_summary.md`.

The train/validation/test split is 763,121/109,017/218,034 rows with 1,226/217/302 positives. Four seed-42 validation candidates completed. BCE + client-local SMOTE-Tomek (0.05) was selected at round 8 with validation F1 0.751131 and AP 0.801182. Its test attempt is invalid because the test evaluation paired round-8 weights with round-9 thresholds; the test data were accessed once, so it must not be rerun or reported.

The one valid new final-test result is BCE `pos_weight=25`, selected round 10: validation F1 0.716279/AP 0.710067; final-test F1 0.679137/AP 0.727406/ROC-AUC 0.993317, precision 0.600509, recall 0.781457, and TP/FP/FN/TN=236/157/66/217,575. It reports epsilon 7.75214 at delta 1e-5 with `standard_prng_demo_only` RNG. The final-test period has been consulted during project iteration and is not an untouched publication estimate.

Earlier saved baseline schemas (validation F1 0.714617, final-test F1 0.680115), the historical 0.7122/0.6788 pair, older 15-round sweeps, and centralized comparisons are distinct artifacts, not parts of the new four-run grid. The original near-99.7% F1 results are invalid due to leakage. Consult the current results record before quoting any historical figure.

## Reproduction and integrity

Validation commands are listed in `COMMANDS.md`; all sweeps must use `--skip-finalize`. A final-test run requires explicit candidate approval, must use the exact validation-selected checkpoint/calibration, and is one-shot. Preserve the preprocessing manifest, source-data checksum, sweep JSON, selected-round record, checkpoint, and final-evaluation audit JSON together.
