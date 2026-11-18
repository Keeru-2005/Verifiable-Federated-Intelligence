# VFI AML Optimization and Ablation Report

## Methodology and scope

The canonical manifest describes a chronological whole-timestamp split: 763,121 train rows (1,226 positives), 109,017 validation rows (217 positives), and 218,034 test rows (302 positives). The scaler, PCA, and post-PCA scaler are fit on pooled training rows only. These transforms are retained as a **shared train-only feature projection** for the simulated horizontal-FL clients; the four clients are randomized shards, not independent institutions.

All four controlled candidates used seed 42, FedAvg, learning rate 0.002, DP noise multiplier 0.5, max-grad-norm 1.0, dropout 0.1, at most 25 rounds, and validation average-precision early stopping with patience 5. Calibration and threshold selection use validation only: scalar temperature scaling, smoothed F1 sweep, and a separate precision-at-least-0.70 operating threshold when available. Model selection honors the server's `selected_by_validation` checkpoint flag.

The final output layer remains `Linear(32, 1)`. It exposes 32 weights plus one bias and preserves the 33-parameter Groth16 interface.

## Diagnostics

`docs/DIAGNOSTIC_REPORT.md` records the underlying sampled diagnostics. On the prior checkpoint, validation logits had min/median/p95/p99.9 of -14.8345/-13.3698/-11.7117/2.8349; probability quantiles were 3.61e-7/1.56e-6/8.20e-6/0.9445. The tiny majority score mass and high positive tail help explain why a threshold near 0.00238 was selected; this is not evidence that raw probabilities are calibrated.

At clip norm 1.0, the sampled positive per-example gradient median/p95 were 19.77/1010.74, with 69.92% exceeding the clipping bound. Sampled negative median/p95 were 2.31e-5/1.64e-4, with 0% clipped. This is a one-client/checkpoint audit, not a population-wide estimate or causal proof.

The prior validation error breakdown at threshold 0.0023786 was FP=60 and FN=63. The previous request's FP=156/FN=66 refers to a different historical final-test record. The present formal-DP final-test record has FP=157/FN=66. Error counts must always be interpreted with the corresponding split, run, checkpoint, and threshold.

The preprocessing artifact retains all 32 PCA components. The diagnostic report includes explained variance and feature-signal caveats; alternative feature representations remain development comparisons, not independently validated improvements. Standardization after PCA is part of the current train-only feature pipeline.

## Four-run validation ablation

| Candidate | Selected round | Validation F1 | Precision | Recall | PR-AUC (AP) | ROC-AUC | TP / FP / FN / TN | Raw / calibrated threshold | Epsilon |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| Baseline BCE | 10 | 0.713253 | 0.747475 | 0.682028 | 0.714746 | 0.982489 | 148 / 50 / 69 / 108,750 | 0.00547861 / 0.0457910 | 7.75214 |
| BCE, `pos_weight=25` | 10 | 0.716279 | 0.723005 | 0.709677 | 0.710067 | 0.981860 | 154 / 59 / 63 / 108,741 | 0.00252348 / 0.0310866 | 7.75214 |
| Focal, `alpha=0.75`, `gamma=2` | 16 | 0.688995 | 0.716418 | 0.663594 | 0.709957 | 0.984576 | 144 / 57 / 73 / 108,743 | 0.2221460 / 0.1203540 | 8.90924 |
| BCE + client-local SMOTE-Tomek, strategy 0.05 | 8 | **0.751131** | 0.737778 | 0.764977 | **0.801182** | 0.991807 | 166 / 59 / 51 / 108,741 | 0.9993300 / 0.9882523 | 7.17958 nominal |

These are validation results, not final-test metrics. The canonical run IDs and full details are in `evaluation/ablation_summary.md` and `evaluation/ablation_results.json`. SMOTE-Tomek leads on validation F1 and AP. Its epsilon is nominal only because synthetic samples depend on private examples; it is not a formal DP guarantee for original records.

## Final-test record and privacy-utility comparison

Two candidates were approved for a one-time `final_test_once` evaluation. The BCE + `pos_weight=25` result is valid: F1=0.679137, precision=0.600509, recall=0.781457, AP=0.727406, ROC-AUC=0.993317, with TP=236, FP=157, FN=66, TN=217,575. The frozen raw/calibrated thresholds are 0.00252348/0.0310866. Its Opacus accountant reports epsilon=7.75214 at delta=1e-5 and RNG mode `standard_prng_demo_only`.

The approved SMOTE-Tomek test attempt is **invalid and not reportable**. The saved model checkpoint is server-selected round 8, but the one-time evaluation paired it with round-9 calibration thresholds. The test data were accessed once; the attempt is archived as invalid and must not be repeated. The only valid final-test row presently available is therefore the formal-DP candidate. This comparison does not support a final-test utility claim for the validation-winning SMOTE candidate.

The shared physical test period has been consulted during project iteration, so even the valid one-time record is development evidence, not an untouched publication estimate. For a paper-grade external estimate, freeze the protocol and evaluate once on a newly reserved later chronological period.

## Calibration and threshold selection

Temperature scaling is fitted on global validation logits only. The validation evaluator records raw and calibrated thresholds; the reported operating threshold is selected from a three-point-smoothed validation F1 sweep. Final-test labels are not used to fit temperature, choose a threshold, select a round, or tune a hyperparameter. For the valid formal-DP candidate, the thresholds are raw=0.00252348 and calibrated=0.0310866.

Calibration does not make the score distribution inherently well-calibrated under distribution shift, and selecting a threshold on one validation split may still overfit that split. The smoothed sweep reduces single-point threshold noise but does not remove that limitation.

## ZK circuit compatibility

The final layer's shape and parameter keys are unchanged: `Linear(32, 1)`, `output_layer.0.weight`, and `output_layer.0.bias`. The proof statement constrains a sample-count-weighted arithmetic relation for 33 quantized output-layer values and their Poseidon commitment. It does not prove full-model training, client data integrity, model accuracy, or metric truthfulness. No circuit dimensions are modified by the ablation work.

The current full pytest run passed all 48 tests, including 8 Groth16 tests with proof generation, verification, and tamper rejection. A separate proof generated from the archived selected round-8 aggregate (4 clients × 33 parameters; 639,989 total samples) passed both immediate pairing verification and the standalone verifier. See `docs/TEST_RESULTS.md`; this verifies circuit compatibility for that archived aggregate, not model accuracy or the invalid SMOTE final-test attempt.

## Reproducibility

Structured records are stored in `evaluation/ablation_results.json`, with the readable candidate comparison in `evaluation/ablation_summary.md`. Validation and final-test fields are kept separate. The finalized candidate archives include the frozen checkpoint, selected-round artifacts, and final-evaluation audit JSON. Failed startup attempts remain recorded; they are not candidate scores.

See `COMMANDS.md` for current CLI invocations and `docs/CURRENT_PIPELINE_RESULTS.md` for the canonical metric table and record status.
