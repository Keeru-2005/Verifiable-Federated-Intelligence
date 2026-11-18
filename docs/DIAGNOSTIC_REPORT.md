# Round-10 Pipeline Diagnostic Report

**Generated:** 2026-11-18T07:02:26.739646Z  
**Preprocessing run:** `abc4b575614845cdb99a25f3f91aa979`  
**Scope:** saved model, training partition, and validation split only. The diagnostic does not open or evaluate `global_test.csv`.

## Artifact provenance

The saved Round-10 record identifies validation F1 **71.4617%**, threshold **0.0023786351**, and confusion counts `TP=154, FP=60, FN=63, TN=108740` over 109,017 validation rows. The saved final record separately identifies `final_test_once` F1 **68.0115%** and `TP=236, FP=156, FN=66, TN=217576`. These records are distinct; notably, the often-cited `FP=156, FN=66` belong to the final-test record, not validation.

The model checkpoint's validation predictions at the recorded threshold yield F1 **71.4617%** and confusion counts `TP=154, FP=60, FN=63, TN=108740`. Checkpoint agreement with the round record: **True**.

### Validation logits (pre-sigmoid)

| min | median | 95th | 99.9th | max |
|---:|---:|---:|---:|---:|
| -14.834484 | -13.369751 | -11.71168 | 2.8349291 | 7.278739 |

### Validation probabilities

| min | median | 95th | 99.9th | max |
|---:|---:|---:|---:|---:|
| 3.6096515e-07 | 1.5616736e-06 | 8.1974462e-06 | 0.94453432 | 0.99931037 |


| Validation class | Samples | Median logit | 95th logit | Median probability | 95th probability |
|---|---:|---:|---:|---:|---:|
| 0 (majority) | 108,800 | -13.371806 | -11.734602 | 1.5584673e-06 | 8.0116834e-06 |
| 1 (minority) | 217 | 1.3530194 | 6.5858018 | 0.79462278 | 0.99862207 |

The aggregate median is dominated by negatives because the validation prevalence is only 0.1991%; it is not representative of the positive-score distribution. The majority-score bulk is compressed near zero while a high-score tail remains. The max-F1 threshold is therefore a validation operating point under this score distribution, not evidence that probabilities are calibrated. The DP clipping audit below indicates disproportionate positive-gradient clipping for the measured checkpoint/shard, but this single-point diagnostic cannot establish that DP clipping alone caused the score distribution.

## Opacus per-sample gradient clipping

The reported audit samples naturally occurring examples from client shard 1 and computes per-sample gradient norms for the saved checkpoint using the recorded positive loss weight and configured clipping bound. A sample is classified as clipped when its unclipped norm exceeds `max_grad_norm`.

| Class | Samples | Median norm | 95th-percentile norm | Above clip bound |
|---|---:|---:|---:|---:|
| 0 (majority) | 256 | 2.30521e-05 | 0.00016356 | 0.00% |
| 1 (minority) | 256 | 19.7665 | 1010.74 | 69.92% |

Clip bound: `1.0`; positive loss weight: `10.0`. Norms are measured at the saved checkpoint on sampled examples from one client shard; they are not a full training-trajectory clipping estimate.

## PCA and feature signal

PCA has 32 inputs and 32 components, retaining an explained-variance-ratio sum of 100.00000000%. Its smallest component variance ratio is `2.36e-31` and smallest eigenvalue is `7.552e-30`. With 32 inputs and 32 components PCA is not dimensionality reduction. The mutual-information ranking is a diagnostic ranking on a natural-prevalence training sample, not a feature-selection result.

The mutual-information estimate uses 50,000 shuffled training rows (64 positives; no validation or test labels). Top PCA axes:

| PCA feature | Mutual information |
|---|---:|
| pca_14 | 0.0046866647 |
| pca_31 | 0.004349079 |
| pca_18 | 0.0035299974 |
| pca_21 | 0.0027894163 |
| pca_10 | 0.0027114144 |
| pca_23 | 0.002304269 |
| pca_22 | 0.0022320602 |
| pca_9 | 0.0021157868 |
| pca_6 | 0.002106304 |
| pca_24 | 0.0017196245 |

The existing paired rolling-origin comparison found that RobustScaler on all 32 raw causal features outperformed the PCA baseline by an average 2.81 F1 percentage points across three historical development windows. This supports further validation-only testing of retained topological ratios, but is not a fresh blind estimate. PCA already retains 32 dimensions; any raw-feature candidate must keep the classifier representation at 32 inputs and preserve the final `Linear(32, 1)` layer.

## Validation error analysis

There are 60 validation false positives and 63 validation false negatives at the saved threshold.

### Highest-scoring false positives

| Validation row | Probability | Margin from threshold | PCA feature values (first six) |
|---:|---:|---:|---|
| 1939 | 0.99736398 | 0.99498535 | pca_1=-0.599, pca_2=0.8233, pca_3=-0.9151, pca_4=-1.606, pca_5=0.9184, pca_6=1.621 |
| 21244 | 0.99731332 | 0.99493469 | pca_1=0.2884, pca_2=1.261, pca_3=0.6258, pca_4=0.9505, pca_5=-0.6526, pca_6=0.4422 |
| 64865 | 0.99699771 | 0.99461908 | pca_1=-0.1676, pca_2=1.208, pca_3=-0.7634, pca_4=-0.3154, pca_5=-1.027, pca_6=1.94 |
| 20489 | 0.996566 | 0.99418736 | pca_1=-0.3178, pca_2=0.5621, pca_3=-0.4356, pca_4=-1.451, pca_5=0.03351, pca_6=0.9017 |
| 67385 | 0.9961592 | 0.99378056 | pca_1=-0.295, pca_2=1.812, pca_3=-0.03487, pca_4=-0.4963, pca_5=-0.1595, pca_6=1.581 |

### Lowest-scoring false negatives

| Validation row | Probability | Margin from threshold | PCA feature values (first six) |
|---:|---:|---:|---|
| 13689 | 8.6752004e-07 | -0.0023777676 | pca_1=1.113, pca_2=1.044, pca_3=0.3136, pca_4=0.2128, pca_5=0.5426, pca_6=0.01757 |
| 85243 | 1.8469749e-06 | -0.0023767881 | pca_1=0.6038, pca_2=0.5845, pca_3=0.5041, pca_4=0.4406, pca_5=-0.7353, pca_6=0.1065 |
| 72077 | 1.8557304e-06 | -0.0023767794 | pca_1=0.8487, pca_2=0.808, pca_3=-1.968, pca_4=0.5623, pca_5=-2.116, pca_6=2.233 |
| 62487 | 2.5443815e-06 | -0.0023760907 | pca_1=0.7375, pca_2=0.6236, pca_3=0.1616, pca_4=0.3966, pca_5=-0.7349, pca_6=0.2128 |
| 66094 | 3.0971526e-06 | -0.002375538 | pca_1=1.245, pca_2=0.6955, pca_3=0.1987, pca_4=0.5964, pca_5=-1.135, pca_6=0.4354 |

## Interpretation and constraints

- The Round-10 JSON, not the prose summary, is the source of the validation values above. Reconcile documentation before citing results.
- This audit is descriptive and does not tune the model, threshold, calibration, or feature pipeline.
- No final-test data was loaded by this script.
- The Groth16 output layer remains a 32-to-1 linear head (32 weights plus one bias); this diagnostic makes no model changes.
