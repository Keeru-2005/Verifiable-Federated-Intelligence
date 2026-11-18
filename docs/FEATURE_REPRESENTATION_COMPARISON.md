# Feature Representation Comparison

**Status:** completed three-fold DP federated comparison. The outer periods precede the saved final-test period, but were used during prior project development; results are not a fresh blind estimate.

## Protocol

The current representation starts with 32 causal, graph-informed features, fits a `StandardScaler`, retains all 32 PCA components, and then fits a second scaler on those components. The two alternatives preserve the original feature axes:

- **Standardized raw:** training-fold-fitted `StandardScaler`, no PCA.
- **Robust raw:** training-fold-fitted `RobustScaler`, no PCA.

For each rolling-origin fold, causal features were constructed from past timestamps only. Scalers were fitted on that fold's training rows; validation and outer rows were transformed with those fitted scalers. The existing model, DP-SGD noise multiplier (0.5), positive loss weight (10), learning rate (0.002), seed (42), and 10-round limit were held fixed. Round and threshold were selected using only inner validation; each representation's outer period was evaluated once.

## Results

| Representation | Fold 1 F1 | Fold 2 F1 | Fold 3 F1 | Mean F1 (sample SD) | Mean AP | Mean F1 change vs PCA |
|---|---:|---:|---:|---:|---:|---:|
| PCA32 baseline | 68.97% | 58.70% | 68.10% | 65.25% (5.70 pp) | 68.55% | — |
| Standardized raw | 55.12% | 60.51% | 64.33% | 59.99% (4.63 pp) | 58.12% | -5.27 pp |
| Robust raw | 72.19% | 63.21% | 68.79% | **68.06% (4.53 pp)** | **69.69%** | **+2.81 pp** |

Robust raw features had higher outer F1 than PCA in all three windows (+3.22, +4.51, and +0.69 percentage points). Their pooled confusion-count F1 was 67.95%, versus 65.65% for PCA. Robust scaling increased pooled recall (67.49% vs. 58.74%) while reducing pooled precision (68.42% vs. 74.39%); it found more positives but also produced more false positives.

The standardized-raw alternative was not reliable: it lost 13.85 points in the first fold, gained 1.82 in the second, and lost 3.77 in the third. Its mean F1 and AP were below the PCA baseline.

## Interpretation and limits

The evidence makes **RobustScaler on the causal raw features a promising candidate**, not a proven production improvement. Its gains repeated across all three windows, but the experiment used one seed and development periods that have already informed the project. Each outer period contained only 79, 115, or 172 positive cases. The result does not justify claiming a new blind-test score. The current valid new final-test record is BCE `pos_weight=25` at F1 0.679137; the higher validation SMOTE-Tomek candidate has an invalid one-time test attempt due to a checkpoint/threshold mismatch. See `CURRENT_PIPELINE_RESULTS.md`.

PCA is not reducing the feature count here: 32 inputs map to 32 components. The saved canonical PCA artifact has matrix rank 32, but its smallest explained variance is approximately `7.55e-30`; the post-PCA scaler's smallest scale is approximately `2.75e-15`. That near-degenerate direction is then normalized, which could magnify numerical noise. This is a plausible conditioning concern, not proof that it caused the observed metric differences.

Keep the current PCA pipeline as the canonical configuration for now. If choosing a candidate for further development, test RobustScaler across multiple seeds and retain its per-window results. Do not select or tune it using the already-consulted canonical final-test period.

The complete fold metrics, thresholds, selected rounds, and confusion counts are in [the JSON report](../evaluation/feature_representation_comparison_20261117T180059Z.json). Re-run the comparison with:

```powershell
python evaluation\compare_feature_representations.py
```
