# Training Settings Tuning Results

Date: 2026-11-18

## Scope

This run tuned the existing PCA/32-feature DP-FedAvg MLP at
`DP_NOISE_MULTIPLIER=0.5`, `FedProx mu=0.0`, `local_epochs=1`, and
`seeds=42,43` where completed. The sweep changed positive loss weight,
client learning rate, and training rounds. It did not change the model
architecture or preprocessing representation.

## Completed validation results

| Weight | Learning rate | Rounds | Seed 42 F1 | Seed 43 F1 | Mean F1 | Epsilon at best round |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.001 | 5 | 61.70% | 47.19% | 54.44% | 6.528 |
| 10 | 0.002 | 5 | 65.65% | 54.45% | 60.05% | 6.528 |
| 10 | 0.002 | 10 | 71.46% | not completed | 71.46% (one seed) | 7.752 |
| 25 | 0.001 | 5 | 61.70% | 47.19% | 54.44% | 6.528 |
| 25 | 0.002 | 5 | 65.48% | not completed | 65.48% (one seed) | 6.528 |

The 10-round, learning-rate 0.002 result used F1-threshold selection on
validation and reached precision 71.96%, recall 70.97%, F1 71.46%, and
average precision 71.15% at seed 42. The 10-round run shows that more
training rounds materially improve validation performance relative to
five rounds.

Positive weights 10 and 25 gave nearly identical five-round performance
at learning rate 0.002 for seed 42: 65.65% versus 65.48% F1. Weight 50
and several remaining seed pairs could not complete because the Windows
host exhausted memory while loading/running the four DP clients.

## Privacy accounting

The epsilon values above are Opacus RDP estimates at `delta=1e-5`.
Increasing from five to ten rounds changed the worst completed client
epsilon from 6.528 to 7.752. The privacy budget is therefore not constant
across round counts, and validation gains at ten rounds require accepting
this higher epsilon.

## Final-test status

This tuning note predates the focused validation ablation and the approved
final evaluations, so its status below is superseded. The valid BCE
`pos_weight=25` final-test record reports F1 0.679137 and AP 0.727406 at
epsilon 7.75214, delta 1e-5. The validation-leading SMOTE-Tomek candidate
has no valid final-test score: its one-time attempt paired the round-8 model
with round-9 calibration thresholds. Do not rerun it because the test
partition was already read for that attempt. See
`docs/CURRENT_PIPELINE_RESULTS.md`.

The robust-scaled raw-feature rolling-origin result remains 68.06% mean
outer F1 versus 65.25% for PCA, but it is development evidence rather than
a final-test result.

## Retention recommendation

1. **Retain option 1 threshold tuning:** useful for evaluation and the
   operating threshold, but not as proof of a stronger model.
2. **Retain option 2 robust scaling for the next model candidate:** it
   improved all three paired rolling-origin windows and may preserve
   rare-event signals better than PCA.
3. **Retain the training settings 10 positive weight, 0.002 learning rate,
   0.5 DP noise, five or ten rounds:** learning rate 0.002 is clearly
   better than 0.001, and ten rounds improved validation substantially.
   Retain ten rounds only if epsilon 7.752 at delta 1e-5 is acceptable.

Recommended first combined experiment:
`RobustScaler(raw causal features) + LR 0.002 + weight 10 + sigma 0.5`
with a multi-seed rolling-origin validation sweep and the canonical
final-test period reserved until selection is complete.
