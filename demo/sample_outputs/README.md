# Demo sample outputs

These files are demonstration fixtures for the demo workflow. They are not the current canonical training/evaluation records and must not be cited as current model performance. In particular, the model/proof/input files are a matched sample set and should not be overwritten piecemeal with outputs from another run.

Use the current saved pipeline records for performance statements:

- `evaluation/ablation_summary.md` — current validation candidates and valid final-test result.
- `evaluation/finalized_candidates/bce_pos_weight_25_seed42/final_evaluation.json` — valid new final-test record.
- `evaluation/finalized_candidates/smote_tomek_005_seed42/final_evaluation.json` — invalid test attempt retained only for audit; do not cite or rerun.
- `evaluation/comparison_results.json` — historical centralized result and federated validation history.

See `../../docs/CURRENT_PIPELINE_RESULTS.md` for split counts, confusion matrices, and claim limitations.
