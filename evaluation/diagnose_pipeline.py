"""Audit the saved VFI round-10 checkpoint without reading the final-test split."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from opacus.grad_sample import GradSampleModule
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.mlp import GraphAwareMLP


LABEL_COLUMN = "is_laundering"
DROP_COLUMNS = {
    LABEL_COLUMN, "sender", "receiver", "timestamp", "nameOrig", "nameDest",
    "_pk", "_edge_key",
}


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Diagnostic values must be nonempty and finite.")
    quantiles = np.percentile(values, [0, 50, 95, 99.9, 100])
    return {
        "min": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p999": float(quantiles[3]),
        "max": float(quantiles[4]),
    }


def load_model(checkpoint_path, input_dim):
    model = GraphAwareMLP(input_dim=input_dim)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_logits(model, features, batch_size):
    logits = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(
                features[start:start + batch_size], dtype=torch.float32
            )
            hidden = model.input_projection(batch)
            hidden = model.residual_block1(hidden)
            hidden = model.residual_block2(hidden)
            hidden = model.representation(hidden)
            logits.append(model.output_layer[0](hidden).reshape(-1).numpy())
    return np.concatenate(logits)


def audit_gradients(model, training_csv, feature_columns, max_per_class, clip_norm,
                    positive_weight, seed):
    frame = pd.read_csv(
        training_csv,
        usecols=feature_columns + [LABEL_COLUMN],
    )
    labels = frame[LABEL_COLUMN].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = {}
    for label in (0, 1):
        indexes = np.flatnonzero(labels == label)
        if indexes.size == 0:
            raise ValueError(f"Training shard has no examples for label {label}.")
        count = min(max_per_class, indexes.size)
        selected[label] = rng.choice(indexes, size=count, replace=False)

    indexes = np.concatenate([selected[0], selected[1]])
    y = torch.as_tensor(labels[indexes], dtype=torch.float32)
    x = torch.as_tensor(
        frame.iloc[indexes][feature_columns].to_numpy(dtype=np.float32)
    )
    gradient_model = GradSampleModule(
        model, loss_reduction="sum", batch_first=True
    )
    gradient_model.train()
    gradient_model.zero_grad(set_to_none=True)
    probabilities = gradient_model(x).reshape(-1)
    per_example_loss = F.binary_cross_entropy(
        probabilities, y, reduction="none"
    )
    weights = torch.where(
        y > 0.5,
        torch.full_like(y, positive_weight),
        torch.ones_like(y),
    )
    (per_example_loss * weights).sum().backward()

    squared_norms = torch.zeros(len(indexes), dtype=torch.float64)
    for parameter in gradient_model.parameters():
        sample_grad = parameter.grad_sample
        if sample_grad is None:
            continue
        flattened = sample_grad.reshape(len(indexes), -1).to(torch.float64)
        squared_norms += flattened.square().sum(dim=1)
        parameter.grad_sample = None

    norms = squared_norms.sqrt().numpy()
    results = {}
    for label in (0, 1):
        class_norms = norms[y.numpy().astype(np.int64) == label]
        results[str(label)] = {
            "samples": int(class_norms.size),
            "norms": percentile_summary(class_norms),
            "fraction_above_clip_norm": float(np.mean(class_norms > clip_norm)),
        }
    results["clip_norm"] = float(clip_norm)
    results["loss_positive_weight"] = float(positive_weight)
    results["source_training_shard"] = str(training_csv)
    results["warning"] = (
        "Norms are measured at the saved checkpoint on sampled examples from one "
        "client shard; they are not a full training-trajectory clipping estimate."
    )
    return results


def audit_pca(pca_path, training_csv, feature_columns, max_rows, seed):
    import joblib

    pca = joblib.load(pca_path)
    columns = feature_columns + [LABEL_COLUMN]
    sample = pd.read_csv(training_csv, usecols=columns, nrows=max_rows)
    labels = sample[LABEL_COLUMN].to_numpy(dtype=np.int64)
    features = sample[feature_columns].to_numpy(dtype=np.float64)
    if np.unique(labels).size != 2:
        raise ValueError(
            "The sampled training prefix must contain both classes for mutual information."
        )
    information = mutual_info_classif(
        features, labels, discrete_features=False, n_neighbors=3, random_state=seed
    )
    ranked = np.argsort(information)[::-1]
    ratios = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    return {
        "sample_rows": int(len(sample)),
        "sample_positives": int(labels.sum()),
        "sample_negatives": int((labels == 0).sum()),
        "sample_strategy": "first rows from the saved training partition, which is shuffled by preprocessing",
        "component_count": int(pca.n_components_),
        "input_feature_count": int(pca.n_features_in_),
        "explained_variance_ratio_sum": float(ratios.sum()),
        "largest_component_variance_ratio": float(ratios.max()),
        "smallest_component_variance_ratio": float(ratios.min()),
        "ranked_mutual_information": [
            {"feature": feature_columns[index], "mutual_information": float(information[index])}
            for index in ranked[:10]
        ],
        "pca_eigenvalue_min": float(np.min(pca.explained_variance_)),
        "pca_eigenvalue_max": float(np.max(pca.explained_variance_)),
        "interpretation": (
            "With 32 inputs and 32 components PCA is not dimensionality reduction. "
            "The mutual-information ranking is a diagnostic ranking on a natural-prevalence "
            "training sample, not a feature-selection result."
        ),
    }


def error_examples(labels, probabilities, threshold, feature_frame, count=5):
    predicted_positive = probabilities >= threshold
    false_positive = np.flatnonzero((labels == 0) & predicted_positive)
    false_negative = np.flatnonzero((labels == 1) & ~predicted_positive)
    fp_order = false_positive[np.argsort(probabilities[false_positive])[::-1]][:count]
    fn_order = false_negative[np.argsort(probabilities[false_negative])][:count]
    pca_columns = [
        column for column in feature_frame.columns if column.startswith("pca_")
    ][:6]

    def rows(indexes):
        output = []
        for index in indexes:
            output.append({
                "row": int(index),
                "probability": float(probabilities[index]),
                "margin_from_threshold": float(probabilities[index] - threshold),
                "largest_abs_pca_features": {
                    column: float(feature_frame.iloc[index][column])
                    for column in pca_columns
                },
            })
        return output

    return {
        "threshold": float(threshold),
        "false_positive_count": int(false_positive.size),
        "false_negative_count": int(false_negative.size),
        "highest_confidence_false_positives": rows(fp_order),
        "highest_confidence_false_negatives": rows(fn_order),
    }


def make_markdown(report):
    logits = report["validation"]["logit_quantiles"]
    probabilities = report["validation"]["probability_quantiles"]
    class_scores = report["validation"]["class_score_quantiles"]
    gradient = report["gradient_clipping"]
    pca = report["pca_signal"]
    saved = report["saved_round_record"]
    final = report["saved_final_record"]

    def quantile_table(title, values):
        return (
            f"### {title}\n\n"
            "| min | median | 95th | 99.9th | max |\n|---:|---:|---:|---:|---:|\n"
            f"| {values['min']:.8g} | {values['median']:.8g} | "
            f"{values['p95']:.8g} | {values['p999']:.8g} | {values['max']:.8g} |\n"
        )

    mi_rows = "\n".join(
        f"| {row['feature']} | {row['mutual_information']:.8g} |"
        for row in pca["ranked_mutual_information"]
    )
    gradient_rows = "\n".join(
        f"| {label} | {values['samples']} | {values['norms']['median']:.6g} | "
        f"{values['norms']['p95']:.6g} | "
        f"{values['fraction_above_clip_norm']:.2%} |"
        for label, values in (("0 (majority)", gradient["0"]), ("1 (minority)", gradient["1"]))
    )
    class_score_rows = "\n".join(
        f"| {name} | {values['samples']:,} | {values['logits']['median']:.8g} | "
        f"{values['logits']['p95']:.8g} | {values['probabilities']['median']:.8g} | "
        f"{values['probabilities']['p95']:.8g} |"
        for name, values in (
            ("0 (majority)", class_scores["0"]),
            ("1 (minority)", class_scores["1"]),
        )
    )

    def error_table(title, examples):
        lines = [
            f"### {title}", "",
            "| Validation row | Probability | Margin from threshold | PCA feature values (first six) |",
            "|---:|---:|---:|---|",
        ]
        for row in examples:
            encoded = ", ".join(
                f"{key}={value:.4g}"
                for key, value in row["largest_abs_pca_features"].items()
            )
            lines.append(
                f"| {row['row']} | {row['probability']:.8g} | "
                f"{row['margin_from_threshold']:.8g} | {encoded} |"
            )
        return "\n".join(lines)

    return f"""# Round-10 Pipeline Diagnostic Report

**Generated:** {report['generated_at']}  
**Preprocessing run:** `{report['run_id']}`  
**Scope:** saved model, training partition, and validation split only. The diagnostic does not open or evaluate `global_test.csv`.

## Artifact provenance

The saved Round-10 record identifies validation F1 **{saved['metrics']['f1']:.4%}**, threshold **{saved['metrics']['decision_threshold']:.8g}**, and confusion counts `TP={saved['metrics']['tp']}, FP={saved['metrics']['fp']}, FN={saved['metrics']['fn']}, TN={saved['metrics']['tn']}` over {saved['metrics']['support']:,} validation rows. The saved final record separately identifies `final_test_once` F1 **{final['metrics']['f1']:.4%}** and `TP={final['metrics']['tp']}, FP={final['metrics']['fp']}, FN={final['metrics']['fn']}, TN={final['metrics']['tn']}`. These records are distinct; notably, the often-cited `FP=156, FN=66` belong to the final-test record, not validation.

The model checkpoint's validation predictions at the recorded threshold yield F1 **{report['validation']['computed_metrics']['f1']:.4%}** and confusion counts `TP={report['validation']['computed_metrics']['tp']}, FP={report['validation']['computed_metrics']['fp']}, FN={report['validation']['computed_metrics']['fn']}, TN={report['validation']['computed_metrics']['tn']}`. Checkpoint agreement with the round record: **{report['validation']['round_record_matches_checkpoint']}**.

{quantile_table("Validation logits (pre-sigmoid)", logits)}
{quantile_table("Validation probabilities", probabilities)}

| Validation class | Samples | Median logit | 95th logit | Median probability | 95th probability |
|---|---:|---:|---:|---:|---:|
{class_score_rows}

The aggregate median is dominated by negatives because the validation prevalence is only {report['validation']['positives'] / report['validation']['rows']:.4%}; it is not representative of the positive-score distribution. The majority-score bulk is compressed near zero while a high-score tail remains. The max-F1 threshold is therefore a validation operating point under this score distribution, not evidence that probabilities are calibrated. The DP clipping audit below indicates disproportionate positive-gradient clipping for the measured checkpoint/shard, but this single-point diagnostic cannot establish that DP clipping alone caused the score distribution.

## Opacus per-sample gradient clipping

The reported audit samples naturally occurring examples from client shard 1 and computes per-sample gradient norms for the saved checkpoint using the recorded positive loss weight and configured clipping bound. A sample is classified as clipped when its unclipped norm exceeds `max_grad_norm`.

| Class | Samples | Median norm | 95th-percentile norm | Above clip bound |
|---|---:|---:|---:|---:|
{gradient_rows}

Clip bound: `{gradient['clip_norm']}`; positive loss weight: `{gradient['loss_positive_weight']}`. {gradient['warning']}

## PCA and feature signal

PCA has {pca['input_feature_count']} inputs and {pca['component_count']} components, retaining an explained-variance-ratio sum of {pca['explained_variance_ratio_sum']:.8%}. Its smallest component variance ratio is `{pca['smallest_component_variance_ratio']:.4g}` and smallest eigenvalue is `{pca['pca_eigenvalue_min']:.4g}`. {pca['interpretation']}

The mutual-information estimate uses {pca['sample_rows']:,} shuffled training rows ({pca['sample_positives']} positives; no validation or test labels). Top PCA axes:

| PCA feature | Mutual information |
|---|---:|
{mi_rows}

The existing paired rolling-origin comparison found that RobustScaler on all 32 raw causal features outperformed the PCA baseline by an average 2.81 F1 percentage points across three historical development windows. This supports further validation-only testing of retained topological ratios, but is not a fresh blind estimate. PCA already retains 32 dimensions; any raw-feature candidate must keep the classifier representation at 32 inputs and preserve the final `Linear(32, 1)` layer.

## Validation error analysis

There are {report['error_analysis']['false_positive_count']} validation false positives and {report['error_analysis']['false_negative_count']} validation false negatives at the saved threshold.

{error_table("Highest-scoring false positives", report['error_analysis']['highest_confidence_false_positives'])}

{error_table("Lowest-scoring false negatives", report['error_analysis']['highest_confidence_false_negatives'])}

## Interpretation and constraints

- The Round-10 JSON, not the prose summary, is the source of the validation values above. Reconcile documentation before citing results.
- This audit is descriptive and does not tune the model, threshold, calibration, or feature pipeline.
- No final-test data was loaded by this script.
- The Groth16 output layer remains a 32-to-1 linear head (32 weights plus one bias); this diagnostic makes no model changes.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "preprocess_manifest.json")
    parser.add_argument("--validation-csv", type=Path,
                        default=ROOT / "fl_implementation" / "data" / "global_validation.csv")
    parser.add_argument("--training-csv", type=Path,
                        default=ROOT / "fl_implementation" / "data" / "bank_1.csv")
    parser.add_argument("--training-partition", type=Path,
                        default=ROOT / "data" / "processed_partition_best.csv")
    parser.add_argument("--pca", type=Path, default=ROOT / "data" / "pca.pkl")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "fl_implementation" / "model.pt")
    parser.add_argument("--round-record", type=Path,
                        default=ROOT / "schemas" / "global_model_round_10.json")
    parser.add_argument("--final-record", type=Path,
                        default=ROOT / "schemas" / "final_evaluation.json")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "DIAGNOSTIC_REPORT.md")
    parser.add_argument("--max-mi-rows", type=int, default=50000)
    parser.add_argument("--gradient-samples-per-class", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_mi_rows < 2 or args.gradient_samples_per_class < 1:
        parser.error("MI row count and gradient sample count must be positive.")
    if args.clip_norm <= 0 or args.positive_weight <= 0 or args.batch_size < 1:
        parser.error("Clip norm, positive weight, and batch size must be positive.")

    manifest = read_json(args.manifest)
    round_record = read_json(args.round_record)
    final_record = read_json(args.final_record)
    if manifest.get("pipeline_version") != "causal-v3" or manifest.get("status") != "complete":
        raise RuntimeError("Diagnostics require completed causal-v3 preprocessing artifacts.")
    if round_record.get("evaluation") != "validation":
        raise RuntimeError("The selected round record must be explicitly labeled validation.")
    if final_record.get("evaluation") != "final_test_once":
        raise RuntimeError("The final record must be explicitly labeled final_test_once.")
    if round_record.get("run_id") != manifest.get("run_id"):
        raise RuntimeError("Round record and preprocessing manifest run IDs do not match.")

    feature_columns = manifest["feature_columns"]
    validation_frame = pd.read_csv(args.validation_csv)
    if LABEL_COLUMN not in validation_frame:
        raise ValueError("Validation CSV does not contain is_laundering.")
    if not set(feature_columns).issubset(validation_frame.columns):
        raise ValueError("Validation CSV does not match the manifest feature schema.")
    labels = validation_frame[LABEL_COLUMN].to_numpy(dtype=np.int64)
    features = validation_frame[feature_columns].to_numpy(dtype=np.float32)
    model = load_model(args.checkpoint, input_dim=len(feature_columns))
    logits = predict_logits(model, features, args.batch_size)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    class_score_quantiles = {}
    for label in (0, 1):
        mask = labels == label
        if not mask.any():
            raise ValueError(f"Validation split has no examples for label {label}.")
        class_score_quantiles[str(label)] = {
            "samples": int(mask.sum()),
            "logits": percentile_summary(logits[mask]),
            "probabilities": percentile_summary(probabilities[mask]),
        }

    from evaluation.threshold_metrics import binary_classification_metrics

    threshold = float(round_record["metrics"]["decision_threshold"])
    computed_metrics = binary_classification_metrics(labels, probabilities, threshold)
    expected_metrics = round_record["metrics"]
    matches = all(
        computed_metrics[key] == expected_metrics[key]
        for key in ("tp", "fp", "fn", "tn")
    )
    gradients = audit_gradients(
        model,
        args.training_csv,
        feature_columns,
        args.gradient_samples_per_class,
        args.clip_norm,
        args.positive_weight,
        args.seed,
    )
    pca_signal = audit_pca(
        args.pca, args.training_partition, feature_columns, args.max_mi_rows, args.seed
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": manifest["run_id"],
        "validation": {
            "rows": int(labels.size),
            "positives": int(labels.sum()),
            "negatives": int((labels == 0).sum()),
            "logit_quantiles": percentile_summary(logits),
            "probability_quantiles": percentile_summary(probabilities),
            "class_score_quantiles": class_score_quantiles,
            "computed_metrics": computed_metrics,
            "round_record_matches_checkpoint": bool(matches),
            "checkpoint": str(args.checkpoint),
            "recorded_threshold": threshold,
        },
        "gradient_clipping": gradients,
        "pca_signal": pca_signal,
        "error_analysis": error_examples(
            labels, probabilities, threshold, validation_frame
        ),
        "saved_round_record": round_record,
        "saved_final_record": final_record,
        "final_test_data_loaded": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(make_markdown(report), encoding="utf-8")
    print(f"Diagnostic report written to {args.report}")
    print(
        f"Validation checkpoint match={matches}; "
        f"F1={computed_metrics['f1']:.4f}; threshold={threshold:.8g}"
    )


if __name__ == "__main__":
    main()
