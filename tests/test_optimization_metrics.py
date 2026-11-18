import numpy as np
import pytest
import torch
from opacus.grad_sample import GradSampleModule

from evaluation.ablation_logging import append_ablation_result
from evaluation.threshold_metrics import fit_temperature, select_smoothed_thresholds
from fl_implementation.client import per_example_loss
from models.mlp import GraphAwareMLP


def test_cost_sensitive_bce_uses_positive_weight_per_example():
    logits = torch.tensor([0.0, 0.0])
    labels = torch.tensor([0.0, 1.0])

    losses = per_example_loss(
        logits, labels, loss_type="bce", positive_weight=10.0
    )

    assert losses.shape == (2,)
    assert torch.isclose(losses[1], losses[0] * 10.0)


def test_focal_loss_is_finite_and_downweights_easy_examples():
    logits = torch.tensor([-8.0, 0.0])
    labels = torch.tensor([0.0, 1.0])

    losses = per_example_loss(
        logits, labels, loss_type="focal", focal_alpha=0.75, focal_gamma=2.0
    )

    assert losses.shape == (2,)
    assert torch.isfinite(losses).all()
    assert losses[0] < losses[1]


def test_focal_loss_produces_per_sample_opacus_gradients():
    model = GraphAwareMLP(input_dim=32, return_logits=True)
    private_model = GradSampleModule(model, loss_reduction="sum")
    features = torch.randn(4, 32)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

    logits = private_model(features).reshape(-1)
    losses = per_example_loss(
        logits, labels, loss_type="focal", focal_alpha=0.75, focal_gamma=2.0
    )
    losses.sum().backward()

    assert all(
        parameter.grad_sample.shape[0] == 4
        for parameter in private_model.parameters()
        if parameter.requires_grad
    )


def test_temperature_and_smoothed_thresholds_are_finite():
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    logits = np.asarray([-3.0, -1.5, 0.2, 0.3, 1.5, 3.0])

    temperature = fit_temperature(logits, labels)
    probabilities = 1.0 / (1.0 + np.exp(-logits / temperature))
    thresholds = select_smoothed_thresholds(labels, probabilities)

    assert np.isfinite(temperature) and temperature > 0
    assert np.isfinite(thresholds["f1_threshold"])
    assert thresholds["smoothing_window"] == 3
    if thresholds["precision_threshold"] is not None:
        assert thresholds["precision_threshold_precision"] >= 0.70


def test_output_head_remains_32_to_1_with_stable_parameter_keys():
    model = GraphAwareMLP(input_dim=32, return_logits=True)

    assert model.output_layer[0].weight.shape == (1, 32)
    assert model.output_layer[0].bias.shape == (1,)
    assert "output_layer.0.weight" in model.state_dict()
    assert model(torch.zeros(2, 32)).shape == (2, 1)


def test_centralized_dataset_loader_does_not_open_final_test(tmp_path):
    import pandas as pd
    from evaluation.compare_baseline import load_datasets

    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    pd.DataFrame({
        "pca_1": [0.0, 1.0, 2.0, 3.0],
        "is_laundering": [0, 1, 0, 1],
    }).to_csv(train_path, index=False)
    pd.DataFrame({
        "pca_1": [0.5, 2.5],
        "is_laundering": [0, 1],
    }).to_csv(validation_path, index=False)

    loaded = load_datasets(train_path, validation_path)

    assert len(loaded) == 6
    assert loaded[0].shape == (4, 1)
    assert loaded[2].shape == (2, 1)


def test_final_test_evaluator_loads_only_when_invoked_and_once(tmp_path, monkeypatch):
    import pandas as pd
    from fl_implementation import server

    test_path = tmp_path / "final_test.csv"
    data = {f"pca_{index}": [0.0, 0.1, 0.9, 1.0] for index in range(1, 33)}
    data["is_laundering"] = [0, 0, 1, 1]
    pd.DataFrame(data).to_csv(test_path, index=False)

    original_read_csv = server.pd.read_csv
    calls = []

    def tracked_read_csv(path, *args, **kwargs):
        calls.append(str(path))
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(server.pd, "read_csv", tracked_read_csv)
    evaluate = server.get_eval_fn(str(test_path), split_name="final_test_once")

    assert callable(evaluate)
    assert calls == []

    model = GraphAwareMLP(input_dim=32)
    _, metrics = evaluate(
        1,
        model.get_parameters(),
        {
            "decision_threshold": 0.5,
            "calibrated_threshold": 0.5,
            "temperature": 1.0,
        },
    )

    assert calls == [str(test_path)]
    assert metrics["support"] == 4
    assert metrics["split"] == "final_test_once"


def test_sweep_result_uses_server_selected_checkpoint_not_raw_ap_max(tmp_path):
    import json
    from evaluation.run_fl_validation_sweep import read_trial_result

    for round_number, average_precision, selected in (
        (6, 0.8007, True),
        (8, 0.80118, True),
        (9, 0.80135, False),
    ):
        record = {
            "round": round_number,
            "run_id": "screening-run",
            "evaluation": "validation",
            "selected_by_validation": selected,
            "metrics": {
                "average_precision": average_precision,
                "f1": 0.75,
            },
        }
        (tmp_path / f"global_model_round_{round_number}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )

    best, rounds = read_trial_result(tmp_path, "screening-run")

    assert best["round"] == 8
    assert best["selected_by_validation"] is True
    assert len(rounds) == 3


def test_finalizer_rejects_mismatched_round_threshold_before_test_load(
    tmp_path, monkeypatch
):
    import json
    from evaluation import finalize_validation_candidate as finalizer

    root = tmp_path
    manifest_path = root / "data" / "preprocess_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"run_id": "manifest-run"}), encoding="utf-8")
    sweep_id = "sweep-1"
    artifact_dir = (
        root / "evaluation" / "manifest-run_sweep-1" / "candidate"
    )
    (artifact_dir / "round_updates").mkdir(parents=True)
    model = GraphAwareMLP(input_dim=32, dropout=0.1)
    torch.save(model.state_dict(), artifact_dir / "best_model.pt")

    validation = {
        "round": 8,
        "f1": 0.75,
        "average_precision": 0.80,
        "auc": 0.99,
        "decision_threshold": 0.35,
        "calibrated_threshold": 0.40,
        "temperature": 1.5,
        "tp": 10,
        "fp": 3,
        "fn": 4,
        "tn": 90,
    }
    round_metrics = dict(validation, decision_threshold=0.30)
    round_record = {
        "evaluation": "validation",
        "run_id": "manifest-run",
        "selected_by_validation": True,
        "metrics": round_metrics,
        "weights": [value.tolist() for value in model.get_parameters()],
    }
    (artifact_dir / "global_model_round_8.json").write_text(
        json.dumps(round_record), encoding="utf-8"
    )
    (artifact_dir / "round_updates" / "round_8.json").write_text(
        json.dumps({
            "round": 8,
            "global_output_layer_parameters": [0.0] * 33,
        }),
        encoding="utf-8",
    )
    sweep_path = root / "evaluation" / "sweep.json"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_path.write_text(json.dumps({
        "run_id": "manifest-run",
        "sweep_id": sweep_id,
        "screening_rounds": 25,
        "local_epochs": 1,
        "candidates": [{
            "loss_type": "bce",
            "positive_loss_weight": 25,
            "resampling": "none",
            "smote_strategy": 0.05,
            "model_dropout": 0.1,
            "artifact_dir": str(artifact_dir),
            "best_validation": validation,
        }],
    }), encoding="utf-8")

    monkeypatch.setattr(finalizer, "ROOT", root)
    monkeypatch.setattr(
        finalizer,
        "get_eval_fn",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Final-test evaluator must not be loaded.")
        ),
    )

    with pytest.raises(ValueError, match="disagree on validation decision_threshold"):
        finalizer.finalize_candidate(
            sweep_path,
            loss_type="bce",
            positive_weight=25,
            resampling="none",
            smote_strategy=0.05,
            output_dir=root / "evaluation" / "finalized",
        )

    assert not (root / "evaluation" / "finalized").exists()


def test_ablation_logging_keeps_validation_and_test_fields_separate(tmp_path):
    record = append_ablation_result(
        tmp_path,
        run_id="screening-1",
        experiment_name="validation_bce",
        hyperparameters={"rounds": 2, "epsilon": 4.2},
        validation_metrics={"f1": 0.7, "average_precision": 0.8},
    )
    append_ablation_result(
        tmp_path,
        run_id="finalized-1",
        experiment_name="finalized_bce_smote_tomek",
        hyperparameters={
            "pos_weight": 1,
            "resampling": "smote_tomek",
            "smote_strategy": 0.05,
            "epsilon": 4.2,
            "delta": 1e-5,
            "dp_accounting_validity": "nominal_only",
        },
        validation_metrics={"f1": 0.7, "average_precision": 0.8},
        final_test_metrics={
            "f1": 0.65,
            "average_precision": 0.75,
            "auc": 0.9,
            "precision": 0.6,
            "recall": 0.7,
            "tp": 7,
            "fp": 4,
            "fn": 3,
            "tn": 86,
            "decision_threshold": 0.3,
            "calibrated_threshold": 0.4,
            "confusion_matrix": [[86, 4], [3, 7]],
        },
    )

    assert record["validation_metrics"]["f1"] == 0.7
    assert record["final_test_metrics"]["f1"] is None
    assert (tmp_path / "evaluation" / "ablation_results.json").exists()
    summary = (tmp_path / "evaluation" / "ablation_summary.md").read_text(
        encoding="utf-8"
    )
    assert "Validation F1" in summary
    assert "Final-test F1" in summary
    assert "Completed validation-only comparison" in summary
    assert "Validation PR-AUC" in summary
    assert "Baseline BCE" in summary
    assert "Finalized candidate comparison" in summary
    assert "nominal_only" in summary
    assert "| 7 | 4 | 3 | 86 |" in summary
