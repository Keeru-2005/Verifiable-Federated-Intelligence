"""
Data Preprocessing & Partitioning Test Suite
Validates the leak-free contract:
  * raw split happens before graph feature fitting
  * graph features are label-invariant on the holdout
  * scalers/PCA are fit on train only
  * bank shards are natural (SMOTE happens inside each client)
  * the global holdout is untouched and naturally imbalanced
"""
import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "fl_implementation", "data")
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MANIFEST_PATH = os.path.join(BASE_DATA_DIR, "preprocess_manifest.json")


def _manifest():
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def test_raw_dataset_columns_and_types():
    """Verify raw dataset has required columns and non-empty records."""
    csv_path = os.path.join(BASE_DATA_DIR, "amlnet.csv")
    if not os.path.exists(csv_path):
        pytest.skip("amlnet.csv not found in data/ directory")

    df = pd.read_csv(csv_path, nrows=100)
    required_cols = ["amount", "nameOrig", "nameDest", "isMoneyLaundering"]
    for col in required_cols:
        assert col in df.columns, f"Required column '{col}' missing from amlnet.csv"
    assert pd.api.types.is_numeric_dtype(df["amount"]), "Column 'amount' must be numeric"
    assert pd.api.types.is_numeric_dtype(df["isMoneyLaundering"]), "Column 'isMoneyLaundering' must be numeric"


def test_scaler_and_pca_artifacts():
    """Verify scaler/PCA/post-PCA scaler exist and were fitted."""
    scaler_path = os.path.join(BASE_DATA_DIR, "scaler.pkl")
    pca_path = os.path.join(BASE_DATA_DIR, "pca.pkl")
    post_scaler_path = os.path.join(BASE_DATA_DIR, "post_pca_scaler.pkl")

    if not all(os.path.exists(path) for path in (scaler_path, pca_path, post_scaler_path)):
        pytest.skip("Preprocessing scaler/PCA artifacts are not available")

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    post_scaler = joblib.load(post_scaler_path)

    assert hasattr(scaler, "mean_"), "Scaler must be fitted"
    assert hasattr(pca, "components_"), "PCA must be fitted"
    assert hasattr(post_scaler, "mean_"), "Post-PCA scaler must be fitted"
    assert pca.n_components == 32, f"PCA must have 32 components, got {pca.n_components}"


def test_four_bank_partitions_exist_and_conform():
    """Verify four bank partitions exist, have identical columns, and valid numeric types."""
    bank_files = [f"bank_{i}.csv" for i in range(1, 5)]
    expected_cols = [f"pca_{i}" for i in range(1, 33)] + ["is_laundering"]

    if not all(os.path.exists(os.path.join(DATA_DIR, bf)) for bf in bank_files):
        pytest.skip("Preprocessed bank partitions are not available")

    for bf in bank_files:
        fpath = os.path.join(DATA_DIR, bf)
        assert os.path.exists(fpath), f"Bank partition missing: {fpath}"
        df = pd.read_csv(fpath)

        assert list(df.columns) == expected_cols, f"Columns mismatch in {bf}"
        assert df.isnull().sum().sum() == 0, f"Found null/NaN values in {bf}"
        assert set(df["is_laundering"].unique()).issubset({0, 1}), f"Non-binary targets in {bf}"


def test_holdout_test_set_conforms():
    """Verify global_test.csv exists, has 32 PCA features + target, and zero missing values."""
    test_path = os.path.join(DATA_DIR, "global_test.csv")
    if not os.path.exists(test_path):
        pytest.skip("Preprocessed final-test split is not available")

    df = pd.read_csv(test_path)
    expected_cols = [f"pca_{i}" for i in range(1, 33)] + ["is_laundering"]
    assert list(df.columns) == expected_cols
    assert df.isnull().sum().sum() == 0
    assert set(df["is_laundering"].unique()).issubset({0, 1})


def test_bank_shards_preserve_natural_training_rows():
    """
    Bank shards must contain the natural training rows; imbalance is handled
    with weighted loss rather than synthetic resampling.
    """
    manifest = _manifest()
    if manifest.get("pipeline_version") != "causal-v3" or manifest.get("status") != "complete":
        pytest.skip("Completed causal-v3 preprocess manifest not available")
    if not all(os.path.exists(os.path.join(DATA_DIR, f"bank_{i}.csv")) for i in range(1, 5)):
        pytest.skip("Preprocessed bank partitions are not available")

    bank_frames = [pd.read_csv(os.path.join(DATA_DIR, f"bank_{i}.csv")) for i in range(1, 5)]
    total_rows = sum(len(b) for b in bank_frames)
    total_pos = sum(int(b["is_laundering"].sum()) for b in bank_frames)

    assert total_rows == manifest["graph_train_rows"], (
        f"Bank rows ({total_rows}) != natural training rows ({manifest['graph_train_rows']}); "
        "bank shards appear to contain resampled data."
    )
    assert total_pos == manifest["train_positives"], (
        f"Bank positives ({total_pos}) != natural training positives ({manifest['train_positives']})"
    )


def test_holdout_matches_manifest_counts_and_is_untouched():
    """Holdout counts must match the manifest exactly (no SMOTE / no row dropping)."""
    manifest = _manifest()
    if manifest.get("pipeline_version") != "causal-v3" or manifest.get("status") != "complete":
        pytest.skip("Completed causal-v3 preprocess manifest not available")
    test_path = os.path.join(DATA_DIR, "global_test.csv")
    if not os.path.exists(test_path):
        pytest.skip("Preprocessed final-test split is not available")

    # Graph extraction must not silently drop rows from any chronological period.
    assert manifest["raw_rows"] == (
        manifest["raw_train_rows"]
        + manifest["raw_validation_rows"]
        + manifest["raw_test_rows"]
    )
    assert manifest["graph_train_rows"] == manifest["raw_train_rows"]
    assert manifest["graph_validation_rows"] == manifest["raw_validation_rows"]
    assert manifest["graph_test_rows"] == manifest["raw_test_rows"]

    test_df = pd.read_csv(test_path)
    assert len(test_df) == manifest["graph_test_rows"]
    assert int(test_df["is_laundering"].sum()) == manifest["test_positives"]
    assert int((test_df["is_laundering"] == 0).sum()) == manifest["test_negatives"]

    assert abs(test_df["is_laundering"].mean() - manifest["test_prevalence"]) < 1e-9


def test_chronological_split_reports_missing_class_support():
    from data.preprocess import chronological_train_validation_test_split

    df = pd.DataFrame({
        "timestamp": np.repeat([0, 1, 2], 4),
        "txn_id": range(12),
        "is_laundering": [0, 1, 0, 1] + [1] * 8,
    })

    with pytest.raises(
        ValueError,
        match=r"Chronological validation split \(1\.\.1\) has normal=0, laundering=4",
    ):
        chronological_train_validation_test_split(df, validation_size=0.1, test_size=0.2)


def test_graph_features_ignore_holdout_labels():
    """
    Regression test for the original leakage bug.

    Flipping every holdout label must not change a single graph feature.
    """
    from data.graph_features import GraphFeatureExtractor

    rng = np.random.RandomState(7)
    n = 240
    accounts = [f"A{i}" for i in range(50)]
    df = pd.DataFrame({
        "txn_id": range(n),
        "sender": rng.choice(accounts, n),
        "receiver": rng.choice(accounts, n),
        "timestamp": rng.randint(0, 25, n),
        "amount": rng.exponential(500, n) + 1,
        "is_laundering": (rng.rand(n) < 0.12).astype(int),
    })
    train = df.iloc[:180].reset_index(drop=True)
    holdout = df.iloc[180:].reset_index(drop=True)

    extractor = GraphFeatureExtractor(use_gat=True, gat_epochs=5)
    extractor.fit_transform(train)

    features_a = extractor.transform(holdout.drop(columns=["is_laundering"]))
    flipped = holdout.copy()
    flipped["is_laundering"] = 1 - flipped["is_laundering"]
    features_b = extractor.transform(flipped.drop(columns=["is_laundering"]))

    shared_cols = [c for c in features_a.columns if c in features_b.columns]
    pd.testing.assert_frame_equal(
        features_a[shared_cols].reset_index(drop=True),
        features_b[shared_cols].reset_index(drop=True),
    )


def test_graph_pruning_rules_ignore_labels():
    """The label-free pruning rules must be identical when labels are flipped."""
    from data.graph_features import GraphFeatureExtractor

    rng = np.random.RandomState(11)
    n = 200
    accounts = [f"B{i}" for i in range(40)]
    train = pd.DataFrame({
        "txn_id": range(n),
        "sender": rng.choice(accounts, n),
        "receiver": rng.choice(accounts, n),
        "timestamp": rng.randint(0, 20, n),
        "amount": rng.exponential(300, n) + 1,
        "is_laundering": (rng.rand(n) < 0.1).astype(int),
    })

    ext_a = GraphFeatureExtractor(use_gat=False)
    ext_a.fit(train)
    thresholds_a = dict(ext_a.thresholds)

    flipped = train.copy()
    flipped["is_laundering"] = 1 - flipped["is_laundering"]
    ext_b = GraphFeatureExtractor(use_gat=False)
    ext_b.fit(flipped)

    assert thresholds_a == ext_b.thresholds


def test_client_keeps_local_splits_natural_with_class_weighted_loss(tmp_path, monkeypatch):
    """Client training keeps natural labels; imbalance is handled by weighted loss."""
    from fl_implementation import client as client_module
    from fl_implementation.client import AMLClient

    rng = np.random.RandomState(42)
    feature_columns = [f"pca_{i}" for i in range(1, 33)]
    labels = np.zeros(1000, dtype=np.int64)
    labels[rng.choice(len(labels), size=50, replace=False)] = 1
    data = pd.DataFrame(rng.normal(size=(len(labels), len(feature_columns))), columns=feature_columns)
    data["is_laundering"] = labels
    data_path = tmp_path / "bank_1.csv"
    data.to_csv(data_path, index=False)

    monkeypatch.setattr(client_module, "DATA_FILE", str(data_path))
    monkeypatch.setattr(client_module, "ENABLE_DP", False)
    monkeypatch.setattr(
        client_module,
        "_load_manifest",
        lambda: {
            "pipeline_version": "causal-v3",
            "status": "complete",
            "feature_columns": feature_columns,
        },
    )

    client = AMLClient(client_id="1")
    train_y = client.y_train.numpy()
    test_y = client.y_test.numpy()

    assert len(client.train_dataset) == 800
    assert len(client.test_dataset) == 200
    assert float(train_y.mean()) == pytest.approx(float(labels.mean()))
    assert float(test_y.mean()) == pytest.approx(float(labels.mean()))


def test_reproducibility_of_quick_preprocessing(tmp_path):
    """Verify the quick pipeline completes using an isolated, time-stratified sample.

    Both classes occur at every timestamp so chronological validation and test
    periods are representative. Artefacts stay under the temporary directory.
    """
    rng = np.random.RandomState(17)
    rows = []
    for step in range(20):
        for index in range(10):
            rows.append({
                "step": 0,
                "metadata": f"{{'timestamp': datetime.datetime(2025, 1, {step + 1}, 0, 0, 0)}}",
                "amount": float(rng.uniform(1, 1000)),
                "nameOrig": f"sender_{rng.randint(0, 40)}",
                "nameDest": f"receiver_{rng.randint(0, 40)}",
                "isMoneyLaundering": int(index < 2),
                "oldbalanceOrg": float(rng.uniform(0, 5000)),
                "newbalanceOrig": float(rng.uniform(0, 5000)),
                "hour": step % 24,
                "day_of_week": step % 7,
                "day_of_month": step % 28 + 1,
                "month": step % 12 + 1,
            })
    input_path = tmp_path / "synthetic_transactions.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    cmd = [
        sys.executable,
        os.path.join(BASE_DATA_DIR, "preprocess.py"),
        "--input", str(input_path),
        "--sample-size", "200",
        "--output-dir", str(tmp_path),
    ]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, f"preprocess.py failed with: {res.stderr[-2000:]}"
    out = res.stdout + res.stderr
    assert "Leak-free preprocessing and 4-bank partitioning complete!" in out
    manifest_path = tmp_path / "preprocess_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    assert manifest["pipeline_version"] == "causal-v3"
    assert manifest["status"] == "complete"
    assert (tmp_path / "fl_data" / "bank_1.csv").exists()
    assert (tmp_path / "fl_data" / "global_test.csv").exists()


def test_out_of_distribution_feature_robustness():
    """Verify that extreme out-of-distribution feature values do not cause overflow or NaN in trained model."""
    import torch
    from models.mlp import GraphAwareMLP

    model = GraphAwareMLP(input_dim=32)
    model.eval()

    extreme_inputs = torch.tensor([
        [1000.0] * 32,
        [-1000.0] * 32,
        [0.0] * 32,
        [float(i * 10) for i in range(32)],
    ], dtype=torch.float32)

    with torch.no_grad():
        preds = model(extreme_inputs)

    assert not torch.isnan(preds).any(), "Predictions must not be NaN"
    assert not torch.isinf(preds).any(), "Predictions must not be Inf"
    assert (preds >= 0.0).all() and (preds <= 1.0).all(), "Sigmoid output probabilities must remain in [0, 1]"
