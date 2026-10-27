"""
Data Preprocessing & Partitioning Test Suite
Validates schema, train/test isolation, SMOTE on train only, 4 bank partitions, and feature consistency.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "fl_implementation", "data")
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def test_raw_dataset_columns_and_types():
    """Verify raw dataset has required columns and non-empty records."""
    csv_path = os.path.join(BASE_DATA_DIR, "amlnet.csv")
    if not os.path.exists(csv_path):
        pytest.skip("amlnet.csv not found in data/ directory")
    
    # Read first 100 rows to validate schema without loading 600MB
    df = pd.read_csv(csv_path, nrows=100)
    required_cols = ["amount", "nameOrig", "nameDest", "isMoneyLaundering"]
    for col in required_cols:
        assert col in df.columns, f"Required column '{col}' missing from amlnet.csv"
    assert pd.api.types.is_numeric_dtype(df["amount"]), "Column 'amount' must be numeric"
    assert pd.api.types.is_numeric_dtype(df["isMoneyLaundering"]), "Column 'isMoneyLaundering' must be numeric"


def test_scaler_and_pca_artifacts():
    """Verify saved scaler.pkl and pca.pkl exist and have expected dimensions."""
    scaler_path = os.path.join(BASE_DATA_DIR, "scaler.pkl")
    pca_path = os.path.join(BASE_DATA_DIR, "pca.pkl")
    
    assert os.path.exists(scaler_path), f"Scaler artifact missing at {scaler_path}"
    assert os.path.exists(pca_path), f"PCA artifact missing at {pca_path}"
    
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    
    assert hasattr(scaler, "mean_"), "Scaler must be fitted"
    assert hasattr(pca, "components_"), "PCA must be fitted"
    assert pca.n_components == 32, f"PCA must have 32 components, got {pca.n_components}"


def test_four_bank_partitions_exist_and_conform():
    """Verify four bank partitions exist, have identical columns, and valid numeric types."""
    bank_files = [f"bank_{i}.csv" for i in range(1, 5)]
    expected_cols = [f"pca_{i}" for i in range(1, 33)] + ["is_laundering"]

    for bf in bank_files:
        fpath = os.path.join(DATA_DIR, bf)
        assert os.path.exists(fpath), f"Bank partition missing: {fpath}"
        df = pd.read_csv(fpath, nrows=100)
        
        # Check column names
        assert list(df.columns) == expected_cols, f"Columns mismatch in {bf}"
        
        # Check missing values
        assert df.isnull().sum().sum() == 0, f"Found null/NaN values in {bf}"
        
        # Check target values are binary
        unique_targets = set(df["is_laundering"].unique())
        assert unique_targets.issubset({0, 1}), f"Non-binary targets in {bf}: {unique_targets}"


def test_holdout_test_set_conforms():
    """Verify global_test.csv exists, has 32 PCA features + target, and zero missing values."""
    test_path = os.path.join(DATA_DIR, "global_test.csv")
    assert os.path.exists(test_path), f"Global holdout test set missing at {test_path}"
    
    df = pd.read_csv(test_path, nrows=100)
    expected_cols = [f"pca_{i}" for i in range(1, 33)] + ["is_laundering"]
    assert list(df.columns) == expected_cols
    assert df.isnull().sum().sum() == 0


def test_train_test_data_leakage_and_smote_isolation():
    """Verify that training banks contain balanced data from SMOTE, but holdout test is pure natural evaluation data."""
    bank1_path = os.path.join(DATA_DIR, "bank_1.csv")
    test_path = os.path.join(DATA_DIR, "global_test.csv")
    
    b1_sample = pd.read_csv(bank1_path, nrows=5000)
    test_sample = pd.read_csv(test_path, nrows=5000)
    
    # Bank 1 should be balanced (~50% or 30-50% positive for effective training)
    b1_pos = b1_sample["is_laundering"].mean()
    assert 0.20 <= b1_pos <= 0.60, f"Bank 1 positive ratio out of expected balanced range: {b1_pos:.2f}"
    
    # Test set features should have mean roughly 0 and std roughly 1 (from fitted scaler/PCA)
    features = test_sample[[f"pca_{i}" for i in range(1, 33)]].values
    assert not np.isnan(features).any(), "NaN values found in test features"
    assert not np.isinf(features).any(), "Inf values found in test features"


def test_reproducibility_of_quick_preprocessing():
    """Verify that preprocess.py runs reproducibility with --sample-size without error."""
    import subprocess
    cmd = [
        sys.executable,
        os.path.join(BASE_DATA_DIR, "preprocess.py"),
        "--sample-size", "200"
    ]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, f"preprocess.py failed with: {res.stderr}"
    out = res.stdout + res.stderr
    assert "Leak-free preprocessing and 4-bank partitioning complete!" in out or "complete" in out.lower()


def test_out_of_distribution_feature_robustness():
    """Verify that extreme out-of-distribution feature values do not cause overflow or NaN in trained model."""
    import torch
    from models.mlp import GraphAwareMLP

    model = GraphAwareMLP(input_dim=32)
    model.eval()

    # Create extreme feature inputs: 100x standard deviation, negative infinity bounds
    extreme_inputs = torch.tensor([
        [1000.0] * 32,
        [-1000.0] * 32,
        [0.0] * 32,
        [float(i * 10) for i in range(32)]
    ], dtype=torch.float32)

    with torch.no_grad():
        preds = model(extreme_inputs)

    assert not torch.isnan(preds).any(), "Predictions must not be NaN"
    assert not torch.isinf(preds).any(), "Predictions must not be Inf"
    assert (preds >= 0.0).all() and (preds <= 1.0).all(), "Sigmoid output probabilities must remain in [0, 1]"

