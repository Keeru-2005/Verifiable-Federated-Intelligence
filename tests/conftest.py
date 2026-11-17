import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def causal_client_data(tmp_path, monkeypatch):
    from fl_implementation import client as client_module

    rng = np.random.RandomState(42)
    feature_columns = [f"pca_{i}" for i in range(1, 33)]
    labels = np.zeros(500, dtype=np.int64)
    labels[rng.choice(len(labels), size=25, replace=False)] = 1
    data = pd.DataFrame(
        rng.normal(size=(len(labels), len(feature_columns))),
        columns=feature_columns,
    )
    data["is_laundering"] = labels
    data_path = tmp_path / "bank_1.csv"
    data.to_csv(data_path, index=False)

    manifest = {
        "pipeline_version": "causal-v3",
        "status": "complete",
        "feature_columns": feature_columns,
    }
    monkeypatch.setattr(client_module, "DATA_FILE", str(data_path))
    monkeypatch.setattr(client_module, "ENABLE_DP", False)
    monkeypatch.setattr(client_module, "_load_manifest", lambda: manifest)
    return data_path
