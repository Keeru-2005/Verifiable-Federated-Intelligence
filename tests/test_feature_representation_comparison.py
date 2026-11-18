import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.mark.parametrize("representation", ["standardized_raw", "robust_raw"])
def test_feature_transformers_fit_only_on_training_rows(representation):
    from evaluation.compare_feature_representations import transform_feature_splits

    train = pd.DataFrame({"a": [0.0, 2.0], "b": [10.0, 14.0]})
    validation = pd.DataFrame({"a": [100.0], "b": [-100.0]})
    outer_test = pd.DataFrame({"a": [-100.0], "b": [100.0]})

    transformed_train, transformed_validation, transformed_test = transform_feature_splits(
        representation,
        train,
        validation,
        outer_test,
    )

    assert np.isfinite(transformed_train).all()
    assert np.isfinite(transformed_validation).all()
    assert np.isfinite(transformed_test).all()
    np.testing.assert_array_equal(transformed_train, [[-1.0, -1.0], [1.0, 1.0]])
    np.testing.assert_array_equal(transformed_validation, [[99.0, -56.0]])
    assert not np.array_equal(transformed_validation, transformed_test)


def test_unknown_feature_representation_is_rejected():
    from evaluation.compare_feature_representations import transform_feature_splits

    with pytest.raises(ValueError, match="Unknown feature representation"):
        transform_feature_splits("pca", np.zeros((2, 2)), np.zeros((1, 2)), np.zeros((1, 2)))
