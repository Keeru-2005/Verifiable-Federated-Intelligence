import sys
import os
import json
import copy
import logging
import argparse
import uuid

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

# Add project root directory to path for imports
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.graph_features import CausalTransactionFeatureExtractor

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

METADATA_TIMESTAMP_PATTERN = (
    r"datetime\.datetime\((?P<year>\d{4}),\s*(?P<month>\d{1,2}),\s*"
    r"(?P<day>\d{1,2}),\s*(?P<hour>\d{1,2}),\s*(?P<minute>\d{1,2}),\s*"
    r"(?P<second>\d{1,2})(?:,\s*(?P<microsecond>\d{1,6}))?\)"
)


def create_visualizations_dir(base_dir):
    viz_dir = os.path.join(base_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    return viz_dir


def plot_class_distribution(y, title, output_path):
    plt.figure(figsize=(8, 6))
    counts = pd.Series(y).value_counts()

    count_0 = counts.get(0, 0)
    count_1 = counts.get(1, 0)

    labels = [f"Legitimate ({count_0})", f"Laundering ({count_1})"]
    sns.barplot(x=labels, y=[count_0, count_1], hue=labels, palette=["#2ecc71", "#e74c3c"])

    plt.title(title)
    plt.ylabel("Transaction Count")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(output_path, dpi=300)
    plt.close()


def stratified_sample(df, sample_size):
    """Sample rows while keeping a workable number of positives for graph/GAT training."""
    if not sample_size or len(df) <= sample_size:
        return df.reset_index(drop=True)

    logging.info(f"Sampling {sample_size} records for quick pipeline run...")
    if 'is_laundering' not in df.columns:
        return df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    laundering = df[df['is_laundering'] == 1]
    normal = df[df['is_laundering'] == 0]

    n_laundering = min(len(laundering), max(10, int(sample_size * 0.1)))
    n_normal = min(len(normal), sample_size - n_laundering)

    sampled = pd.concat([
        laundering.sample(n=n_laundering, random_state=42),
        normal.sample(n=n_normal, random_state=42),
    ])
    return sampled.sample(frac=1, random_state=42).reset_index(drop=True)


def split_features_and_labels(features_df, raw_df):
    """
    Separate X from y using txn_id as the join key.

    This guarantees the feature matrix can never contain the label even if a
    graph helper accidentally returns it.
    """
    if 'txn_id' not in features_df.columns:
        raise ValueError("txn_id missing from graph feature output; cannot align labels safely.")

    label_by_id = raw_df.set_index('txn_id')['is_laundering']
    y = label_by_id.reindex(features_df['txn_id'])
    if y.isna().any():
        raise ValueError("Label alignment failed: some graph feature rows have no matching txn_id.")

    numeric = features_df.select_dtypes(include=[np.number]).copy()
    X = numeric.drop(['is_laundering', 'txn_id', 'timestamp'], axis=1, errors='ignore')
    X = X.fillna(0)
    return X, y.astype(int).reset_index(drop=True)


def metadata_timestamps_to_seconds(metadata):
    """Extract embedded datetime.datetime values as numeric epoch seconds."""
    parts = metadata.str.extract(METADATA_TIMESTAMP_PATTERN)
    required_parts = ["year", "month", "day", "hour", "minute", "second"]
    missing = parts[required_parts].isna().any(axis=1)
    if missing.any():
        raise ValueError(
            f"Could not parse embedded transaction timestamps from metadata for "
            f"{int(missing.sum())} row(s)."
        )

    parts["microsecond"] = parts["microsecond"].fillna("0")
    timestamps = pd.to_datetime(parts.astype("int64"), errors="coerce")
    invalid = timestamps.isna()
    if invalid.any():
        raise ValueError(
            f"Embedded metadata contains {int(invalid.sum())} invalid transaction timestamp(s)."
        )
    return timestamps.astype("int64").astype("float64") / 1_000_000_000


def chronological_train_validation_test_split(df, validation_size, test_size):
    """Split whole timestamps into chronological train, validation, and test periods."""
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation-size and test-size must be positive and sum to less than 1.")
    if "timestamp" not in df or df["timestamp"].isna().any():
        raise ValueError("A complete timestamp column is required for the chronological split.")

    counts = df.groupby("timestamp", sort=True).size()
    timestamps = counts.index.to_list()
    if len(timestamps) < 3:
        raise ValueError("At least three distinct timestamps are required for train/validation/test splits.")
    cumulative = counts.cumsum().to_numpy()
    train_end = int(np.searchsorted(cumulative, len(df) * (1 - validation_size - test_size), side="left")) + 1
    validation_end = int(np.searchsorted(cumulative, len(df) * (1 - test_size), side="left")) + 1
    train_end = min(max(train_end, 1), len(timestamps) - 2)
    validation_end = min(max(validation_end, train_end + 1), len(timestamps) - 1)

    train_times = set(timestamps[:train_end])
    validation_times = set(timestamps[train_end:validation_end])
    test_times = set(timestamps[validation_end:])
    train = df[df["timestamp"].isin(train_times)].copy()
    validation = df[df["timestamp"].isin(validation_times)].copy()
    test = df[df["timestamp"].isin(test_times)].copy()

    for split_name, split_df in (("train", train), ("validation", validation), ("test", test)):
        if split_df.empty or split_df["is_laundering"].nunique() < 2:
            class_counts = split_df["is_laundering"].value_counts()
            normal_count = int(class_counts.get(0, 0))
            positive_count = int(class_counts.get(1, 0))
            if split_df.empty:
                time_range = "empty"
            else:
                time_range = f"{split_df['timestamp'].min()}..{split_df['timestamp'].max()}"
            raise ValueError(
                f"Chronological {split_name} split ({time_range}) has "
                f"normal={normal_count}, laundering={positive_count}; both classes are required."
                + " Provide source data with both classes in each chronological period."
                + " The pipeline will not fall back to a random split."
            )
    return tuple(part.sort_values(["timestamp", "txn_id"], kind="mergesort").reset_index(drop=True)
                 for part in (train, validation, test))


def main():
    parser = argparse.ArgumentParser(description="VFI Data Preprocessing Pipeline (Leak-Free)")
    parser.add_argument("--input", default=os.path.join(base_dir, "amlnet.csv"),
                        help="Path to the raw transaction CSV")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Optional sample size for fast testing/benchmarking")
    parser.add_argument("--validation-size", type=float, default=0.1,
                        help="Fraction of timestamps reserved for model selection")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of latest timestamps reserved for final evaluation")
    parser.add_argument("--num-banks", type=int, default=4, help="Number of bank shards")
    parser.add_argument("--output-dir", default=None,
                        help="Optional directory for all artefacts (used by tests to avoid clobbering a real run)")
    args = parser.parse_args()
    if args.num_banks < 2:
        raise ValueError("num-banks must be at least 2 for a federated simulation.")

    if args.output_dir:
        output_root = os.path.abspath(args.output_dir)
        os.makedirs(output_root, exist_ok=True)
        viz_dir = os.path.join(output_root, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        artifact_dir = output_root
        fl_data_dir = os.path.join(output_root, "fl_data")
        manifest_path = os.path.join(output_root, "preprocess_manifest.json")
    else:
        viz_dir = create_visualizations_dir(project_root)
        artifact_dir = base_dir
        fl_data_dir = os.path.join(project_root, "fl_implementation", "data")
        manifest_path = os.path.join(base_dir, "preprocess_manifest.json")

    run_id = uuid.uuid4().hex
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump({"pipeline_version": "causal-v3", "run_id": run_id, "status": "preprocessing"}, manifest_file)

    data_path = os.path.abspath(args.input)

    # ---------------------------------------------------------
    # 1. Load Data & Rename for Graph Compatibility
    # ---------------------------------------------------------
    logging.info(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}!")

    # Prefer an explicit timestamp, then the embedded transaction datetime, and
    # use step only for datasets without either source.
    header = pd.read_csv(data_path, nrows=0)
    time_source = next(
        (column for column in ("timestamp", "metadata", "step") if column in header.columns),
        None,
    )
    if time_source is None:
        raise ValueError("Input data must contain a 'timestamp', 'metadata' datetime, or 'step' column.")

    # Read only used fields. Metadata is streamed in bounded chunks because it
    # can be large, and is discarded immediately after extracting timestamps.
    wanted_columns = [
        time_source, 'amount', 'nameOrig', 'nameDest', 'isMoneyLaundering',
        'oldbalanceOrg', 'newbalanceOrig', 'hour', 'day_of_week', 'day_of_month', 'month',
    ]
    usecols = [c for c in wanted_columns if c in header.columns]
    if time_source == "metadata":
        data_chunks = []
        for chunk in pd.read_csv(data_path, usecols=usecols, chunksize=50_000):
            chunk["timestamp"] = metadata_timestamps_to_seconds(chunk.pop("metadata"))
            data_chunks.append(chunk)
        if not data_chunks:
            raise ValueError("Input dataset contains no transaction rows.")
        df = pd.concat(data_chunks, ignore_index=True)
    else:
        df = pd.read_csv(data_path, usecols=usecols)

    rename_map = {
        'nameOrig': 'sender',
        'nameDest': 'receiver',
        'isMoneyLaundering': 'is_laundering',
    }
    if time_source == "step":
        rename_map["step"] = "timestamp"
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    logging.info(f"Using {time_source} as the chronological timestamp source.")

    # Drop data leaking fallback targets to prevent ML cheating
    leaky_cols = ['isFraud', 'fraud_probability', 'laundering_typology', 'isFlaggedFraud']
    df.drop(columns=[col for col in leaky_cols if col in df.columns], inplace=True)

    if 'txn_id' not in df.columns:
        df['txn_id'] = range(len(df))

    df = stratified_sample(df, args.sample_size)

    if 'is_laundering' not in df.columns:
        raise ValueError("Target column 'is_laundering' not found.")

    # ---------------------------------------------------------
    # 2. Split into forward-in-time periods before deriving graph features.
    # ---------------------------------------------------------
    logging.info("Creating chronological train/validation/test periods before feature extraction...")
    train_raw, validation_raw, test_raw = chronological_train_validation_test_split(
        df, args.validation_size, args.test_size
    )

    train_pos = int(train_raw['is_laundering'].sum())
    validation_pos = int(validation_raw['is_laundering'].sum())
    test_pos = int(test_raw['is_laundering'].sum())
    logging.info(
        f"Raw train: {len(train_raw)} rows (positives: {train_pos}, negatives: {len(train_raw) - train_pos})"
    )
    logging.info(
        f"Raw validation: {len(validation_raw)} rows (positives: {validation_pos}, negatives: {len(validation_raw) - validation_pos})"
    )
    logging.info(f"Raw final test: {len(test_raw)} rows (positives: {test_pos}, negatives: {len(test_raw) - test_pos})")

    plot_class_distribution(
        train_raw['is_laundering'],
        "Raw Training Split Class Distribution (Before Any Resampling)",
        os.path.join(viz_dir, "train_class_distribution.png"),
    )

    # ---------------------------------------------------------
    # 3. Build causal per-transaction graph features in chronological order.
    # ---------------------------------------------------------
    logging.info("Extracting history-only graph features; each period uses only earlier timestamps...")
    extractor = CausalTransactionFeatureExtractor()
    train_features = extractor.fit_transform(train_raw)
    evaluation_extractor = copy.deepcopy(extractor)
    validation_features = evaluation_extractor.transform(validation_raw.drop(columns=["is_laundering"]))
    test_features = evaluation_extractor.transform(test_raw.drop(columns=["is_laundering"]))

    X_train_raw, y_train = split_features_and_labels(train_features, train_raw)
    X_validation_raw, y_validation = split_features_and_labels(validation_features, validation_raw)
    X_test_raw, y_test = split_features_and_labels(test_features, test_raw)

    # Align holdout columns to the training schema (graph features can legitimately
    # be missing for accounts that never occur in the holdout split).
    X_validation_raw = X_validation_raw.reindex(columns=X_train_raw.columns, fill_value=0)
    X_test_raw = X_test_raw.reindex(columns=X_train_raw.columns, fill_value=0)

    logging.info(
        f"Graph-feature train: {len(X_train_raw)} rows (positives: {int(y_train.sum())}) | "
        f"validation: {len(X_validation_raw)} rows (positives: {int(y_validation.sum())}) | "
        f"test: {len(X_test_raw)} rows (positives: {int(y_test.sum())})"
    )

    # ---------------------------------------------------------
    # 4. Fit StandardScaler, PCA, and post-PCA scaler on TRAIN ONLY
    # ---------------------------------------------------------
    logging.info("Fitting StandardScaler on training data only...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_validation_scaled = scaler.transform(X_validation_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    n_components = min(32, X_train_scaled.shape[1])
    logging.info(f"Fitting PCA ({n_components} dimensions) on training data only...")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_validation_pca = pca.transform(X_validation_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    logging.info("Fitting post-PCA StandardScaler on training data only...")
    post_pca_scaler = StandardScaler()
    X_train_pca = post_pca_scaler.fit_transform(X_train_pca)
    X_validation_pca = post_pca_scaler.transform(X_validation_pca)
    X_test_pca = post_pca_scaler.transform(X_test_pca)

    pca_columns = [f"pca_{i + 1}" for i in range(n_components)]

    # Resampling is deliberately disabled. Client training uses class-weighted loss.
    train_df = pd.DataFrame(X_train_pca, columns=pca_columns)
    train_df['is_laundering'] = y_train.values
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    validation_df = pd.DataFrame(X_validation_pca, columns=pca_columns)
    validation_df['is_laundering'] = y_validation.values
    validation_df = validation_df.reset_index(drop=True)

    test_df = pd.DataFrame(X_test_pca, columns=pca_columns)
    test_df['is_laundering'] = y_test.values
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ---------------------------------------------------------
    # 5. Visualization & Storage
    # ---------------------------------------------------------
    plot_class_distribution(
        train_df['is_laundering'],
        "Natural Training Partition (class weighting; no synthetic resampling)",
        os.path.join(viz_dir, "train_class_distribution_after_pca.png"),
    )

    # Save the natural, leak-free training partition (pre-SMOTE).
    best_output_csv = os.path.join(artifact_dir, "processed_partition_best.csv")
    logging.info(f"Saving natural training partition to {best_output_csv} ({len(train_df)} rows)...")
    train_df.to_csv(best_output_csv, index=False)

    # Save validation and final test separately. Only validation is used during FL.
    os.makedirs(fl_data_dir, exist_ok=True)
    validation_output_csv = os.path.join(fl_data_dir, "global_validation.csv")
    validation_df.to_csv(validation_output_csv, index=False)
    test_output_csv = os.path.join(fl_data_dir, "global_test.csv")
    logging.info(f"Saving untouched global holdout to {test_output_csv} ({len(test_df)} natural transactions)...")
    test_df.to_csv(test_output_csv, index=False)

    # Partition only the earliest, natural training period into simulated clients.
    num_nodes = args.num_banks
    shuffled_train = train_df.sample(frac=1, random_state=42)
    banks = np.array_split(shuffled_train, num_nodes)
    for i, bank in enumerate(banks, 1):
        bank_csv = os.path.join(fl_data_dir, f"bank_{i}.csv")
        bank.to_csv(bank_csv, index=False)
        pos_ratio = bank['is_laundering'].mean()
        logging.info(
            f"Saved Bank {i} shard: {len(bank)} natural rows to {bank_csv} "
            f"(positives: {pos_ratio * 100:.2f}%)"
        )

    # Save preprocessing artefacts
    joblib.dump(scaler, os.path.join(artifact_dir, "scaler.pkl"))
    joblib.dump(pca, os.path.join(artifact_dir, "pca.pkl"))
    joblib.dump(post_pca_scaler, os.path.join(artifact_dir, "post_pca_scaler.pkl"))
    try:
        joblib.dump(extractor, os.path.join(artifact_dir, "graph_extractor.pkl"))
    except Exception as exc:  # pragma: no cover - model objects are not always picklable
        logging.warning(f"Could not persist graph extractor: {exc}")

    manifest = {
        "pipeline_version": "causal-v3",
        "run_id": run_id,
        "status": "complete",
        "random_state": 42,
        "split_strategy": "chronological_whole_timestamp_groups",
        "timestamp_source": time_source,
        "validation_size": args.validation_size,
        "test_size": args.test_size,
        "num_banks": num_nodes,
        "resampling": "none",
        "raw_rows": int(len(df)),
        "sample_size": args.sample_size,
        "sampling_strategy": "label-stratified quick sample" if args.sample_size else "full dataset",
        "raw_train_rows": int(len(train_raw)),
        "raw_validation_rows": int(len(validation_raw)),
        "raw_test_rows": int(len(test_raw)),
        "graph_train_rows": int(len(X_train_raw)),
        "graph_validation_rows": int(len(X_validation_raw)),
        "graph_test_rows": int(len(X_test_raw)),
        "train_positives": int(y_train.sum()),
        "train_negatives": int((y_train == 0).sum()),
        "validation_positives": int(y_validation.sum()),
        "validation_negatives": int((y_validation == 0).sum()),
        "validation_prevalence": float(y_validation.mean()) if len(y_validation) else 0.0,
        "test_positives": int(y_test.sum()),
        "test_negatives": int((y_test == 0).sum()),
        "test_prevalence": float(y_test.mean()) if len(y_test) else 0.0,
        "pca_components": int(n_components),
        "feature_columns": [str(c) for c in pca_columns],
        "graph_feature_columns": [str(c) for c in X_train_raw.columns],
        "leakage_controls": [
            "whole timestamps split chronologically before feature extraction",
            "causal graph features use only transactions from strictly earlier timestamps",
            "validation and test labels are removed before feature transformation",
            "StandardScaler and PCA fitted on train only",
            "post-PCA scaler fitted on train only",
            "no synthetic oversampling; client training uses class-weighted loss",
            "bank shards contain only the earliest training period",
        ],
    }
    manifest_tmp = manifest_path + ".tmp"
    with open(manifest_tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(manifest_tmp, manifest_path)

    logging.info("[SUCCESS] Leak-free preprocessing and 4-bank partitioning complete!")
    logging.info(
        f"[FINAL TEST] {manifest['test_positives']} positives / "
        f"{manifest['test_negatives']} negatives "
        f"(prevalence {manifest['test_prevalence'] * 100:.2f}%)"
    )


if __name__ == "__main__":
    main()
