"""
Optional helper for re-partitioning an already-preprocessed training file.

The canonical pipeline is now ``data/preprocess.py``: it splits raw transactions
first, fits all preprocessing on train only, writes the untouched holdout, and
partitions the natural training rows into bank shards.

This script therefore only re-shards a *natural* training partition.  It never
creates ``global_test.csv`` from training data (that was the old behaviour and it
made the "holdout" overlap the training pool).
"""

import json
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
np.random.seed(42)


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    manifest_path = os.path.join(project_root, "data", "preprocess_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("pipeline_version") != "causal-v3" or manifest.get("status") != "complete":
            raise RuntimeError(
                "Existing preprocess manifest is not from the causal-v3 pipeline. "
                "Rerun `python data/preprocess.py` before re-partitioning."
            )
    else:
        logging.warning(
            "No preprocess_manifest.json found. The training CSV may contain "
            "post-SMOTE synthetic rows; rerun data/preprocess.py for a leak-free split."
        )

    data_path = os.path.join(project_root, "data", "processed_partition_best.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found. Run `python data/preprocess.py` first."
        )

    logging.info(f"Loading natural training data from {data_path}")
    df = pd.read_csv(data_path)

    num_nodes = 4
    shuffled = df.sample(frac=1, random_state=42)
    banks = np.array_split(shuffled, num_nodes)

    data_out_dir = os.path.join(current_dir, "data")
    os.makedirs(data_out_dir, exist_ok=True)

    logging.info("Partitioning natural training data into 4 shards (no SMOTE here)...")
    for i, bank in enumerate(banks, 1):
        out_path = os.path.join(data_out_dir, f"bank_{i}.csv")
        bank.to_csv(out_path, index=False)
        pos_ratio = bank["is_laundering"].mean()
        logging.info(
            f"Bank Node {i}: {len(bank)} natural transactions. Saved to {out_path} | "
            f"Distribution: {pos_ratio * 100:.2f}% positive"
        )

    logging.info("[SUCCESS] Bank sharding complete. global_test.csv was left untouched.")


if __name__ == "__main__":
    main()
