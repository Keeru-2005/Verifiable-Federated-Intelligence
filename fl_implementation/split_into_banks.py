import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
np.random.seed(42)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming processed_partition_best.csv is in fl_implementation
    data_path = os.path.join(current_dir, "processed_partition_best.csv")
    
    if not os.path.exists(data_path):
        # Fallback to data dir
        data_path = os.path.join(current_dir, "..", "data", "processed_partition_best.csv")
        
    logging.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    logging.info("Splitting into train and test sets (80/20)...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["is_laundering"], random_state=42
    )

    data_out_dir = os.path.join(current_dir, "data")
    os.makedirs(data_out_dir, exist_ok=True)

    test_output_path = os.path.join(data_out_dir, "global_test.csv")
    test_df.to_csv(test_output_path, index=False)
    logging.info(f"Global Test Set: {len(test_df)} records saved to {test_output_path}")

    # Partitioning train set into 4 shards
    num_nodes = 4
    banks = np.array_split(train_df.sample(frac=1, random_state=42), num_nodes)
    
    logging.info("Partitioning train data into 4 shards...")
    
    for i, bank in enumerate(banks, 1):
        out_path = os.path.join(data_out_dir, f"bank_{i}.csv")
        bank.to_csv(out_path, index=False)
        pos_ratio = bank['is_laundering'].mean()
        logging.info(f"Bank Node {i}: {len(bank)} transactions. Saved to {out_path} | Distribution: {pos_ratio*100:.2f}% positive")

    logging.info("✅ Canonical data split complete.")

if __name__ == "__main__":
    main()