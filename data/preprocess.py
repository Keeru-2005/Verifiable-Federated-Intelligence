import sys
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

# Add project root directory to path for imports
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our topological extraction engine using workspace relative import
from data.graph_features import extract_features

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

import argparse
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def main():
    parser = argparse.ArgumentParser(description="VFI Data Preprocessing Pipeline (Leak-Free)")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional sample size for fast testing/benchmarking")
    parser.add_argument("--ratio", type=str, default="50_50", choices=["30_70", "40_60", "50_50"], help="SMOTE balance ratio for training data")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    viz_dir = create_visualizations_dir(project_root)
    
    data_path = os.path.join(base_dir, "amlnet.csv")
    
    # ---------------------------------------------------------
    # 1. Load Data & Rename for Graph Compatibility
    # ---------------------------------------------------------
    logging.info(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}!")
    
    df = pd.read_csv(data_path)
    
    if args.sample_size and len(df) > args.sample_size:
        logging.info(f"Sampling {args.sample_size} records for quick pipeline run...")
        # Stratified sample if possible
        if 'isMoneyLaundering' in df.columns:
            laundering = df[df['isMoneyLaundering'] == 1]
            normal = df[df['isMoneyLaundering'] == 0]
            n_laundering = min(len(laundering), max(10, int(args.sample_size * 0.1)))
            n_normal = args.sample_size - n_laundering
            df = pd.concat([
                laundering.sample(n=n_laundering, random_state=42),
                normal.sample(n=n_normal, random_state=42)
            ]).sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
        
    rename_map = {
        'nameOrig': 'sender',
        'nameDest': 'receiver',
        'step': 'timestamp',
        'isMoneyLaundering': 'is_laundering'
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    
    # Drop data leaking fallback targets to prevent ML cheating
    leaky_cols = ['isFraud', 'fraud_probability', 'laundering_typology', 'isFlaggedFraud']
    df.drop(columns=[col for col in leaky_cols if col in df.columns], inplace=True)
    
    if 'txn_id' not in df.columns:
        df['txn_id'] = range(len(df))
        
    # ---------------------------------------------------------
    # 2. Extract Topological Graph Features
    # ---------------------------------------------------------
    logging.info("Extracting topological graph features...")
    df = extract_features(df)
    
    # ---------------------------------------------------------
    # 3. Strip Strings & Isolate Target
    # ---------------------------------------------------------
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    
    if 'is_laundering' in numeric_df.columns:
        X = numeric_df.drop(['is_laundering', 'txn_id', 'timestamp'], axis=1, errors='ignore')
        y = numeric_df['is_laundering']
    else:
        raise ValueError("Target column 'is_laundering' not found.")

    X.fillna(0, inplace=True)

    # ---------------------------------------------------------
    # 4. Strict Train/Test Split BEFORE Scaling, PCA, or Resampling (NO DATA LEAKAGE)
    # ---------------------------------------------------------
    logging.info("Splitting dataset into train (80%) and holdout test (20%) prior to any fitting...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"Train set: {len(X_train_raw)} rows (positives: {int(y_train.sum())}) | Test set: {len(X_test_raw)} rows (positives: {int(y_test.sum())})")

    # Visualize class distribution
    logging.info("Generating Before-SMOTE visualization...")
    plot_class_distribution(y_train, "Train Set Class Imbalance (Before SMOTE)", os.path.join(viz_dir, "before_smote.png"))
    
    # ---------------------------------------------------------
    # 5. Fit StandardScaler & PCA on TRAIN ONLY
    # ---------------------------------------------------------
    logging.info("Fitting StandardScaler on training data only...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    n_components = min(32, X_train_scaled.shape[1])
    logging.info(f"Fitting PCA ({n_components} dimensions) on training data only...")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    pca_columns = [f"pca_{i+1}" for i in range(n_components)]
    
    # ---------------------------------------------------------
    # 6. Apply SMOTE ONLY to Training Data
    # ---------------------------------------------------------
    ratios = {
        "30_70": 30 / 70,
        "40_60": 40 / 60,
        "50_50": 50 / 50
    }
    target_ratio = ratios.get(args.ratio, 50 / 50)
    current_minority = int(sum(y_train == 1))
    current_majority = int(sum(y_train == 0))
    logging.info(f"Applying SMOTE ({args.ratio}) to TRAINING data only (minority count: {current_minority})...")

    k_neighbors = min(3, max(1, current_minority - 1))
    if current_minority > 1:
        over = SMOTE(sampling_strategy=target_ratio, k_neighbors=k_neighbors, random_state=42)
        if current_majority > 500000:
            under = RandomUnderSampler(sampling_strategy={0: 500000, 1: current_minority}, random_state=42)
            pipeline = Pipeline(steps=[('u', under), ('o', over)])
        else:
            pipeline = Pipeline(steps=[('o', over)])
        X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train_pca, y_train)
    else:
        logging.warning("Not enough minority samples for SMOTE, using raw training set.")
        X_train_resampled, y_train_resampled = X_train_pca, y_train

    # Reconstruct Train DataFrame
    train_df = pd.DataFrame(X_train_resampled, columns=pca_columns)
    train_df['is_laundering'] = y_train_resampled.values if hasattr(y_train_resampled, 'values') else y_train_resampled
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Reconstruct Test DataFrame (Pure, natural holdout test set with NO synthetic samples!)
    test_df = pd.DataFrame(X_test_pca, columns=pca_columns)
    test_df['is_laundering'] = y_test.values if hasattr(y_test, 'values') else y_test
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ---------------------------------------------------------
    # 7. Visualization & Storage
    # ---------------------------------------------------------
    plot_class_distribution(
        train_df['is_laundering'], 
        f"Train Class Balance (After SMOTE {args.ratio})", 
        os.path.join(viz_dir, f"after_smote_{args.ratio}.png")
    )

    # Save processed training dataset
    best_output_csv = os.path.join(base_dir, "processed_partition_best.csv")
    logging.info(f"Saving balanced training partition to {best_output_csv} ({len(train_df)} rows)...")
    train_df.to_csv(best_output_csv, index=False)

    # Save test dataset directly to fl_implementation/data/global_test.csv
    fl_data_dir = os.path.join(project_root, "fl_implementation", "data")
    os.makedirs(fl_data_dir, exist_ok=True)
    test_output_csv = os.path.join(fl_data_dir, "global_test.csv")
    logging.info(f"Saving leak-free global test set to {test_output_csv} ({len(test_df)} natural transactions)...")
    test_df.to_csv(test_output_csv, index=False)

    # Partition training data into 4 bank shards
    num_nodes = 4
    banks = np.array_split(train_df, num_nodes)
    for i, bank in enumerate(banks, 1):
        bank_csv = os.path.join(fl_data_dir, f"bank_{i}.csv")
        bank.to_csv(bank_csv, index=False)
        pos_ratio = bank['is_laundering'].mean()
        logging.info(f"Saved Bank {i} shard: {len(bank)} rows to {bank_csv} (positives: {pos_ratio*100:.2f}%)")

    # Save Scaler and PCA
    joblib.dump(scaler, os.path.join(base_dir, "scaler.pkl"))
    joblib.dump(pca, os.path.join(base_dir, "pca.pkl"))
    logging.info("[SUCCESS] Leak-free preprocessing and 4-bank partitioning complete!")

if __name__ == "__main__":
    main()
