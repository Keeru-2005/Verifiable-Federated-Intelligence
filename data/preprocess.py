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

# Import our topological extraction engine
from graph_features import extract_features

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

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    viz_dir = create_visualizations_dir(project_root)
    
    data_path = os.path.join(base_dir, "amlnet.csv")
    
    # ---------------------------------------------------------
    # 1. Load Data & Rename for Graph Compatibility
    # ---------------------------------------------------------
    logging.info(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError("amlnet.csv not found!")
    else:
        df = pd.read_csv(data_path)
        
    # Standardize column names dynamically based on PaySim/AMLNet generic structures
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
    
    # Mock a txn_id if absent
    if 'txn_id' not in df.columns:
        df['txn_id'] = range(len(df))
        
    # ---------------------------------------------------------
    # 2. Extract Topological Graph Features (BEFORE dropping strings)
    # ---------------------------------------------------------
    logging.info("Extracting topological graphs from entire dataset. This may take several minutes...")
    df = extract_features(df)
    
    # ---------------------------------------------------------
    # 3. Strip Strings & Isolate Target
    # ---------------------------------------------------------
    # Now that we calculated PageRank using the strings, we safely strip them out!
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    
    if 'is_laundering' in numeric_df.columns:
        X = numeric_df.drop(['is_laundering', 'txn_id', 'timestamp'], axis=1, errors='ignore')
        y = numeric_df['is_laundering']
        feature_names = X.columns
    else:
        raise ValueError("Target column 'is_laundering' not found.")

    X.fillna(0, inplace=True)

    # ---------------------------------------------------------
    # 4. Data Visualization (BEFORE)
    # ---------------------------------------------------------
    logging.info("Generating Before-SMOTE visualization...")
    plot_class_distribution(y, "Class Imbalance (Before SMOTE)", os.path.join(viz_dir, "before_smote.png"))
    
    # ---------------------------------------------------------
    # 5. Data Pre-processing: StandardScaler
    # ---------------------------------------------------------
    logging.info("Applying StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ---------------------------------------------------------
    # 6. Data Pre-processing: SMOTE Variations
    # ---------------------------------------------------------
    ratios = {
        "30_70": 30 / 70,
        "40_60": 40 / 60,
        "50_50": 50 / 50
    }
    
    best_ratio_name = "50_50" # Selecting 50:50 as the best dataset version for balanced representation
    
    for ratio_name, target_ratio in ratios.items():
        logging.info(f"Applying Hybrid Balancing (UnderSampler to 500k + SMOTE target {ratio_name.replace('_', ':')})...")
        
        # Step 1: Reduce legitimate transactions to firmly 500,000. Retain all laundering data.
        current_minority_count = int(sum(y == 1))
        under = RandomUnderSampler(sampling_strategy={0: 500000, 1: current_minority_count}, random_state=42)
        
        # Step 2: SMOTE the laundering transactions for the final leap to the target_ratio
        over = SMOTE(sampling_strategy=target_ratio, k_neighbors=3, random_state=42)
        
        pipeline = Pipeline(steps=[('u', under), ('o', over)])
        X_resampled, y_resampled = pipeline.fit_resample(X_scaled, y)
        
        # ---------------------------------------------------------
        # 7. Data Visualization (AFTER)
        # ---------------------------------------------------------
        logging.info(f"Generating After-SMOTE visualization for {ratio_name}...")
        plot_class_distribution(
            y_resampled, 
            f"Class Balance (After SMOTE {ratio_name.replace('_', ':')})", 
            os.path.join(viz_dir, f"after_smote_{ratio_name}.png")
        )
        
        # ---------------------------------------------------------
        # 8. Reconstruct Fully Interpretable DataFrame
        # ---------------------------------------------------------
        logging.info(f"Reconstructing DataFrame preserving features for {ratio_name}...")
        processed_df = pd.DataFrame(X_resampled, columns=feature_names)
        processed_df['is_laundering'] = y_resampled
        
        # SMOTE appends synthetic samples at the end — shuffle to mix classes throughout
        processed_df = processed_df.sample(frac=1, random_state=42).reset_index(drop=True)
        logging.info(f"Shuffled dataset for {ratio_name} — class 0: {(processed_df['is_laundering']==0).sum()}, class 1: {(processed_df['is_laundering']==1).sum()}")
        
        # ---------------------------------------------------------
        # 9. Storage
        # ---------------------------------------------------------
        output_csv = os.path.join(base_dir, f"processed_partition_{ratio_name}.csv")
        logging.info(f"Saving {ratio_name} explicitly interpreted partitioned dataframe to {output_csv}...")
        processed_df.to_csv(output_csv, index=False)
        
        # Identify the best dataset version and save it distinctly
        if ratio_name == best_ratio_name:
            best_output_csv = os.path.join(base_dir, "processed_partition_best.csv")
            logging.info(f"Saving Best Dataset Version ({ratio_name}) to {best_output_csv}...")
            processed_df.to_csv(best_output_csv, index=False)
            
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    logging.info("Saving Scaler object via joblib...")
    joblib.dump(scaler, scaler_path)
    # Notice: No PCA object saved!
    
    logging.info("Graph-Enhanced Pre-processing Complete!")

if __name__ == "__main__":
    main()
