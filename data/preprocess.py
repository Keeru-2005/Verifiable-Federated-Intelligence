import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
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
    # To save time in the demo, we take the first 50,000 txns to avoid massive graph buildup
    if len(df) > 50000:
        logging.info("Truncating to 50k transactions to keep computation times reasonable for the demo.")
        df = df.head(50000)
        
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
    # 6. Data Pre-processing: SMOTE
    # ---------------------------------------------------------
    logging.info("Applying SMOTE (Targeting 30% minority ratio)...")
    target_minority_ratio = 30 / 70
    smote = SMOTE(sampling_strategy=target_minority_ratio, k_neighbors=3, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # ---------------------------------------------------------
    # 7. Data Visualization (AFTER)
    # ---------------------------------------------------------
    logging.info("Generating After-SMOTE visualization...")
    plot_class_distribution(y_resampled, "Class Balance (After SMOTE)", os.path.join(viz_dir, "after_smote.png"))
    
    # ---------------------------------------------------------
    # 8. Reconstruct Fully Interpretable DataFrame
    # ---------------------------------------------------------
    logging.info(f"Reconstructing DataFrame fully preserving {len(feature_names)} readable features...")
    processed_df = pd.DataFrame(X_resampled, columns=feature_names)
    processed_df['is_laundering'] = y_resampled
    
    # ---------------------------------------------------------
    # 9. Storage
    # ---------------------------------------------------------
    output_csv = os.path.join(base_dir, "processed_partition.csv")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    
    logging.info(f"Saving explicitly interpreted partitioned dataframe to {output_csv}...")
    processed_df.to_csv(output_csv, index=False)
    
    logging.info("Saving Scaler object via joblib...")
    joblib.dump(scaler, scaler_path)
    # Notice: No PCA object saved!
    
    logging.info("Graph-Enhanced Pre-processing Complete!")

if __name__ == "__main__":
    main()
