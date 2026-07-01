import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load processed dataset
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data", "processed_partition_best.csv")
df = pd.read_csv(data_path)

# Split into train and test
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["is_laundering"], random_state=42
)

# Save test set
test_output_path = os.path.join(current_dir, "data", "global_test.csv")
test_df.to_csv(test_output_path, index=False)
print(f"Global Test Set: {len(test_df)} records saved to {test_output_path}")

# Split train into 4 banks
banks = np.array_split(train_df.sample(frac=1, random_state=42), 4)

for i, bank in enumerate(banks):
    bank_output_path = os.path.join(current_dir, "data", f"bank_{i+1}.csv")
    bank.to_csv(bank_output_path, index=False)
    print(f"Bank {i+1}: {len(bank)} records saved to {bank_output_path}")

print("✅ Data split complete.")