import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load processed dataset
df = pd.read_csv("processed_partition.csv")

# Split into train and test
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["is_laundering"], random_state=42
)

# Save test set
test_df.to_csv("global_test.csv", index=False)

# Split train into 4 banks
banks = np.array_split(train_df.sample(frac=1, random_state=42), 4)

for i, bank in enumerate(banks):
    bank.to_csv(f"bank_{i+1}.csv", index=False)
    print(f"Bank {i+1}: {len(bank)} records")

print("✅ Data split complete.")