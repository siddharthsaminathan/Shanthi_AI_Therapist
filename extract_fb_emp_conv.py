from empathetic_dialogues import EmpatheticDialogues
import os
import pandas as pd
from datasets import load_dataset
from datasets import Dataset


# Initialize the dataset builder
dataset_builder = EmpatheticDialogues()

# Download and prepare the dataset
dataset_builder.download_and_prepare()

# Load the dataset
data = dataset_builder.as_dataset()

# Access splits
train_data = data["train"]
valid_data = data["validation"]
test_data = data["test"]

# Inspect the data
for example in train_data:
    print(example)
    break

# Create output directory
output_dir = "empathetic_dialogues"
os.makedirs(output_dir, exist_ok=True)

# Save splits as CSV files
train_df = pd.DataFrame(train_data)
valid_df = pd.DataFrame(valid_data)
test_df = pd.DataFrame(test_data)

train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
valid_df.to_csv(os.path.join(output_dir, "valid.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

print(f"Datasets saved to {output_dir}")