import os
import json
import re

# Directory containing cloned datasets
data_dir = "hf_git_clones"
output_dir = "preprocessed_data"
os.makedirs(output_dir, exist_ok=True)

def clean_and_normalize_text(text):
    # Remove extra whitespace and newlines
    text = re.sub(r"\s+", " ", text.strip())
    # Normalize quotes and special characters
    text = text.replace("\u00a0", " ").replace("\u201c", '"').replace("\u201d", '"')
    return text

# Preprocess the mental_health_counseling_conversations dataset
def preprocess_mental_health_counseling():
    input_file = os.path.join(data_dir, "mental_health_counseling_conversations", "combined_dataset.json")
    output_file = os.path.join(output_dir, "mental_health_counseling_preprocessed.json")

    try:
        print(f"Preprocessing {input_file}...")
        preprocessed_data = []
        with open(input_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Extract and clean relevant fields (e.g., Context and Response)
                    if "Context" in entry and "Response" in entry:
                        preprocessed_data.append({
                            "context": clean_and_normalize_text(entry["Context"]),
                            "response": clean_and_normalize_text(entry["Response"])
                        })
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {e}")

        with open(output_file, "w") as f:
            json.dump(preprocessed_data, f, indent=4)

        # Print two data points for verification
        if len(preprocessed_data) >= 2:
            print("Sample Data Point 1:", preprocessed_data[0])
            print("Sample Data Point 2:", preprocessed_data[1])

        print(f"Preprocessed data saved to {output_file}")
    except Exception as e:
        print(f"Failed to preprocess {input_file}: {e}")

if __name__ == "__main__":
    print("Starting data preprocessing...")
    preprocess_mental_health_counseling()
    print("Data preprocessing complete.")