import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import optuna

# Check if MPS is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load dataset from CSV files under empathetic_dialogues
train_data = pd.read_csv("/Users/siddharthsaminathan/Downloads/Gomathi/empathetic_dialogues/train.csv")
test_data = pd.read_csv("/Users/siddharthsaminathan/Downloads/Gomathi/empathetic_dialogues/test.csv")
valid_data = pd.read_csv("/Users/siddharthsaminathan/Downloads/Gomathi/empathetic_dialogues/valid.csv")

# Combine train and valid datasets for training
data = pd.concat([train_data, valid_data])

# Prepare the dataset for Zephyr
def preprocess_data(data):
    # Rename columns to match expected format
    data = data.rename(columns={"prompt": "text", "context": "label"})
    return data

# Preprocess the data
train_data = preprocess_data(data)
test_data = preprocess_data(test_data)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# Load Zephyr tokenizer and model from local directory
local_model_path = "/Users/siddharthsaminathan/Downloads/Gomathi/zephyr"  # Replace with the actual path to your local model
tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    local_model_path, 
    num_labels=len(train_data["label"].unique()),
    local_files_only=True
).to(device)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Define metrics for evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predicted_labels = predictions.argmax(axis=1)
    return {
        "accuracy": (predicted_labels == labels).mean(),
    }

# Define the hyperparameter search space
def hyperparameter_search(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
        "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 1000),
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Perform hyperparameter search
best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    n_trials=10,  # Number of trials to run
    hp_space=hyperparameter_search,
)

# Print the best hyperparameters
print("Best hyperparameters:", best_run.hyperparameters)

# Train the model with the best hyperparameters
trainer.train()

# Evaluate the model
predictions = trainer.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(axis=1)

# Print classification report
print(classification_report(test_dataset["label"], predicted_labels))