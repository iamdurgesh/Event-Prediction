import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import EventDataset
from model import TransformerSeq2SeqModel
from train import train_model
from utils import collate_fn

# Dataset paths
dataset_paths = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
}

# Ensure dataset files exist
for name, path in dataset_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name.capitalize()} dataset file {path} not found.")

# Dynamically create event vocabulary
sample_data = pd.read_csv(dataset_paths["train"])  # Load the training dataset
event_columns = [col for col in sample_data.columns if col != "cycle"]  # Exclude the "cycle" column
event_vocab = {event: idx for idx, event in enumerate(event_columns, start=1)}  # Map event names to indices
event_vocab["<PAD>"] = 0  # Add <PAD> token with ID 0
if "<EXCLUDED>" not in event_vocab:
    event_vocab["<EXCLUDED>"] = len(event_vocab)  # Add <EXCLUDED> token for excluded events
vocab_size = len(event_vocab)

print("Event Vocabulary:", event_vocab)
print("Vocabulary Size:", vocab_size)

# Experiment configurations for hyperparameter tuning
experiments = [
    {"embed_size": 128, "num_heads": 4, "num_layers": 2, "learning_rate": 0.0001},
    {"embed_size": 256, "num_heads": 8, "num_layers": 4, "learning_rate": 0.0005},
    {"embed_size": 128, "num_heads": 4, "num_layers": 6, "learning_rate": 0.0010},
    {"embed_size": 256, "num_heads": 8, "num_layers": 2, "learning_rate": 0.0001},
    {"embed_size": 512, "num_heads": 16, "num_layers": 6, "learning_rate": 0.0003},  # Added another experiment
]

# Loop through experiments
for i, exp in enumerate(experiments):
    print(f"Running Experiment {i + 1}/{len(experiments)} with {exp}...")
    config = {
        "batch_size": 32,
        "embed_size": exp["embed_size"],
        "num_heads": exp["num_heads"],
        "num_layers": exp["num_layers"],
        "learning_rate": exp["learning_rate"],
        "vocab_size": vocab_size,
        "max_len": 77,
        "epochs": 10,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "excluded_events": ["event_task_start", "event_task_end"],  # Ensure excluded events are passed
    }

    # Create datasets and dataloaders
    train_dataset = EventDataset(
        dataset_paths["train"],
        event_vocab,
        max_len=config["max_len"],
        excluded_events=config["excluded_events"]
    )
    val_dataset = EventDataset(
        dataset_paths["val"],
        event_vocab,
        max_len=config["max_len"],
        excluded_events=config["excluded_events"]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn
    )

    # Initialize the model
    model = TransformerSeq2SeqModel(
        vocab_size=config["vocab_size"],
        embed_size=config["embed_size"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_len=config["max_len"],
    ).to(config["device"])

    # Train the model
    training_losses, validation_losses = train_model(model, train_loader, val_loader, config)

    # Save the results
    results_path = f"results/experiment_{i + 1}_results.txt"
    os.makedirs("results", exist_ok=True)
    with open(results_path, "w") as f:
        f.write(f"Experiment {i + 1}/{len(experiments)}\n")
        f.write(f"Config: {exp}\n")
        f.write(f"Final Training Loss: {training_losses[-1]:.4f}\n")
        f.write(f"Final Validation Loss: {validation_losses[-1]:.4f}\n")

    print(f"Experiment {i + 1} completed. Results saved to {results_path}.")

print("Hyperparameter tuning completed!")