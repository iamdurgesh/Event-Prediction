import os
import torch
from train1 import train_model
from dataset1 import EventDataset
from model1 import TransformerModel
from utils1 import collate_fn
from evaluate import evaluate_model
import pandas as pd
import json
import matplotlib.pyplot as plt

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def hyperparameter_tuning():
    """
    Perform hyperparameter tuning for Transformer model and log results.
    """

    # Set device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Event vocabulary mapping
    sample_data_path = "final_model/train.csv"
    if not os.path.exists(sample_data_path):
        raise FileNotFoundError(f"Dataset file {sample_data_path} not found.")

    sample_data = pd.read_csv(sample_data_path)
    event_vocab = {event: idx for idx, event in enumerate(sample_data.columns) if event != "cycle"}
    vocab_size = len(event_vocab)

    print("Event Vocabulary:", event_vocab)
    print("Vocabulary Size:", vocab_size)

    # Dataset paths
    dataset_paths = {
        "train": "final_model/train.csv",
        "val": "final_model/val.csv",
        "test": "final_model/test.csv",
    }

    train_dataset = EventDataset(dataset_paths["train"], event_vocab)
    val_dataset = EventDataset(dataset_paths["val"], event_vocab)
    test_dataset = EventDataset(dataset_paths["test"], event_vocab)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
    )

    # Hyperparameter combinations to try
    hyperparameters = [
        {"embed_size": 128, "num_heads": 4, "num_layers": 2, "learning_rate": 1e-4},
        {"embed_size": 256, "num_heads": 8, "num_layers": 4, "learning_rate": 5e-4},
        {"embed_size": 128, "num_heads": 4, "num_layers": 6, "learning_rate": 1e-3},
        {"embed_size": 256, "num_heads": 8, "num_layers": 2, "learning_rate": 1e-4},
    ]

    results = []

    for idx, params in enumerate(hyperparameters):
        print(f"\nRunning Experiment {idx + 1}/{len(hyperparameters)} with params: {params}")
        
        # Initialize Model
        model = TransformerModel(
            vocab_size=vocab_size,
            embed_size=params["embed_size"],
            num_heads=params["num_heads"],
            num_layers=params["num_layers"],
            max_len=77,  # Example fixed max_len
        ).to(device)

        # Training configuration
        config = {
            "batch_size": 64,
            "embed_size": params["embed_size"],
            "num_heads": params["num_heads"],
            "num_layers": params["num_layers"],
            "vocab_size": vocab_size,
            "max_len": 77,
            "learning_rate": params["learning_rate"],
            "epochs": 10,
            "device": device,
        }

        # Train Model
        training_losses, validation_losses = train_model(model, train_loader, val_loader, config)

        # Evaluate Model
        evaluation_metrics = evaluate_model(model, test_loader, config)

        # Save Results
        result = {
            "hyperparameters": params,
            "evaluation_metrics": evaluation_metrics,
            "training_losses": training_losses,
            "validation_losses": validation_losses,
        }
        results.append(result)

        # Save model checkpoint
        model_save_path = f"results/transformer_model_{idx + 1}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model for Experiment {idx + 1} saved at {model_save_path}")

        # Save loss curves
        plot_path = f"results/transformer_loss_curve_{idx + 1}.png"
        plot_losses(training_losses, validation_losses, save_path=plot_path)

    # Save all results to a JSON file
    results_path = "results/transformer_hyperparameter_tuning_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"All results saved to {results_path}")


def plot_losses(training_losses, validation_losses, save_path="results/transformer_loss_curve.png"):
    """
    Plot and save training and validation loss curves.

    Args:
        training_losses (list): Average training losses per epoch.
        validation_losses (list): Average validation losses per epoch.
        save_path (str): Path to save the plot.
    """
    if len(training_losses) != len(validation_losses):
        raise ValueError("Mismatch between training and validation loss lengths.")

    epochs = range(1, len(training_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_losses, label="Training Loss", marker="o")
    plt.plot(epochs, validation_losses, label="Validation Loss", marker="x")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Loss curve saved at {save_path}")
    plt.show()


if __name__ == "__main__":
    hyperparameter_tuning()
