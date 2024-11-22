import os
import json
import torch
from train_lstm import train_lstm
from dataset_lstm import EventDataset
from model_lstm import LSTMModel
from utils_lstm import collate_fn
from evaluate_lstm import evaluate_model
import matplotlib.pyplot as plt
import pandas as pd
import logging


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Set device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    sample_data_path = "final_model_lstm/train.csv"
    if not os.path.exists(sample_data_path):
        raise FileNotFoundError(f"Dataset file {sample_data_path} not found.")

    # Dynamically create event vocabulary
    sample_data = pd.read_csv(sample_data_path)
    event_vocab = {event: idx for idx, event in enumerate(sample_data.columns) if event != "cycle"}
    vocab_size = len(event_vocab)
    print("Event Vocabulary:", event_vocab)
    print("Vocabulary Size:", vocab_size)

    # Configuration
    config = {
        "batch_size": 64,
        "embed_size": 128,
        "hidden_size": 256,
        "num_layers": 2,
        "vocab_size": vocab_size,
        "learning_rate": 1e-4,
        "epochs": 10,
        "device": device,
        "verbose": True,  # Enable detailed output in evaluation
    }

    # Load datasets
    dataset_paths = {
        "train": "final_model_lstm/train.csv",
        "val": "final_model_lstm/val.csv",
        "test": "final_model_lstm/test.csv",
    }
    train_dataset = EventDataset(dataset_paths["train"], event_vocab)
    val_dataset = EventDataset(dataset_paths["val"], event_vocab)
    test_dataset = EventDataset(dataset_paths["test"], event_vocab)

    print(f"Number of sequences in Train Dataset: {len(train_dataset)}")
    print(f"Number of sequences in Validation Dataset: {len(val_dataset)}")
    print(f"Number of sequences in Test Dataset: {len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn
    )

    # Initialize model
    model = LSTMModel(
        vocab_size=config["vocab_size"],
        embed_size=config["embed_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
    ).to(config["device"])

    # Train model
    training_losses, validation_losses = train_lstm(model, train_loader, val_loader, config)

    # Save the trained model
    model_save_path = "results/trained_lstm_model.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved at {model_save_path}")

    # Save loss curves
    loss_curve_path = "results/lstm_loss_curve.png"
    os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)
    plot_losses(training_losses, validation_losses, save_path=loss_curve_path)

    # Evaluate model on test set
    print("Starting evaluation...")

    # Debugging: Check test loader contents
    print(f"Number of batches in test_loader: {len(test_loader)}")
    for batch in test_loader:
        print(f"Sample batch shape: Inputs={batch[0].shape}, Targets={batch[1].shape}, Mask={batch[2].shape}")
        break  # Inspect only the first batch

    # Perform evaluation
    evaluation_metrics = evaluate_model(model, test_loader, config)
    print("Evaluation completed successfully.")
    print(f"Evaluation Metrics: {evaluation_metrics}")

    # Save evaluation metrics
    results_path = "results/lstm_evaluation_metrics.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    try:
        with open(results_path, "w") as f:
            json.dump(evaluation_metrics, f, indent=4)
        print(f"Evaluation metrics saved at {results_path}")
    except Exception as e:
        print(f"Error saving evaluation metrics: {e}")

    # Print metrics
    print("\nTest Set Evaluation Metrics:")
    for metric, value in evaluation_metrics.items():
        if metric == "event_metrics":
            print("Event-wise Metrics:")
            for event, scores in value.items():
                print(f"{event}: Precision={scores['precision']:.4f}, Recall={scores['recall']:.4f}, F1={scores['f1']:.4f}")
        else:
            print(f"{metric.capitalize()}: {value:.4f}")


def plot_losses(training_losses, validation_losses, save_path="results/lstm_loss_curve.png"):
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
    main()
