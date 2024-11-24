
#-------------for duplicated events
# import os
# import torch
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import pandas as pd
# from dataset import EventDataset
# from model import TransformerSeq2SeqModel
# from train import train_model
# from utils import collate_fn
# # from evaluation import evaluate_model  # Use standard evaluation function
# from masked_evaluation import evaluate_model_with_mask


# def load_trained_model(model_path, vocab_size, embed_size, num_heads, num_layers, max_len, device):
#     """
#     Load a pre-trained model from disk.

#     Args:
#         model_path (str): Path to the saved model weights.
#         vocab_size (int): Vocabulary size.
#         embed_size (int): Embedding size.
#         num_heads (int): Number of attention heads.
#         num_layers (int): Number of Transformer layers.
#         max_len (int): Maximum sequence length.
#         device (str): Device to load the model on.

#     Returns:
#         model: The loaded model.
#     """
#     model = TransformerSeq2SeqModel(
#         vocab_size=vocab_size,
#         embed_size=embed_size,
#         num_heads=num_heads,
#         num_layers=num_layers,
#         max_len=max_len,
#     ).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     print(f"Model loaded from {model_path}")
#     return model


# def main():
#     # Set device
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     # Configurations
#     model_path = "model_weights_epoch10.pth"
#     evaluate_only = True  # Set to True to skip training and directly evaluate

#     # Dataset paths
#     # dataset_paths = {
#     #     "train": "train_balanced.csv",
#     #     "val": "model_new/val.csv",
#     #     "test": "model_new/test.csv",
#     # }

#     dataset_paths = {
#         "train": "train.csv",
#         "val": "val.csv",
#         "test": "test.csv",
#     }

#     # Ensure dataset files exist
#     for name, path in dataset_paths.items():
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"{name.capitalize()} dataset file {path} not found.")

#     # Load a sample dataset to extract column names
#     sample_data = pd.read_csv(dataset_paths["train"])  # Load training data for column extraction

#     # Dynamically create event vocabulary
#     event_columns = [col for col in sample_data.columns if col != "cycle"]  # Exclude the "cycle" column
#     event_vocab = {event: idx for idx, event in enumerate(event_columns, start=1)}  # Map event names to indices
#     event_vocab["<PAD>"] = 0  # Add <PAD> token with ID 0
#     vocab_size = len(event_vocab)

#     print("Event Vocabulary:", event_vocab)
#     print("Vocabulary Size:", vocab_size)

#     # Model configuration
#     config = {
#         "batch_size": 32,
#         "embed_size": 256,
#         "num_heads": 2,
#         "num_layers": 8,
#         "vocab_size": vocab_size,
#         "max_len": 77,
#         "learning_rate": 1e-3,
#         "epochs": 10,
#         "device": device,
#     }

#     # Initialize datasets and dataloaders
#     train_dataset = EventDataset(dataset_paths["train"], event_vocab, max_len=config["max_len"])
#     val_dataset = EventDataset(dataset_paths["val"], event_vocab, max_len=config["max_len"])
#     test_dataset = EventDataset(dataset_paths["test"], event_vocab, max_len=config["max_len"])

#     train_loader = DataLoader(
#         train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn
#     )
#     val_loader = DataLoader(
#         val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn
#     )

#     if evaluate_only:
#         print("Loading pre-trained model for evaluation...")
#         model = load_trained_model(
#             model_path=model_path,
#             vocab_size=config["vocab_size"],
#             embed_size=config["embed_size"],
#             num_heads=config["num_heads"],
#             num_layers=config["num_layers"],
#             max_len=config["max_len"],
#             device=config["device"],
#         )

#         # Perform evaluation
#         # print("\nEvaluating on the test set...")
#         # metrics = evaluate_model(model, test_loader, config)

#         # Print evaluation metrics
#         print("Evaluation Metrics:")
#         print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
#         print(f"Precision: {metrics['precision'] * 100:.2f}%")
#         print(f"Recall: {metrics['recall'] * 100:.2f}%")
#         print(f"F1 Score: {metrics['f1'] * 100:.2f}%")
#     else:
#         print("Training a new model...")
#         model = TransformerSeq2SeqModel(
#             vocab_size=config["vocab_size"],
#             embed_size=config["embed_size"],
#             num_heads=config["num_heads"],
#             num_layers=config["num_layers"],
#             max_len=config["max_len"],
#         ).to(config["device"])

#         # Train the model
#         training_losses, validation_losses = train_model(model, train_loader, val_loader, config)

#         # Save the trained model
#         torch.save(model.state_dict(), model_path)
#         print(f"Model saved to {model_path}")

#         # Save Loss Curves
#         save_loss_plot(training_losses, validation_losses, save_path="results/loss_curve.png")

#         # Perform evaluation after training
#         # print("\nEvaluating on the test set...")
#         # metrics = evaluate_model(model, test_loader, config)

#         # print("Evaluation Metrics:")
#         # print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
#         # print(f"Precision: {metrics['precision'] * 100:.2f}%")
#         # print(f"Recall: {metrics['recall'] * 100:.2f}%")
#         # print(f"F1 Score: {metrics['f1'] * 100:.2f}%")


# def save_loss_plot(training_losses, validation_losses, save_path="results/loss_curve.png"):
#     """
#     Plot and save training and validation loss curves.

#     Args:
#         training_losses (list): Average training losses per epoch.
#         validation_losses (list): Average validation losses per epoch.
#         save_path (str): Path to save the plot.
#     """
#     if len(training_losses) != len(validation_losses):
#         raise ValueError("Mismatch between training and validation loss lengths.")

#     # Generate epoch indices
#     epochs = range(1, len(training_losses) + 1)

#     # Plot the losses
#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, training_losses, label="Training Loss", marker="o")
#     plt.plot(epochs, validation_losses, label="Validation Loss", marker="x")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.title("Training and Validation Loss Over Epochs")
#     plt.legend()
#     plt.grid()
#     plt.tight_layout()

#     # Save the plot
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path)
#     print(f"Loss curve saved at {save_path}")
#     plt.show()


# if __name__ == "__main__":
#     main()


# ----------- Execute for masked unique events 

import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from dataset import EventDataset
from model import TransformerSeq2SeqModel
from train import train_model
from utils import collate_fn
from masked_evaluation import evaluate_model_with_mask


def load_trained_model(model_path, vocab_size, embed_size, num_heads, num_layers, max_len, device):
    """
    Load a pre-trained model from disk.
    """
    model = TransformerSeq2SeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_heads=num_heads,
        num_layers=num_layers,
        max_len=max_len,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
    return model


def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Configurations
    model_path = "model_weights_epoch10.pth"
    evaluate_only = False  # Set to True to skip training and directly evaluate

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

    # Exclude specific events
    excluded_events = ["event_task_start", "event_task_end", "event_start_calc_local_context"]

    # Load a sample dataset to extract column names and build vocabulary
    sample_data = pd.read_csv(dataset_paths["train"])
    event_columns = [col for col in sample_data.columns if col != "cycle"]
    event_vocab = {event: idx for idx, event in enumerate(event_columns, start=1) if event not in excluded_events}
    event_vocab["<PAD>"] = 0
    vocab_size = len(event_vocab)

    print("Filtered Event Vocabulary:", event_vocab)
    print("Vocabulary Size:", vocab_size)

    # Model configuration
    config = {
        "batch_size": 32,
        "embed_size": 256,
        "num_heads": 2,
        "num_layers": 8,
        "vocab_size": vocab_size,
        "max_len": 77,
        "learning_rate": 1e-3,
        "epochs": 1,
        "device": device,
    }

    # Excluded event IDs
    excluded_ids = {event_vocab[event] for event in excluded_events if event in event_vocab}

    # Initialize datasets and dataloaders
    train_dataset = EventDataset(dataset_paths["train"], event_vocab, max_len=config["max_len"], excluded_events=excluded_events)
    val_dataset = EventDataset(dataset_paths["val"], event_vocab, max_len=config["max_len"], excluded_events=excluded_events)
    test_dataset = EventDataset(dataset_paths["test"], event_vocab, max_len=config["max_len"], excluded_events=excluded_events)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, excluded_ids=excluded_ids),  # Pass excluded_ids
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, excluded_ids=excluded_ids),  # Pass excluded_ids
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, excluded_ids=excluded_ids),  # Pass excluded_ids
    )

    if evaluate_only:
        print("Loading pre-trained model for evaluation...")
        model = load_trained_model(
            model_path=model_path,
            vocab_size=config["vocab_size"],
            embed_size=config["embed_size"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            max_len=config["max_len"],
            device=config["device"],
        )

        # Perform evaluation
        print("\nEvaluating on the test set with mask-based evaluation...")
        metrics, classification_rep = evaluate_model_with_mask(model, test_loader, config, excluded_ids)

        # Print evaluation metrics
        print("Evaluation Metrics (Masked):")
        print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"Precision: {metrics['precision'] * 100:.2f}%")
        print(f"Recall: {metrics['recall'] * 100:.2f}%")
        print(f"F1 Score: {metrics['f1'] * 100:.2f}%")
        print("\nClassification Report (Masked):")
        print(classification_rep)
    else:
        print("Training a new model...")
        model = TransformerSeq2SeqModel(
            vocab_size=config["vocab_size"],
            embed_size=config["embed_size"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            max_len=config["max_len"],
        ).to(config["device"])

        # Train the model
        training_losses, validation_losses = train_model(model, train_loader, val_loader, config)

        # Save the trained model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Save Loss Curves
        save_loss_plot(training_losses, validation_losses, save_path="results/loss_curve.png")

        # Perform evaluation after training
        print("\nEvaluating on the test set with mask-based evaluation...")
        metrics, classification_rep = evaluate_model_with_mask(model, test_loader, config, excluded_ids)

        print("Evaluation Metrics (Masked):")
        print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"Precision: {metrics['precision'] * 100:.2f}%")
        print(f"Recall: {metrics['recall'] * 100:.2f}%")
        print(f"F1 Score: {metrics['f1'] * 100:.2f}%")
        print("\nClassification Report (Masked):")
        print(classification_rep)


def save_loss_plot(training_losses, validation_losses, save_path="results/loss_curve.png"):
    """
    Plot and save training and validation loss curves.
    """
    if len(training_losses) != len(validation_losses):
        raise ValueError("Mismatch between training and validation loss lengths.")

    # Generate epoch indices
    epochs = range(1, len(training_losses) + 1)

    # Plot the losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_losses, label="Training Loss", marker="o")
    plt.plot(epochs, validation_losses, label="Validation Loss", marker="x")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Loss curve saved at {save_path}")
    plt.show()


if __name__ == "__main__":
    main()