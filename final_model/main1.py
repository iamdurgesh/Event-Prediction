# # from train1 import train_model
# # from dataset1 import EventDataset
# # from model1 import TransformerModel
# # from utils1 import collate_fn
# # from test1 import test_model
# # from evaluate import evaluate_model
# # import torch
# # from torch.utils.data import DataLoader
# # import matplotlib.pyplot as plt

# # def main():
# #     # Configuration
# #     config = {
# #         "batch_size": 64,
# #         "embed_size": 128,
# #         "num_heads": 4,
# #         "num_layers": 2,
# #         "vocab_size": 32,  # Adjust based on your dataset
# #         "max_len": 77,
# #         "learning_rate": 0.001,
# #         "epochs": 10,
# #         "device": "mps" if torch.backends.mps.is_available() else "cpu",
# #     }

# #     # # Load datasets
# #     # train_dataset = EventDataset("final_model/train.csv")
# #     # val_dataset = EventDataset("final_model/val.csv")
# #     # test_dataset = EventDataset("final_model/test.csv")

# #     # train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
# #     # val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
# #     # test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# #         # Load datasets
# #     train_dataset = EventDataset("final_model/train.csv")
# #     val_dataset = EventDataset("final_model/val.csv")
# #     test_dataset = EventDataset("final_model/test.csv")

# #     # Use collate_fn in DataLoader
# #     train_loader = DataLoader(
# #         train_dataset,
# #         batch_size=config["batch_size"],
# #         shuffle=True,
# #         collate_fn=collate_fn
# #     )
# #     val_loader = DataLoader(
# #         val_dataset,
# #         batch_size=config["batch_size"],
# #         shuffle=False,
# #         collate_fn=collate_fn
# #     )
# #     test_loader = DataLoader(
# #         test_dataset,
# #         batch_size=config["batch_size"],
# #         shuffle=False,
# #         collate_fn=collate_fn
# #     )

# #     # Initialize Model
# #     model = TransformerModel(
# #         vocab_size=config["vocab_size"],
# #         embed_size=config["embed_size"],
# #         num_heads=config["num_heads"],
# #         num_layers=config["num_layers"],
# #         max_len=config["max_len"]
# #     ).to(config["device"])

# #     # Train Model
# #     training_losses, validation_losses = train_model(model, train_loader, val_loader, config)

# #     # Evaluate on Test Set
# #     test_model(model, test_loader, config)

# #         # Visualize losses
# #     epochs = range(1, config["epochs"] + 1)
# #     plt.figure(figsize=(8, 6))
# #     plt.plot(epochs, training_losses, label="Training Loss")
# #     plt.plot(epochs, validation_losses, label="Validation Loss")
# #     plt.xlabel("Epochs")
# #     plt.ylabel("Loss")
# #     plt.title("Training and Validation Loss Over Epochs")
# #     plt.legend()
# #     plt.show()

# # if __name__ == "__main__":
# #     main()



# from train1 import train_model
# from dataset1 import EventDataset, preprocess_dataset  # Ensure preprocess_dataset is available
# from model1 import TransformerModel
# from utils1 import collate_fn
# from test1 import test_model  
# from evaluate import evaluate_model
# import torch
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt


# def main():
#     # Configuration
#     config = {
#         "batch_size": 64,
#         "embed_size": 128,
#         "num_heads": 4,
#         "num_layers": 2,
#         "vocab_size": 32,  # Adjust based on your dataset
#         "max_len": 77,
#         "learning_rate": 0.001,
#         "epochs": 10,
#         "device": "mps" if torch.backends.mps.is_available() else "cpu",
#     }

#     # Load datasets
#     train_dataset = EventDataset("final_model/train.csv")
#     val_dataset = EventDataset("final_model/val.csv")
#     test_dataset = EventDataset("final_model/test.csv")

#     # Use collate_fn in DataLoader
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config["batch_size"],
#         shuffle=True,
#         collate_fn=collate_fn
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config["batch_size"],
#         shuffle=False,
#         collate_fn=collate_fn
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=config["batch_size"],
#         shuffle=False,
#         collate_fn=collate_fn
#     )

#     # Initialize Model
#     model = TransformerModel(
#         vocab_size=config["vocab_size"],
#         embed_size=config["embed_size"],
#         num_heads=config["num_heads"],
#         num_layers=config["num_layers"],
#         max_len=config["max_len"]
#     ).to(config["device"])

#     # Train Model
#     training_losses, validation_losses = train_model(model, train_loader, val_loader, config)

#     # Evaluate on Test Set Before Excluding Duplicated Events
#     print("\n=== Evaluation Before Excluding Duplicated Events ===")
#     test_model(model, test_loader, config)  # Ensure this line is included for initial evaluation
#     evaluate_model(model, test_loader, config)

#     # Visualize losses
#     epochs = range(1, config["epochs"] + 1)
#     plt.figure(figsize=(8, 6))
#     plt.plot(epochs, training_losses, label="Training Loss")
#     plt.plot(epochs, validation_losses, label="Validation Loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.title("Training and Validation Loss Over Epochs")
#     plt.legend()
#     plt.show()

#     # Exclude duplicated events and preprocess datasets
#     duplicated_events = ["event_task_start", "event_task_end", "event_start_calc_local_context"]
#     train_dataset_excluded = preprocess_dataset("final_model/train.csv", duplicated_events)
#     val_dataset_excluded = preprocess_dataset("final_model/val.csv", duplicated_events)
#     test_dataset_excluded = preprocess_dataset("final_model/test.csv", duplicated_events)

#     train_loader_excluded = DataLoader(
#         train_dataset_excluded,
#         batch_size=config["batch_size"],
#         shuffle=True,
#         collate_fn=collate_fn
#     )
#     val_loader_excluded = DataLoader(
#         val_dataset_excluded,
#         batch_size=config["batch_size"],
#         shuffle=False,
#         collate_fn=collate_fn
#     )
#     test_loader_excluded = DataLoader(
#         test_dataset_excluded,
#         batch_size=config["batch_size"],
#         shuffle=False,
#         collate_fn=collate_fn
#     )

#     # Adjust vocab size after excluding events
#     config["vocab_size"] -= len(duplicated_events)

#     # Retrain model on updated dataset
#     print("\n=== Retraining Model After Excluding Duplicated Events ===")
#     model = TransformerModel(
#         vocab_size=config["vocab_size"],
#         embed_size=config["embed_size"],
#         num_heads=config["num_heads"],
#         num_layers=config["num_layers"],
#         max_len=config["max_len"]
#     ).to(config["device"])

#     training_losses_excluded, validation_losses_excluded = train_model(
#         model, train_loader_excluded, val_loader_excluded, config
#     )

#     # Evaluate on Test Set After Excluding Duplicated Events
#     print("\n=== Evaluation After Excluding Duplicated Events ===")
#     test_model(model, test_loader, config)  # Ensure this line is included for initial evaluation

#     evaluate_model(model, test_loader_excluded, config)

#     # Visualize new losses
#     epochs_excluded = range(1, config["epochs"] + 1)
#     plt.figure(figsize=(8, 6))
#     plt.plot(epochs_excluded, training_losses_excluded, label="Training Loss (Excluded Events)")
#     plt.plot(epochs_excluded, validation_losses_excluded, label="Validation Loss (Excluded Events)")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.title("Training and Validation Loss (After Excluding Events)")
#     plt.legend()
#     plt.show()


# if __name__ == "__main__":
#     main()



################################
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import os
import torch
from train1 import train_model
from dataset1 import EventDataset
from model1 import TransformerModel
from utils1 import collate_fn
from test1 import test_model
from evaluate import evaluate_model
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # Set device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Event vocabulary mapping
    sample_data_path = "final_model/train.csv"
    if not os.path.exists(sample_data_path):
        raise FileNotFoundError(f"Dataset file {sample_data_path} not found.")

    # Load a sample dataset to extract column names
    sample_data = pd.read_csv(sample_data_path)

    # Dynamically create event vocabulary
    event_vocab = {event: idx for idx, event in enumerate(sample_data.columns) if event != "cycle"}
    vocab_size = len(event_vocab)

    print("Event Vocabulary:", event_vocab)
    print("Vocabulary Size:", vocab_size)

    # Configuration
    config = {
        "batch_size": 64,
        "embed_size": 128,
        "num_heads": 4,
        "num_layers": 2,
        "vocab_size": vocab_size,
        "max_len": 77,
        "learning_rate": 1e-5,
        "epochs": 2,  # Set back to 5 for full training
        "device": device,
        "debug": False,  # Set to True for detailed debugging output
    }

    # Load datasets
    dataset_paths = {
        "train": "final_model/train.csv",
        "val": "final_model/val.csv",
        "test": "final_model/test.csv",
    }
    for name, path in dataset_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name.capitalize()} dataset file {path} not found.")

    train_dataset = EventDataset(dataset_paths["train"], event_vocab)
    val_dataset = EventDataset(dataset_paths["val"], event_vocab)
    test_dataset = EventDataset(dataset_paths["test"], event_vocab)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn
    )

    # Initialize Model
    model = TransformerModel(
        vocab_size=config["vocab_size"],
        embed_size=config["embed_size"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_len=config["max_len"]
    ).to(config["device"])
    print(model) 

    # Train Model
    training_losses, validation_losses = train_model(model, train_loader, val_loader, config)

    # Save Loss Curves
    loss_curve_path = "results/loss_curve.png"
    os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)
    plot_losses(training_losses, validation_losses, save_path=loss_curve_path)

    # Evaluate on Test Set
    test_model(model, test_loader, config)


def plot_losses(training_losses, validation_losses, save_path="results/loss_curve.png"):
    """
    Plot and save training and validation loss curves.

    Args:
        training_losses (list): Average training losses per epoch.
        validation_losses (list): Average validation losses per epoch.
        save_path (str): Path to save the plot.
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
    plt.savefig(save_path)
    print(f"Loss curve saved at {save_path}")
    plt.show()


if __name__ == "__main__":
    main()

