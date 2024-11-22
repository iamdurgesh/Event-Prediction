import torch
import json
from dataset1 import EventDataset
from utils import collate_fn
from evaluate import evaluate_model
from model1 import TransformerModel
from lstm_model import LSTMModel
from torch.utils.data import DataLoader

def test_model(model, test_loader, config, model_name):
    """
    Evaluate a trained model on the test dataset and save metrics.

    Args:
        model: Trained model (Transformer or LSTM).
        test_loader: DataLoader for the test dataset.
        config: Configuration dictionary.
        model_name: Name of the model ("Transformer" or "LSTM").

    Returns:
        None
    """
    print(f"Testing {model_name} model...")

    # Evaluate the model
    metrics = evaluate_model(model, test_loader, config)

    # Save the metrics
    results_path = f"results/test_results_{model_name.lower()}.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"{model_name} test metrics saved to {results_path}")

def load_model(model_type, config):
    """
    Load a trained model (Transformer or LSTM) from a saved checkpoint.

    Args:
        model_type: "Transformer" or "LSTM".
        config: Configuration dictionary.

    Returns:
        model: Loaded PyTorch model.
    """
    if model_type == "Transformer":
        model = TransformerModel(
            vocab_size=config["vocab_size"],
            embed_size=config["embed_size"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            max_len=config["max_len"]
        )
    elif model_type == "LSTM":
        model = LSTMModel(
            vocab_size=config["vocab_size"],
            embed_size=config["embed_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"]
        )
    else:
        raise ValueError("Invalid model type. Choose 'Transformer' or 'LSTM'.")
    
    model.load_state_dict(torch.load(f"results/{model_type.lower()}_checkpoint.pth"))
    return model

def main():
    # Set device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load test dataset
    test_dataset_path = "final_model/test.csv"
    event_vocab = {f"event_{i}": i for i in range(33)}  # Adjust as per vocab size
    test_dataset = EventDataset(test_dataset_path, event_vocab)
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
    )

    # Configurations
    config = {
        "vocab_size": len(event_vocab),
        "embed_size": 128,
        "num_heads": 4,
        "num_layers": 2,
        "hidden_size": 256,  # Only used for LSTM
        "max_len": 77,
        "device": device,
    }

    # Test Transformer model
    transformer_model = load_model("Transformer", config).to(device)
    test_model(transformer_model, test_loader, config, "Transformer")

    # Test LSTM model
    lstm_model = load_model("LSTM", config).to(device)
    test_model(lstm_model, test_loader, config, "LSTM")

if __name__ == "__main__":
    main()
