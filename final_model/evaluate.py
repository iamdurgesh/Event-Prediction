from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

# def evaluate_model(model, data_loader, config):
#     model.eval()
#     all_targets = []
#     all_predictions = []
#     all_masks = []
#     event_names = [f"event_{i}" for i in range(config["vocab_size"])]  # Adjust for your event names

#     with torch.no_grad():
#         for inputs, targets, mask in data_loader:
#             inputs, targets, mask = inputs.to(config["device"]), targets.to(config["device"]), mask.to(config["device"])

#             # Forward pass
#             outputs = model(inputs, mask)
#             predictions = (torch.sigmoid(outputs) > 0.5).float()

#             # Masked targets and predictions
#             masked_targets = (targets * mask.unsqueeze(-1)).view(-1, config["vocab_size"])
#             masked_predictions = (predictions * mask.unsqueeze(-1)).view(-1, config["vocab_size"])

#             # Append to all_targets and all_predictions
#             all_targets.append(masked_targets.cpu())
#             all_predictions.append(masked_predictions.cpu())
#             all_masks.append(mask.cpu())

#     # Concatenate results
#     all_targets = torch.cat(all_targets, dim=0)
#     all_predictions = torch.cat(all_predictions, dim=0)

#     # Convert to numpy arrays for sklearn
#     targets_np = all_targets.numpy()
#     predictions_np = all_predictions.numpy()

#     # Calculate metrics
#     accuracy = accuracy_score(targets_np.flatten(), predictions_np.flatten())
#     precision = precision_score(targets_np.flatten(), predictions_np.flatten(), average="macro", zero_division=0)
#     recall = recall_score(targets_np.flatten(), predictions_np.flatten(), average="macro", zero_division=0)
#     f1 = f1_score(targets_np.flatten(), predictions_np.flatten(), average="macro", zero_division=0)

#     # Event-wise metrics
#     event_metrics = {
#         event_names[i]: {
#             "precision": precision_score(targets_np[:, i], predictions_np[:, i], zero_division=0),
#             "recall": recall_score(targets_np[:, i], predictions_np[:, i], zero_division=0),
#             "f1": f1_score(targets_np[:, i], predictions_np[:, i], zero_division=0),
#         }
#         for i in range(len(event_names))
#     }

#     # Print results
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     print("\nEvent-wise Metrics:")
#     for event, metrics in event_metrics.items():
#         print(f"{event}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

#     return accuracy, precision, recall, f1, event_metrics




#---------------------- for sequence prediction

def evaluate_model(model, data_loader, config):
    """
    Evaluate the model's performance for sequence prediction.

    Args:
        model: Trained Transformer model.
        data_loader: DataLoader for evaluation dataset.
        config: Configuration dictionary with training parameters.

    Returns:
        dict: Metrics including sequence accuracy, event accuracy, precision, recall, F1 score, and event-wise metrics.
    """
    model.eval()
    all_targets = []
    all_predictions = []
    
    # Use specific event names if provided, otherwise generate generic names
    event_names = config.get("event_names", [f"event_{i}" for i in range(config["vocab_size"])])

    with torch.no_grad():
        for inputs, targets, mask in data_loader:
            # Move inputs and targets to the configured device
            inputs = inputs.to(config["device"])
            targets = targets.to(config["device"])
            mask = mask.to(config["device"])  # Ensure mask is on the same device

            # Forward pass
            outputs = model(inputs, src_key_padding_mask=~mask.bool())  # Shape: (batch_size, seq_len, vocab_size)
            predictions = torch.argmax(outputs, dim=-1)  # Predicted event indices

            # Collect predictions and targets for metrics
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0).numpy()  # Shape: (num_samples, seq_len)
    all_targets = torch.cat(all_targets, dim=0).numpy()  # Shape: (num_samples, seq_len)

    # Sequence-level accuracy: All events in a sequence must match
    sequence_accuracy = (all_predictions == all_targets).all(axis=1).mean() * 100

    # Flatten the arrays for event-level metrics
    flat_predictions = all_predictions.flatten()
    flat_targets = all_targets.flatten()

    # Overall metrics
    accuracy = accuracy_score(flat_targets, flat_predictions) * 100
    precision = precision_score(flat_targets, flat_predictions, average="macro", zero_division=0)
    recall = recall_score(flat_targets, flat_predictions, average="macro", zero_division=0)
    f1 = f1_score(flat_targets, flat_predictions, average="macro", zero_division=0)

    # Event-wise metrics
    event_metrics = {}
    for i, event_name in enumerate(event_names):
        event_metrics[event_name] = {
            "precision": precision_score(flat_targets == i, flat_predictions == i, zero_division=0),
            "recall": recall_score(flat_targets == i, flat_predictions == i, zero_division=0),
            "f1": f1_score(flat_targets == i, flat_predictions == i, zero_division=0),
        }

    # Print metrics if verbose
    if config.get("verbose", True):
        print("\nEvaluation Metrics:")
        print(f"Sequence Accuracy: {sequence_accuracy:.2f}%")
        print(f"Event Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}\n")

        print("Event-wise Metrics:")
        for event, metrics in event_metrics.items():
            print(f"{event}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

    # Return all metrics as a dictionary
    return {
        "sequence_accuracy": sequence_accuracy,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "event_metrics": event_metrics,
    }
