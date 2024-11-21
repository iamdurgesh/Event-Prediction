# from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
# import torch

# def evaluate_model(model, data_loader, config):
#     """
#     Evaluate the model on the given dataset.

#     Args:
#         model (nn.Module): Trained model.
#         data_loader (DataLoader): DataLoader for the evaluation dataset.
#         config (dict): Configuration dictionary.

#     Returns:
#         dict: Dictionary containing evaluation metrics (accuracy, precision, recall, F1 score).
#     """
#     model.eval()
#     device = config["device"]
#     all_predictions = []
#     all_targets = []

#     with torch.no_grad():
#         for inputs, targets, mask in data_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             mask = mask.to(device)

#             # Get model predictions
#             outputs = model(inputs, inputs, src_mask=None, tgt_mask=None, 
#                             src_key_padding_mask=(mask == 0), tgt_key_padding_mask=(mask == 0))
#             predicted_ids = outputs.argmax(dim=-1)  # Get the predicted class IDs

#             # Append predictions and targets for evaluation
#             all_predictions.extend(predicted_ids.cpu().numpy().tolist())
#             all_targets.extend(targets.cpu().numpy().tolist())

#     # Flatten lists to evaluate at token level
#     flat_predictions = [pred for seq in all_predictions for pred in seq]
#     flat_targets = [true for seq in all_targets for true in seq]

#     # Remove padding (assume <PAD> has ID 0)
#     flat_predictions = [p for p, t in zip(flat_predictions, flat_targets) if t != 0]
#     flat_targets = [t for t in flat_targets if t != 0]

#     # Calculate metrics
#     accuracy = accuracy_score(flat_targets, flat_predictions)
#     precision, recall, f1, _ = precision_recall_fscore_support(flat_targets, flat_predictions, average="weighted")

#     # Print classification report
#     print("\nClassification Report:")
#     print(classification_report(flat_targets, flat_predictions, zero_division=0))

#     return {
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1
#     }





# -------------------- evaluating after excluding duplicated events

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import torch

# def evaluate_model(model, data_loader, config):
#     """
#     Evaluate the model on the given dataset without excluding any event IDs.

#     Args:
#         model (nn.Module): Trained model.
#         data_loader (DataLoader): DataLoader for the evaluation dataset.
#         config (dict): Configuration dictionary.

#     Returns:
#         dict: Dictionary containing evaluation metrics (accuracy, precision, recall, F1 score).
#     """
#     model.eval()
#     device = config["device"]
#     all_predictions = []
#     all_targets = []

#     with torch.no_grad():
#         for inputs, targets, mask in data_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             mask = mask.to(device)

#             # Get model predictions
#             outputs = model(inputs, inputs, src_mask=None, tgt_mask=None, 
#                             src_key_padding_mask=(mask == 0), tgt_key_padding_mask=(mask == 0))
#             predicted_ids = outputs.argmax(dim=-1)  # Get the predicted class IDs

#             # Append predictions and targets for evaluation
#             all_predictions.extend(predicted_ids.cpu().numpy().tolist())
#             all_targets.extend(targets.cpu().numpy().tolist())

#     # Flatten lists to evaluate at token level
#     flat_predictions = [pred for seq in all_predictions for pred in seq]
#     flat_targets = [true for seq in all_targets for true in seq]

#     # Remove padding (assume <PAD> has ID 0)
#     flat_predictions = [p for p, t in zip(flat_predictions, flat_targets) if t != 0]
#     flat_targets = [t for t in flat_targets if t != 0]

#     # Calculate metrics
#     accuracy = accuracy_score(flat_targets, flat_predictions)
#     precision, recall, f1, _ = precision_recall_fscore_support(flat_targets, flat_predictions, average="weighted")

#     # Print classification report
#     print("\nClassification Report:")
#     print(classification_report(flat_targets, flat_predictions, zero_division=0))

#     return {
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1
#     }


def evaluate_model_with_mask(model, data_loader, config, excluded_ids):
    """
    Evaluate the model with a mask to exclude specific events.

    Args:
        model (nn.Module): Trained model.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        config (dict): Configuration dictionary.
        excluded_ids (set): Set of excluded event IDs.

    Returns:
        dict: Evaluation metrics.
        str: Classification report.
    """
    model.eval()
    device = config["device"]
    all_predictions = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for inputs, targets, mask in data_loader:
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)

            # Get model predictions
            outputs = model(inputs, inputs, src_mask=None, tgt_mask=None, 
                            src_key_padding_mask=(mask == 0), tgt_key_padding_mask=(mask == 0))
            predicted_ids = outputs.argmax(dim=-1)

            all_predictions.extend(predicted_ids.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            all_masks.extend(mask.cpu().numpy().tolist())

    # Flatten lists
    flat_predictions = [p for seq in all_predictions for p in seq]
    flat_targets = [t for seq in all_targets for t in seq]
    flat_mask = [m for seq in all_masks for m in seq]

    # Apply mask
    filtered_predictions = [p for p, m in zip(flat_predictions, flat_mask) if m > 0]
    filtered_targets = [t for t, m in zip(flat_targets, flat_mask) if m > 0]

    # Calculate metrics
    accuracy = accuracy_score(filtered_targets, filtered_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(filtered_targets, filtered_predictions, average="weighted")
    classification_rep = classification_report(filtered_targets, filtered_predictions, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }, classification_rep

