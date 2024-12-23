from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import torch

def evaluate_model_with_mask(model, data_loader, config, event_vocab, excluded_events=None):
    """
    Evaluate the model on a dataset while excluding specific events.

    Args:
        model (nn.Module): The trained model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        config (dict): Configuration dictionary.
        event_vocab (dict): Mapping of events to indices.
        excluded_events (list or set): Events to exclude during evaluation.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1 score.
        str: Classification report as a formatted string.
    """

    model.eval()
    device = config["device"]

    # Handle excluded events, including the <EXCLUDED> token
    excluded_event_ids = [event_vocab[event] for event in excluded_events if event in event_vocab]
    
    # Check and add the <EXCLUDED> token to the excluded IDs if it exists in the vocabulary
    if "<EXCLUDED>" in event_vocab:
        excluded_event_ids.append(event_vocab["<EXCLUDED>"])
    else:
        print("Warning: <EXCLUDED> token is not in the event vocabulary. Skipping it.")

    # Initialize lists for storing predictions, targets, and masks
    all_predictions = []
    all_targets = []
    mask = []

    with torch.no_grad():
        for inputs, targets, _ in data_loader:  # Expecting three outputs due to collate_fn
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs, inputs)
            predicted_ids = outputs.argmax(dim=-1)

            all_predictions.extend(predicted_ids.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

            # Create a mask for excluded events
            batch_mask = [[1 if t not in excluded_event_ids else 0 for t in seq] for seq in targets.cpu().numpy()]
            mask.extend(batch_mask)

    # Flatten lists
    flat_predictions = [p for seq in all_predictions for p in seq]
    flat_targets = [t for seq in all_targets for t in seq]
    flat_mask = [m for seq in mask for m in seq]

    # Apply the mask
    filtered_predictions = [p for p, m in zip(flat_predictions, flat_mask) if m == 1]
    filtered_targets = [t for t, m in zip(flat_targets, flat_mask) if m == 1]

    # Calculate metrics
    accuracy = accuracy_score(filtered_targets, filtered_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(filtered_targets, filtered_predictions, average="weighted")

    classification_rep = classification_report(filtered_targets, filtered_predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }, classification_rep


