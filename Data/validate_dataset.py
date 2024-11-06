import json
import pandas as pd
import pickle

def load_dataset(file_path, file_format="json"):
    """Load the dataset from the specified file format."""
    if file_format == "json":
        with open(file_path, 'r') as f:
            dataset = json.load(f)
    elif file_format == "csv":
        dataset = pd.read_csv(file_path)["global_context"].apply(lambda x: x.split(", ")).tolist()
    elif file_format == "pickle":
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        raise ValueError("Unsupported file format. Choose 'json', 'csv', or 'pickle'.")
    return dataset

def validate_dataset_structure(dataset, expected_cycle_count=None):
    """Validate the basic structure and content of the dataset."""
    print("Total cycles in dataset:", len(dataset))

    # Check cycle count consistency, if an expected count is given
    if expected_cycle_count is not None:
        assert len(dataset) == expected_cycle_count, (
            f"Expected {expected_cycle_count} cycles, but found {len(dataset)}."
        )

    # Inspect a few cycles to confirm event sequences
    for i, cycle in enumerate(dataset[:5]):  # Check first 5 cycles for a sample
        print(f"Cycle {i} - Event Count: {len(cycle)} - Events: {cycle}")
        assert isinstance(cycle, list), f"Cycle {i} is not a list."
        assert all(isinstance(event, str) for event in cycle), f"Cycle {i} contains non-string events."

def validate_non_empty_sequences(dataset):
    """Ensure each cycle has a non-empty global context."""
    empty_cycles = [i for i, cycle in enumerate(dataset) if not cycle]
    if empty_cycles:
        print(f"Warning: Found empty cycles at indices: {empty_cycles}")
    else:
        print("All cycles contain events.")

def validate_event_order(dataset):
    """Check that events within each cycle appear in the expected order."""
    # This is a placeholder for specific order checks if there's an expected sequence to follow
    # For instance, we might want to ensure certain events don’t follow others if there's a rule
    print("Event order validation is not configured with specific rules but can be customized.")

def main():
    # Load dataset (update file path and format as needed)
    file_path = 'data/final_global_context_dataset.json'
    file_format = "json"  # Change to "csv" or "pickle" if needed
    dataset = load_dataset(file_path, file_format=file_format)
    
    # Validate dataset structure
    expected_cycle_count = 10  # Update based on the expected number of cycles, if known
    validate_dataset_structure(dataset, expected_cycle_count=expected_cycle_count)
    
    # Check that all cycles have events
    validate_non_empty_sequences(dataset)
    
    # Optionally validate event order if specific rules apply
    validate_event_order(dataset)

if __name__ == "__main__":
    main()
