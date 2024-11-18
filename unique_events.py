import pandas as pd

# Load dataset
df = pd.read_csv("data/event_only_dataset1.csv")

# Extract unique event types
unique_events = set()
for col in df.columns:
    if col.startswith("event_"):  # Adjust based on your dataset's column names
        unique_events.update(df[col].unique())

# Add special tokens
special_tokens = {"<PAD>", "<MASK>"}
vocab_size = len(unique_events) + len(special_tokens)

print(f"Vocabulary Size: {vocab_size}")