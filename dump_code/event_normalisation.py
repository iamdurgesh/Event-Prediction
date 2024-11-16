import pandas as pd
import matplotlib.pyplot as plt

# Load the event-only dataset
file_path = 'data/event_only_dataset.csv'  # Replace with your actual file path
event_data = pd.read_csv(file_path)

# Count occurrences of 1 in each event column (indicating an occurrence)
event_columns = [col for col in event_data.columns if col.startswith('event_')]
event_frequencies = event_data[event_columns].sum()

# Normalize the frequencies to a range of 0 to 1
normalized_frequencies = (event_frequencies - event_frequencies.min()) / (event_frequencies.max() - event_frequencies.min())

# Print the normalized counts for each event
print("Normalized counts for each event:")
for event, count in normalized_frequencies.items():
    print(f"{event}: {count:.2f}")

# Plotting the normalized frequency of events as a bar chart
plt.figure(figsize=(12, 8))
normalized_frequencies.plot(kind='barh', color='skyblue')
plt.xlabel('Normalized Frequency (0 to 1)')
plt.ylabel('Event')
plt.title('Normalized Frequency of Each Event in the Dataset')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
