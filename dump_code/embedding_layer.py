import matplotlib.pyplot as plt
import numpy as np

# Example data
sequence = [
    "event_task_start", "duplicate", "event_gnss", 
    "event_imu", "duplicate", "event_task_end"
]
predictions = [
    "event_task_start", "-", "event_gnss", 
    "event_imu", "-", "event_task_end"
]
mask = [1, 0, 1, 1, 0, 1]  # 1: meaningful, 0: duplicate (masked)

# Create a bar chart for visualization
x = np.arange(len(sequence))  # Positions for bars

plt.figure(figsize=(12, 6))

# Plot all events
plt.bar(x, [1] * len(sequence), color="skyblue", edgecolor="black", label="Meaningful Event")
plt.bar(x, [1 if m == 0 else 0 for m in mask], color="lightgray", label="Masked Duplicate")

# Add labels for the sequence
for i, event in enumerate(sequence):
    plt.text(i, 0.5, event, ha='center', va='center', fontsize=10, rotation=45)

plt.xticks(x, [f"Step {i+1}" for i in x], rotation=0)
plt.yticks([])
plt.title("Visualization of Output Sequence with Masked Duplicates", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()
