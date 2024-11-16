import matplotlib.pyplot as plt
#  Define the hyperparameters and their values again for plotting
hyperparameters = {
    "Learning Rate": 0.001,
    "Batch Size": 32,
    "Embedding Dimension": 128,
    "Number of Layers": 2,
    "Number of Heads": 4,
    "Dropout Rate": 0.1,
    "Weight Decay": 0.01,
    "Padding Token ID": 0,
    "Sequence Length (Max)": 77,
    "Vocabulary Size": 100,
    "Scheduler": "StepLR (step=10, gamma=0.1)"
}

# Separate numerical and non-numerical parameters
numerical_params = {k: v for k, v in hyperparameters.items() if isinstance(v, (int, float))}
categorical_params = {k: v for k, v in hyperparameters.items() if isinstance(v, str)}

# Plot the numerical parameters as a horizontal bar chart
plt.figure(figsize=(12, 6))
plt.barh(list(numerical_params.keys()), list(numerical_params.values()), color='skyblue')
plt.xlabel("Value")
plt.ylabel("Hyperparameter")
plt.title("Hyperparameters Visualization")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# Display a table of hyperparameters using matplotlib
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')
table_data = [[k, v] for k, v in hyperparameters.items()]
ax.table(cellText=table_data, colLabels=["Hyperparameter", "Value"], loc='center', cellLoc='center')
plt.title("Hyperparameters Table")
plt.show()
