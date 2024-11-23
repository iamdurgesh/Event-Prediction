import pandas as pd

# Define the hyperparameter table
data = {
    "Experiment": [1, 2, 3, 4],
    "Embedding Size": [128, 256, 128, 256],
    "Number of Heads": [4, 8, 4, 8],
    "Number of Layers": [2, 4, 6, 2],
    "Learning Rate": [0.0001, 0.0005, 0.001, 0.0001],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the table to a CSV file
file_path = "hyperparameter_tuning_table.csv"
df.to_csv(file_path, index=False)

# Display the table for user
print(df)

