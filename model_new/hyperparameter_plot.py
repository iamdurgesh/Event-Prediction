import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to the directory containing experiment result files
results_dir = "results"  # Update this if your result files are in another directory

# Step 1: Extract Results
files = [f for f in os.listdir(results_dir) if f.endswith("_results.txt")]

# Create a DataFrame to store experiment results
experiment_data = []

for file in files:
    with open(os.path.join(results_dir, file), "r") as f:
        lines = f.readlines()
        experiment = {}
        experiment["Experiment"] = file.replace("_results.txt", "")
        for line in lines:
            line = line.strip()  # Remove leading/trailing spaces
            key_value = line.split(":")  # Split the line into key and value parts
            if len(key_value) == 2:  # Ensure the line is in "key: value" format
                key = key_value[0].strip().lower()
                value = key_value[1].strip().replace("{", "").replace("}", "").replace("'", "")

                if key == "embed_size":
                    experiment["Embedding Size"] = int(value)
                elif key == "num_heads":
                    experiment["Number of Heads"] = int(value)
                elif key == "num_layers":
                    experiment["Number of Layers"] = int(value)
                elif key == "learning_rate":
                    experiment["Learning Rate"] = float(value)
                elif key == "final training loss":
                    experiment["Training Loss"] = float(value)
                elif key == "final validation loss":
                    experiment["Validation Loss"] = float(value)
        experiment_data.append(experiment)

# Convert to DataFrame
results_df = pd.DataFrame(experiment_data)

# Extract numerical experiment index from the "Experiment" column
results_df["Experiment Index"] = results_df["Experiment"].str.extract(r"(\d+)").astype(int)

# Sort the DataFrame by the experiment index
results_df = results_df.sort_values("Experiment Index").reset_index(drop=True)

# Save to CSV for reference
results_df.to_csv("experiment_results_summary.csv", index=False)
print("Experiment results saved to 'experiment_results_summary.csv'")
print("\nExtracted Results:")
print(results_df)

# Step 2: Plot Results
# Plot Training vs Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(results_df["Experiment"], results_df["Training Loss"], marker='o', label="Training Loss")
plt.plot(results_df["Experiment"], results_df["Validation Loss"], marker='x', label="Validation Loss")
plt.xlabel("Experiment")
plt.ylabel("Loss")
plt.title("Training and Validation Loss for Experiments")
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("training_vs_validation_loss.png")
print("Training vs Validation Loss plot saved to 'training_vs_validation_loss.png'")
plt.show()

# Scatter Plot: Validation Loss vs Hyperparameters
plt.figure(figsize=(10, 6))
plt.scatter(results_df["Embedding Size"], results_df["Validation Loss"], label="Embedding Size", color='blue', alpha=0.7)
plt.scatter(results_df["Number of Heads"], results_df["Validation Loss"], label="Number of Heads", color='green', alpha=0.7)
plt.scatter(results_df["Number of Layers"], results_df["Validation Loss"], label="Number of Layers", color='red', alpha=0.7)
plt.xlabel("Hyperparameters")
plt.ylabel("Validation Loss")
plt.title("Validation Loss vs Hyperparameters")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("validation_loss_vs_hyperparameters.png")
print("Validation Loss vs Hyperparameters plot saved to 'validation_loss_vs_hyperparameters.png'")
plt.show()
