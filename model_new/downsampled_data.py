import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Load dataset
data = pd.read_csv("data/event_only_dataset.csv")

# Group by "cycle" to ensure unique cycles in each split
group_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(group_split.split(data, groups=data["cycle"]))

train_data = data.iloc[train_idx]
test_data = data.iloc[test_idx]

# Further split train_data into training and validation
group_split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)  # 20% of total for validation
train_idx, val_idx = next(group_split.split(train_data, groups=train_data["cycle"]))

# Apply splits before resetting index
train_data_split = train_data.iloc[train_idx]
val_data_split = train_data.iloc[val_idx]

# Reset index for all splits
train_data_split = train_data_split.reset_index(drop=True)
val_data_split = val_data_split.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

# Analyze Class Distribution in Training Data
class_counts = train_data_split.drop(columns=["cycle"]).sum(axis=0)
print("Class Distribution in Training Data:")
print(class_counts)

# Optional: Downsample Frequent Events to Balance the Dataset
max_samples_per_class = 500  # Define a threshold for downsampling
event_columns = [col for col in train_data_split.columns if col != "cycle"]  # Exclude the "cycle" column

# Reshape data to long format for easier manipulation
long_data = train_data_split.melt(id_vars=["cycle"], var_name="event", value_name="presence")
long_data = long_data[long_data["presence"] == 1]  # Only keep rows where event is present

# Downsample
balanced_data = long_data.groupby("event").apply(
    lambda x: x.sample(n=min(len(x), max_samples_per_class), random_state=42)
).reset_index(drop=True)

# Pivot back to wide format
balanced_data = balanced_data.pivot_table(index="cycle", columns="event", values="presence", fill_value=0).reset_index()

# Merge the downsampled data with the original cycle information
balanced_data = pd.merge(train_data_split[["cycle"]], balanced_data, on="cycle", how="right")

# Save the datasets
balanced_data.to_csv("train_balanced.csv", index=False)
val_data_split.to_csv("val.csv", index=False)
test_data.to_csv("test.csv", index=False)

print(f"Balanced Train: {len(balanced_data)}, Validation: {len(val_data_split)}, Test: {len(test_data)}")
