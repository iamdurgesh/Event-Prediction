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

# Save the datasets
train_data_split.to_csv("train.csv", index=False)
val_data_split.to_csv("val.csv", index=False)
test_data.to_csv("test.csv", index=False)

print(f"Train: {len(train_data_split)}, Validation: {len(val_data_split)}, Test: {len(test_data)}")
