import pandas as pd

train_data = pd.read_csv("train.csv")
val_data = pd.read_csv("model_new/val.csv")
test_data = pd.read_csv("model_new/test.csv")

print(f"Training file rows: {len(train_data)}")
print(f"Validation file rows: {len(val_data)}")
print(f"Test file rows: {len(test_data)}")