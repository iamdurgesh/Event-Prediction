import pandas as pd
import matplotlib.pyplot as plt

# Original data
data = {
    "Parameter": [
        "Learning Rate", "Batch Size", "Embedding Dimension", 
        "Sequence Length", "Number of Layers", "Number of Heads", 
        "Dropout Rate", "Weight Decay", "Scheduler"
    ],
    "Value Used in Project": [
        "0.001", "32", "128", 
        "77", "2", "4", 
        "0.1", "0.01", "None"
    ],
    "Suggested Adjustments": [
        "Use StepLR (reduce by 0.1 every 10 epochs) or Cosine Annealing.",
        "Increase to 64 if memory allows for faster training.",
        "Increase to 256 for larger datasets or more complex patterns.",
        "Fixed based on the dataset (rows per cycle).",
        "Increase to 4 for more complex event dependencies.",
        "Increase to 8 for better attention on larger datasets.",
        "Adjust between 0.1–0.3 depending on overfitting.",
        "Increase to 0.05 for better regularization if needed.",
        "Use StepLR or Cosine Annealing for dynamic learning rate."
    ]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Dropping the third column ("Suggested Adjustments") and the last row
two_column_df = df.drop(columns=["Suggested Adjustments"]).iloc[:-1]

# Displaying the modified DataFrame as a visual table with a bold header row
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

table = ax.table(
    cellText=two_column_df.values,
    colLabels=two_column_df.columns,
    cellLoc='center',
    loc='center'
)

# Set font size for better readability
table.auto_set_font_size(False)
table.set_fontsize(10)

# Make the header row bold
for (row, col), cell in table.get_celld().items():
    if row == 0:  # Header row
        cell.set_text_props(weight='bold')

table.auto_set_column_width(col=list(range(len(two_column_df.columns))))

plt.show()
