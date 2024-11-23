import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def plot_ffn_smaller_nodes():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define coordinates for layers
    input_layer = [4, 5, 6]  # Input layer adjusted vertically
    hidden_layer = [3, 4, 5, 6, 7]  # Hidden layer adjusted
    output_layer = [4.5, 5.5, 6.5]  # Output layer adjusted

    # Plot nodes with smaller size
    ax.scatter([1]*3, input_layer, s=300, label="Input Layer (x)", color="skyblue")
    ax.scatter([2]*5, hidden_layer, s=300, label="Hidden Layer (ReLU + W1)", color="lightgreen")
    ax.scatter([3]*3, output_layer, s=300, label="Output Layer (W2)", color="salmon")

    # Add arrows (connections)
    for x in input_layer:
        for y in hidden_layer:
            ax.add_patch(FancyArrowPatch((1, x), (2, y), arrowstyle="->", mutation_scale=10, color='gray', alpha=0.6))

    for x in hidden_layer:
        for y in output_layer:
            ax.add_patch(FancyArrowPatch((2, x), (3, y), arrowstyle="->", mutation_scale=10, color='gray', alpha=0.6))

    # Add labels
    ax.text(1, 6.5, "Input (x)", ha="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    ax.text(2, 7.5, "ReLU\n+ Linear\n(W1)", ha="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    ax.text(3, 7.0, "Output", ha="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))

    # Styling
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(2.5, 8)
    ax.axis("off")
    plt.legend(loc="lower right")
    plt.title("Feed-Forward Neural Network (FFN) Architecture", fontsize=14, pad=20)
    plt.show()

plot_ffn_smaller_nodes()
