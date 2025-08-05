import matplotlib.pyplot as plt
import os


def plot_two_files(train_file, eval_file):
    """
    Reads two text files and plots:
      - First column on the x-axis
      - Fifth column on the y-axis
    Labels the curves 'Train' and 'Evaluation' and displays a legend.
    """
    # Read Train data
    x_train, y_train = [], []
    with open(train_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                x_train.append(float(parts[0]))
                y_train.append(float(parts[4]))

    # Read Evaluation data
    x_eval, y_eval = [], []
    with open(eval_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                x_eval.append(float(parts[0]))
                y_eval.append(float(parts[4]))

    # Plot both on the same axes
    plt.figure()
    plt.plot(x_eval, y_eval, label="Evaluation")
    plt.plot(x_train, y_train, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Skeleton Loss")
    plt.title("Train vs Evaluation - mmMesh")
    plt.legend()
    plt.tight_layout()
    plt.show()


script_dir = os.path.dirname(os.path.abspath(__file__))
filename = "20250617_0203.txt"
train_file = os.path.join(script_dir, "loss", filename)
eval_file = os.path.join(script_dir, "eval", filename)
plot_two_files(train_file, eval_file)
