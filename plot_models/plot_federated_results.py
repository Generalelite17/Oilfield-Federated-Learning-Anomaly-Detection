import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
with open("results/federated_nn_results.json", "r") as f:
    data = json.load(f)

rounds = [r["round"] for r in data["round_results"]]
accuracy = [r["accuracy"] for r in data["round_results"]]
f1 = [r["f1_score"] for r in data["round_results"]]

plt.figure(figsize=(8, 5))
plt.plot(rounds, accuracy, marker="o", label="Accuracy")
plt.plot(rounds, f1, marker="o", label="F1-score")

plt.title("Experiment 1: Federated Learning Convergence")
plt.xlabel("Training Rounds")
plt.xticks(rounds)
plt.ylabel("Performance")
plt.legend()
plt.grid()

plt.savefig("plot_outputs/federated_convergence.png")
plt.show()

Path("plot_outputs").mkdir(exist_ok=True)

models = ["Centralized NN", "FedAvg NN"]
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

centralized = [0.9990, 0.9365, 0.9646, 0.9503]
fedavg = [0.9990, 0.9534, 0.9482, 0.9508]

data = np.array([centralized, fedavg])

x = np.arange(len(models))
width = 0.18

plt.figure(figsize=(10, 6))

for i, metric in enumerate(metrics):
    plt.bar(
        x + (i - 1.5) * width,
        data[:, i],
        width,
        label=metric
    )

plt.title("Experiment 1: Centralized vs FedAvg Model Performance")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(x, models)

# More detailed y-axis
plt.ylim(0.85, 1.02)
plt.yticks(np.arange(0.85, 1.025, 0.025))

# Move legend outside graph
plt.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0
)

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("plot_outputs/fig_exp1_centralized_vs_fedavg.png", dpi=300, bbox_inches="tight")
plt.show()