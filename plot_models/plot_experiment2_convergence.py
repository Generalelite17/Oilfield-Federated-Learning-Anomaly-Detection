import json
from pathlib import Path
import matplotlib.pyplot as plt

Path("plot_outputs").mkdir(exist_ok=True)

# Load FedAvg
with open("federated_learning/results/experiment2_fedavg_results.json", "r") as f:
    data = json.load(f)

# Load Centralized
with open("federated_learning/results/experiment2_centralized_results.json", "r") as f:
    cent_data = json.load(f)

valid_results = [
    r for r in data["round_results"]
    if r["accuracy"] is not None and r["f1"] is not None
]

rounds = [r["round"] for r in valid_results]
accuracy = [r["accuracy"] for r in valid_results]
f1 = [r["f1"] for r in valid_results]

# Extract centralized metrics
centralized_acc = cent_data["accuracy"]
centralized_f1 = cent_data["f1"]

plt.figure(figsize=(8, 5))

# FedAvg lines
plt.plot(rounds, accuracy, marker="o", label="FedAvg Accuracy")
plt.plot(rounds, f1, marker="o", label="FedAvg F1-score")

# Centralized (horizontal reference)
plt.axhline(y=centralized_acc, linestyle="--", label="Centralized Accuracy")
plt.axhline(y=centralized_f1, linestyle="--", label="Centralized F1-score")

plt.title("Experiment 2: FedAvg vs Centralized (Multi-Class EDGE Dataset)")
plt.xlabel("Communication Round")
plt.ylabel("Score")
plt.xticks(rounds)
plt.ylim(0.78, 1.00)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plot_outputs/experiment2_combined_convergence.png", dpi=300, bbox_inches="tight")
plt.show()