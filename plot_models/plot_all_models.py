import matplotlib.pyplot as plt
import numpy as np

models = ["Isolation Forest", "Centralized NN", "FedAvg NN"]
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

# Isolation Forest does not produce these same supervised metrics,
# so we use 0 for unavailable metrics and show anomaly rate separately.
accuracy = [0.00, 0.9990, 0.9990]
precision = [0.00, 0.9365, 0.9534]
recall = [0.00, 0.9646, 0.9482]
f1 = [0.00, 0.9503, 0.9508]

x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(10, 6))

plt.bar(x - 1.5*width, accuracy, width, label="Accuracy")
plt.bar(x - 0.5*width, precision, width, label="Precision")
plt.bar(x + 0.5*width, recall, width, label="Recall")
plt.bar(x + 1.5*width, f1, width, label="F1-score")

plt.xticks(x, models)
plt.ylabel("Score")
plt.ylim(0, 1.10)
plt.title("Experiment 1: Comparison of Isolation Forest, Centralized NN, and FedAvg NN")
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.grid(axis="y", alpha=0.3)

plt.text(
    x[0],
    0.05,
    "Unsupervised baseline\nAnomaly rate: 1.0%\n(1,837 / 183,630 flows)",
    ha="center",
    fontsize=9
)

plt.tight_layout()
plt.savefig("plot_outputs/experiment1_all_models_comparison.png", dpi=300, bbox_inches="tight")
plt.show()