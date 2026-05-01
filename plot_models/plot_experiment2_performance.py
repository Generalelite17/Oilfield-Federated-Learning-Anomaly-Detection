from pathlib import Path
import matplotlib.pyplot as plt

Path("plot_outputs").mkdir(exist_ok=True)

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
values = [0.9993, 1.0000, 0.9986, 0.9993]

plt.figure(figsize=(7, 5))
plt.bar(metrics, values)

plt.title("Experiment 2: FedAvg Performance on Modbus vs DDoS")
plt.ylabel("Score")
plt.ylim(0.95, 1.01)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plot_outputs/experiment2_fedavg_performance.png", dpi=300, bbox_inches="tight")
plt.show()