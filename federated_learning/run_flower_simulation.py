import json
import csv
from pathlib import Path

import flwr as fl

from data import load_and_prepare_data, split_clients
from flower_client import FlowerClient

NUM_CLIENTS = 10
NUM_ROUNDS = 10

RESULTS_DIR = Path("federated_learning/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

X, y = load_and_prepare_data()
clients = split_clients(X, y, num_clients=NUM_CLIENTS)
INPUT_SIZE = X.shape[1]


def client_fn(cid: str):
    trainloader, testloader = clients[int(cid)]
    return FlowerClient(cid, trainloader, testloader, INPUT_SIZE).to_client()


def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)

    aggregated = {}
    for metric_name in metrics[0][1].keys():
        aggregated[metric_name] = sum(
            num_examples * metric[metric_name]
            for num_examples, metric in metrics
        ) / total_examples

    return aggregated


def main():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    results = []

    metrics = history.metrics_distributed

    for round_num, accuracy in metrics.get("accuracy", []):
        row = {
            "round": round_num,
            "accuracy": accuracy,
            "precision": None,
            "recall": None,
            "f1": None,
        }

        for metric_name in ["precision", "recall", "f1"]:
            for r, value in metrics.get(metric_name, []):
                if r == round_num:
                    row[metric_name] = value

        results.append(row)

    json_path = RESULTS_DIR / "experiment2_fedavg_results.json"
    csv_path = RESULTS_DIR / "experiment2_fedavg_results.csv"

    with open(json_path, "w") as f:
        json.dump(
            {
                "experiment": "Experiment 2",
                "dataset": "EDGE Modbus vs DDoS",
                "num_clients": NUM_CLIENTS,
                "num_rounds": NUM_ROUNDS,
                "round_results": results,
            },
            f,
            indent=4,
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["round", "accuracy", "precision", "recall", "f1"],
        )
        writer.writeheader()
        writer.writerows(results)

    final = results[-1]

    print("\n========== EXPERIMENT 2 FEDAVG RESULTS ==========")
    print("Dataset : EDGE Modbus vs DDoS")
    print(f"Clients : {NUM_CLIENTS}")
    print(f"Rounds  : {NUM_ROUNDS}")
    print("\nFinal Round Results:")
    print(f"Accuracy : {final['accuracy']:.4f}")
    print(f"Precision: {final['precision']:.4f}")
    print(f"Recall   : {final['recall']:.4f}")
    print(f"F1-score : {final['f1']:.4f}")

    print("\nResults saved to:")
    print(json_path.resolve())
    print(csv_path.resolve())


if __name__ == "__main__":
    main()