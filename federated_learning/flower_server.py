import flwr as fl
from config import NUM_CLIENTS, NUM_ROUNDS

def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)

    return {
        "accuracy": sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples,
        "precision": sum(num_examples * m["precision"] for num_examples, m in metrics) / total_examples,
        "recall": sum(num_examples * m["recall"] for num_examples, m in metrics) / total_examples,
        "f1": sum(num_examples * m["f1"] for num_examples, m in metrics) / total_examples,
    }


def start_server(num_rounds=5):
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_server(num_rounds=5)