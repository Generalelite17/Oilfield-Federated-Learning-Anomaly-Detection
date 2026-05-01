import json
import flwr as fl
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import NUM_CLIENTS, NUM_ROUNDS, RESULTS_DIR
from data import load_dataset, create_client_loaders, create_test_loader
from flower_client import FlowerClient
from model import FederatedNN, evaluate


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {key: torch.tensor(value) for key, value in params_dict}
    model.load_state_dict(state_dict, strict=True)


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def main():
    print("[INFO] Federated NN benchmark simulation started")

    train_dataset, test_dataset, input_dim, dataset_info = load_dataset()
    client_loaders = create_client_loaders(train_dataset)
    test_loader = create_test_loader(test_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = FederatedNN(input_dim=input_dim).to(device)

    round_results = []

    def client_fn(cid: str):
        return FlowerClient(
            cid=cid,
            trainloader=client_loaders[int(cid)],
            input_dim=input_dim,
        ).to_client()

    def evaluate_global_model(server_round, parameters, config):
        set_parameters(global_model, parameters)

        labels, predictions = evaluate(global_model, test_loader, device)

        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        result = {
            "round": int(server_round),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

        round_results.append(result)

        print(
            f"[ROUND {server_round}] "
            f"Accuracy={accuracy:.4f}, "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}"
        )

        return float(accuracy), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(global_model))

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_global_model,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    final_results = {
        "model": "Federated Neural Network",
        "aggregation": "FedAvg",
        "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS,
        "dataset_info": dataset_info,
        "round_results": round_results,
    }

    results_path = RESULTS_DIR / "federated_nn_results.json"

    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"[SUCCESS] Results saved to: {results_path.resolve()}")


if __name__ == "__main__":
    main()