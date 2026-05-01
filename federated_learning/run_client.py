import sys
import flwr as fl

from data import load_and_prepare_data, split_clients
from flower_client import FlowerClient
from config import NUM_CLIENTS


def main():
    cid = int(sys.argv[1])  # client ID (0, 1, 2)

    X, y = load_and_prepare_data()
    input_size = X.shape[1]
    clients = split_clients(X, y, num_clients=NUM_CLIENTS)

    trainloader, testloader = clients[cid]

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(str(cid), trainloader, testloader, input_size).to_client(),
    )


if __name__ == "__main__":
    main()
