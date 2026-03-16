import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import csv

from model import SimpleModel
from client import train_local
from server import average_weights

def split_dataset(dataset, num_clients=3):
    data_size = len(dataset) // num_clients
    subsets = []

    for i in range(num_clients):
        start = i * data_size
        end = (i + 1) * data_size if i != num_clients - 1 else len(dataset)
        subsets.append(Subset(dataset, list(range(start, end))))

    return subsets

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X = X.view(X.size(0), -1)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return 100 * correct / total

def main():
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    client_datasets = split_dataset(train_dataset, num_clients=3)
    client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    global_model = SimpleModel()

    rounds = 3
    results = []
    for round_num in range(rounds):
        client_weights = []

        for loader in client_loaders:
            local_model = SimpleModel()
            local_model.load_state_dict(global_model.state_dict())

            updated_weights = train_local(local_model, loader, epochs=1)
            client_weights.append(updated_weights)

        new_global_weights = average_weights(client_weights)
        global_model.load_state_dict(new_global_weights)

        accuracy = evaluate(global_model, test_loader)
        results.append((round_num + 1, accuracy))
        print(f"Round {round_num + 1}, Test Accuracy: {accuracy:.2f}%")
    
    with open("../results/federated_results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Accuracy"])

        for round_num, accuracy in results:
            writer.writerow([round_num, accuracy])

if __name__ == "__main__":
    main()

