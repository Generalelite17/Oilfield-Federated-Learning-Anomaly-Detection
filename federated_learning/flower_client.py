import flwr as fl
import torch

from model import Net
from local_train import train_local
from sklearn.metrics import precision_score, recall_score, f1_score



class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, trainloader, testloader, input_size):
        self.cid = cid
        self.model = Net(input_size=input_size)
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {
            k: torch.tensor(v) for k, v in params_dict
        }
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_local(self.model, self.trainloader, epochs=3, lr=0.005)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        y_true = []
        y_pred = []
        loss_total = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        self.model.eval()
        with torch.no_grad():
            for X, y in self.testloader:
                X = X.view(X.size(0), -1)
                outputs = self.model(X)

                loss = criterion(outputs, y)
                loss_total += loss.item()

                _, predicted = torch.max(outputs, 1)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        accuracy = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        return float(loss_total), len(y_true), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }