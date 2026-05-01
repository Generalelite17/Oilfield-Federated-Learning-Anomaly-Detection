import flwr as fl
import torch
import torch.nn as nn

from config import LOCAL_EPOCHS, LEARNING_RATE
from model import FederatedNN, train_one_epoch


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, trainloader, input_dim):
        self.cid = cid
        self.trainloader = trainloader
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = FederatedNN(input_dim=input_dim).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {
            key: torch.tensor(value)
            for key, value in params_dict
        }
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        losses = []
        for _ in range(LOCAL_EPOCHS):
            loss = train_one_epoch(
                self.model,
                self.trainloader,
                self.criterion,
                self.optimizer,
                self.device,
            )
            losses.append(loss)

        avg_loss = sum(losses) / len(losses)

        return self.get_parameters(config={}), len(self.trainloader.dataset), {
            "client_id": int(self.cid),
            "loss": float(avg_loss),
        }