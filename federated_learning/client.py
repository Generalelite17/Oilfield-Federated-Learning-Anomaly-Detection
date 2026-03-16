import torch
import torch.nn as nn
import torch.optim as optim

def train_local(model, dataloader, epochs=1, lr=0.01):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for _ in range(epochs):
        for X, y in dataloader:
            X = X.view(X.size(0), -1)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    return model.state_dict()