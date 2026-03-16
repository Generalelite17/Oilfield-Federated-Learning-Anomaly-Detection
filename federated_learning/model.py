import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super(SimpleModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)