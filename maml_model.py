import torch.nn as nn
from collections import OrderedDict

class MAMLModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MAMLModel, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(in_features, hidden_features)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(hidden_features, hidden_features//4)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(hidden_features//4, hidden_features//8)),
            ('relu3', nn.ReLU()),
            ('l4', nn.Linear(hidden_features//8, out_features)),
        ]))
        self.out_features = out_features

    def forward(self, x):
        return self.model(x)

    def parameterised(self, x, weights):
        x = nn.functional.linear(x, weights[0], weights[1])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[2], weights[3])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[4], weights[5])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[6], weights[7])
        return x
