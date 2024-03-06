import torch
import torch.nn as nn
import torch.nn.functional as F
class NN(nn.Module):
    """
    Neural Network todo custom implementation
    """
    def __init__(self, config):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(config["state_size"], 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, config["action_size"])

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)