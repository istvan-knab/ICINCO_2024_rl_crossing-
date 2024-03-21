import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class ActorNetwork(nn.Module):
    def __init__(self, input_size, output_size, config):
        super(ActorNetwork, self).__init__()
        self.actor = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(),
                                   nn.Linear(128, 128), nn.ReLU(),
                                   nn.Linear(128, output_size), nn.Softmax(dim=-1))
        self.optimizer = optim.Adam(self.parameters(), lr = config["ALPHA"])

    def forward(self, x):
        distribution = self.actor(x)
        distribution = Categorical(distribution)

        return distribution

class CriticNetwork(nn.Module):
    def __init__(self, input_size, config):
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(nn.Linear(input_size, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, 1))
        self.optimizer = optim.Adam(self.parameters(), lr = config["ALPHA"])


    def forward(self, state):
        value = self.critic(state)

        return value