import os

import yaml
import gymnasium as gym
import torch
import numpy as np

from algorithms.DQN.dqn import DQNAgent
from algorithms.PPO.proximal_policy_optimization import PPOAgent
from algorithms.io import IO
from algorithms.DQN.epsilon_greedy import EpsilonGreedy

def determine_agent(config) -> object:
    if config['algorithm'] == 'DQN':
        return DQNAgent(config)
    elif config['algorithm'] == 'PPO':
        return PPOAgent(config)
    else:
        raise NotImplementedError

def parameters() -> dict:

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == '__main__':
    config = parameters()
    seed = config.get('seed', 0)  # you can put any default value you want
    torch.manual_seed(seed)
    np.random.seed(seed)
    io = IO()
    agent = determine_agent(config)
    network = agent.train(config)
    io.save_model(network, config, agent.env.config['NUMBER_OF_INTERSECTIONS'], agent.logger.id)
    print("Training Done.........")





