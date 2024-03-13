import os

import yaml
import gymnasium as gym
import torch

from algorithms.DQN.dqn import DQNAgent
from algorithms.PPO.ppo import PPOAgent
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
    io = IO()
    agent = determine_agent(config)
    network = agent.train(config)
    io.save_model(network, config)
    print("Training Done.........")





