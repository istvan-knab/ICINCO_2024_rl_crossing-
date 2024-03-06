import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
from collections import OrderedDict
import gymnasium as gym
import atari_py
import yaml
import numpy as np

from algorithms.DQN.epsilon_greedy import EpsilonGreedy
from algorithms.DQN.replay_memory import ReplayMemory
from algorithms.neural_networks.mlp import NN
from algorithms.logger import Logger
from algorithms.io import IO


class DQNAgent(object):

    def __init__(self, config: dict) -> None:

        self.config = config
        self.dqn_config = self.parameters()
        self.env = gym.make(config["environment"], render_mode=config["RENDER_MODE"])
        config["state_size"] = self.env.observation_space.shape[0]
        config["action_size"] = self.env.action_space.n
        self.device = config["DEVICE"]
        self.criterion = nn.MSELoss()
        self.action_selection = EpsilonGreedy(config, self.env)
        self.memory = ReplayMemory(self.dqn_config["MEMORY_SIZE"], self.dqn_config["BATCH_SIZE"])
        self.model = NN(config).to(self.device)
        self.target = NN(config).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["ALPHA"], amsgrad=False)
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward', 'done'))
        self.io = IO()
        self.logger = Logger(config)
        self.loss = 0

    def parameters(self) -> dict:
        """
        Reading algorithm specific parameters from a config file
        :return: dict
        """
        with open('algorithms/DQN/dqn_config.yaml', 'r') as file:
            dqn_config = yaml.safe_load(file)
        return dqn_config

    def train(self, config: dict) -> None:
        """
        Training loop
        :param config: it contains all the necessary rl parameters, that are not DQN specific
        :return:
        """
        for episode in range(config["EPISODES"]):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=config["DEVICE"]).unsqueeze(0)
            done = False
            self.action_selection.epsilon_update()
            episode_reward = 0
            episode_loss = 0

            while not done:

                action = self.action_selection.epsilon_greedy_selection(self.model, state)
                observation, reward, terminated, truncated, _ = self.env.step(action)

                reward = torch.tensor([[reward]], device=self.device)
                episode_reward += reward
                done = torch.tensor([int(terminated or truncated)], device=self.device)


                next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = torch.tensor([[action]], dtype=torch.long, device=self.device)

                self.memory.push(state, action, next_state, reward, done)
                state = next_state

                loss = self.fit_model()
                episode_loss += loss

                if done:
                    break

            self.logger.step(episode, episode_reward, self.config, episode_loss)
            if episode % self.dqn_config["TAU"]:
                self.target.load_state_dict(OrderedDict(self.model.state_dict()))
                self.target = self.model

        self.io.save_model(self.model, self.config)



    def fit_model(self) -> None:
        """
        Gradient computation based on the sampled batches
        :return: None
        """
        if len(self.memory) < self.dqn_config["BATCH_SIZE"]:
            return 0
        sample = self.memory.sample()
        batch = self.Transition(*zip(*sample))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        with torch.no_grad():
            output_next_state_batch = self.target(next_state_batch).detach()
            output_next_state_batch = torch.max(output_next_state_batch, 1)[0].detach()
            output_next_state_batch = torch.reshape(output_next_state_batch,
                                                    (self.dqn_config["BATCH_SIZE"], -1)).detach()

        y_batch = (reward_batch + self.config['GAMMA'] * output_next_state_batch * (1- done_batch).view(-1, 1)).float()
        output = torch.reshape(self.model(state_batch), (self.dqn_config["BATCH_SIZE"], -1))
        q_values = torch.gather(output, 1, action_batch)


        loss = self.criterion(q_values, y_batch)
        self.loss = float(loss)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss


