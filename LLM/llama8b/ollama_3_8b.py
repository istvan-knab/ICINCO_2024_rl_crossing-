import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, OrderedDict
import yaml
import numpy as np
import traci
import ollama
import re

from algorithms.DQN.epsilon_greedy import EpsilonGreedy
from algorithms.DQN.replay_memory import ReplayMemory
from algorithms.neural_networks.mlp import NN
from algorithms.logger import Logger
from algorithms.io import IO
from environment.traffic_environment import TrafficEnvironment


#REWARD VERSION

class DQNAgent(object):

    def __init__(self, config: dict) -> None:
        """
        Initializes the DQNAgent with necessary components.
        """
        self.config = config
        self.dqn_config = self.parameters()
        self.env = TrafficEnvironment()
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
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
        self.io = IO()
        self.logger = Logger(config)
        self.loss = 0

    def parameters(self) -> dict:
        """
        Reads algorithm-specific parameters from a config file.
        """
        with open('../algorithms/DQN/dqn_config.yaml', 'r') as file:
            dqn_config = yaml.safe_load(file)
        return dqn_config

    def extract_reward(self, response_text):
        """
        Extracts the first numerical reward from LLM response.
        """
        match = re.search(r"[-+]?\d*\.\d+|\d+", response_text)  # Finds first float or integer
        if match:
            return float(match.group())  # Convert to float
        else:
            print(f"[ERROR] Could not extract number from: {response_text}")
            return 0.0  # Default fallback reward

    def prompt_llm(self, state, action):
        """
        Queries the locally running Llama model via Ollama to generate a reward.
        """
        prompt = f"""
        You are a reinforcement learning environment in a traffic simulation. Given the following traffic state and action, return ONLY a numerical reward.
        State: {state}
        Action: {action}
        Provide ONLY a single float number as output, without explanation, formatting, or additional text.
        """

        try:
            response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
            #role : system, assistant or user
            reward_text = response.get("message", {}).get("content", "0.0")  #response

            #Extract numerical reward
            reward = self.extract_reward(reward_text)

            print(f"\n[LLM QUERY] - State: {state}, Action: {action}")
            print(f"[LLM RESPONSE] - Extracted Reward: {reward}\n")

            return reward

        except Exception as e:
            print(f"[ERROR] LLM query failed: {e}")
            return 0.0  # Fallback reward


    def train(self, config: dict) -> None:
        """
        Training loop using the LLM for reward computation.

        :param config: Dictionary containing RL parameters.
        """
        for episode in range(config["EPISODES"]):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=config["DEVICE"]).unsqueeze(0)
            done = False
            self.action_selection.epsilon_update()
            episode_reward = 0.0
            episode_loss = 0.0

            # Warmup steps
            for _ in range(self.env.config["WARMUP_STEPS"]):
                traci.simulationStep()

            while not done:
                states = []
                actions = []
                rewards = []

                for signal in self.env.network.instance.traffic_light:
                    state = self.env.get_state(signal)
                    state = torch.tensor(state, dtype=torch.float32, device=config["DEVICE"]).unsqueeze(0)
                    states.append(state)
                    # llm select action?
                    action = self.action_selection.epsilon_greedy_selection(self.model, state)
                    actions.append(action)

                    # Query LLM for reward labeling
                    llm_reward = self.prompt_llm(state.tolist(), action)
                    rewards.append(llm_reward)
                    print(rewards)

                # Step environment with selected actions
                observation, env_reward, terminated, truncated, _ = self.env.step(actions)

                # LLM-generated rewards, environment reward
                total_reward = sum(rewards) / len(rewards)  # Average across signals
                episode_reward += total_reward
                reward_tensor = torch.tensor([[total_reward]], device=self.device)
                done = torch.tensor([int(terminated or truncated)], device=self.device)

                # Store transitions in replay memory
                for signal in range(len(self.env.network.instance.traffic_light)):
                    next_state = torch.tensor(observation[signal], dtype=torch.float32, device=self.device).unsqueeze(0)
                    action_tensor = torch.tensor([[actions[signal]]], dtype=torch.long, device=self.device)
                    self.memory.push(states[signal], action_tensor, next_state, reward_tensor, done)

                if done:
                    break

                loss = self.fit_model()
                episode_loss += loss

            self.logger.step(episode, episode_reward, self.config, episode_loss)

            # Update target network periodically
            if episode % self.dqn_config["TAU"] == 0:
                self.target.load_state_dict(OrderedDict(self.model.state_dict()))
                self.target = self.model

        return self.model

    def fit_model(self) -> None:
        """
        Computes gradients based on sampled batches.

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

        y_batch = (reward_batch + self.config['GAMMA'] * output_next_state_batch * (1 - done_batch).view(-1, 1)).float()
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
