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


class DQNAgent(object):

    def __init__(self, config: dict) -> None:
        """
        Initializes the DQNAgent with necessary components. DQNAGENT
        """
        self.config = config
        self.dqn_config = self.parameters()
        self.prompt_config = self.load_prompt_config()
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

    def load_prompt_config(self) -> dict:
        """
        Loads the prompt parameters from a config file.
        """
        with open('prompt_config.yaml', 'r') as file:
            return yaml.safe_load(file)

    def extract_action(self, response_text):
        """
        Extracts the first integer action from LLM response.
        """
        match = re.search(r"\d+", response_text)  # Finds first integer
        if match:
            return int(match.group())  # Convert to integer
        else:
            print(f"[ERROR] Could not extract action from: {response_text}")
            return 0  # Default fallback action

    def prompt_llm(self, state, action_space, expected_rewards):
        """
        Queries the locally running Llama model via Ollama to select the best action.
        """
        prompt_template = self.prompt_config["prompt_template"]

        # Construct Chain of Thought reasoning steps
        chain_of_thought_steps = "\n".join(
            f"{i + 1}. {step}" for i, step in enumerate(prompt_template["chain_of_thought"]))

        prompt = f"""
        {prompt_template["content"]}

        State: {state}
        Available Actions: {action_space}
        Expected Rewards: {expected_rewards}
        
        Follow these reasoning steps to determine the best action:
        {chain_of_thought_steps}

        {prompt_template["output_format"]["description"]}
        """
        try:
            response = ollama.chat(model=prompt_template["model"], messages=[{"role": "user", "content": prompt}])
            action_text = response.get("message", {}).get("content", "0")
            action = self.extract_action(action_text)
            print(f"\n[LLM QUERY] - State: {state}, Action Space: {action_space}, Expected Rewards: {expected_rewards}")
            print(f"[LLM RESPONSE] - Selected Action: {action}\n")
            return action
        except Exception as e:
            print(f"[ERROR] LLM query failed: {e}")
            return np.argmax(expected_rewards)  # Default to best expected reward if LLM fails

    def train(self, config: dict) -> None:
        for episode in range(config["EPISODES"]):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=config["DEVICE"]).unsqueeze(0)
            done = False
            self.action_selection.epsilon_update()
            episode_reward = 0.0
            episode_loss = 0.0
            for warmup in range(self.env.config["WARMUP_STEPS"]):
                traci.simulationStep()
            while not done:
                states = []
                actions = []
                for signal in self.env.network.instance.traffic_light:
                    state = self.env.get_state(signal)
                    state = torch.tensor(state, dtype=torch.float32, device=config["DEVICE"]).unsqueeze(0)
                    states.append(state)
                    action_space = list(range(self.env.action_space.n))
                    expected_rewards = self.model(state).detach().to("cuda")
                    action = self.prompt_llm(state.tolist(), action_space, expected_rewards)
                    actions.append(action)
                observation, reward, terminated, truncated, _ = self.env.step(actions)
                episode_reward += reward
                reward = torch.tensor([[reward]], device=self.device)
                done = torch.tensor([int(terminated or truncated)], device=self.device)
                for signal in range(len(self.env.network.instance.traffic_light)):
                    next_state = torch.tensor(observation[signal], dtype=torch.float32, device=self.device).unsqueeze(0)
                    action_tensor = torch.tensor([[actions[signal]]], dtype=torch.long, device=self.device)
                    self.memory.push(states[signal], action_tensor, next_state, torch.tensor([[0.0]], device=self.device), done)
                if done:
                    break
                loss = self.fit_model()
                episode_loss += loss
            self.logger.step(episode, episode_reward, self.config, episode_loss)
            if episode % self.dqn_config["TAU"] == 0:
                self.target.load_state_dict(OrderedDict(self.model.state_dict()))
                self.target = self.model
        return self.model
