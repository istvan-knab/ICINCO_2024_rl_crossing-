import torch
import gymnasium as gym
import yaml
import numpy as np
from collections import namedtuple

from algorithms.DQN.epsilon_greedy import EpsilonGreedy
from algorithms.neural_networks.actor_critic import ActorNetwork, CriticNetwork
from algorithms.PPO.ppo_memory import Memory
from algorithms.logger import Logger
from algorithms.io import IO

class PPOAgent(object):
    def __init__(self, config):
        with open('algorithms/PPO/ppo_config.yaml', 'r') as file:
            self.ppo_config = yaml.safe_load(file)
        self.env = gym.make(config["environment"], render_mode=config["RENDER_MODE"])
        self.action_selection = EpsilonGreedy(config, self.env)
        self.config = config
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        self.actor = ActorNetwork(state_size, action_size, config)
        self.critic = CriticNetwork(state_size, config)
        self.memory = Memory(self.ppo_config["BATCH_SIZE"])
        self.logger = Logger(config)
        self.seed = config["SEED"]
        #TODO:transition also in memory
        #self.transition = namedtuple('Transition', ('state', 'action', ''))


    def train(self, config):
        score_history = []
        avg_score = []
        number_of_steps = 0
        loss = 0
        for episode in range(config["EPISODES"]):
            state, _  = self.env.reset(seed=self.seed)
            done = False
            episode_reward = 0
            while not done:
                number_of_steps += 1
                action, probability, value = self.choose_action(torch.tensor(state))
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    done = True
                self.memory.push(state, action, probability, value, reward, done)
                episode_reward += reward
                state = next_state
                if number_of_steps % self.ppo_config["TAU"] == 0:
                    loss = self.fit()

            score_history.append(episode_reward)
            avg_score.append(np.mean(score_history[100:]))
            self.logger.step(episode, episode_reward, self.config, loss)



    def choose_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float32)
        distribution = self.actor(state)
        value = self.critic(state)
        action = distribution.sample()

        probabilities = torch.squeeze(distribution.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probabilities, value

    def fit(self):

        for run_policy_iteration in range(self.ppo_config["POLICY_ITERATIONS"]):
            episode_loss = 0
            state_batch, action_batch, old_probabilities_batch, values,\
            rewards, dones, batches = self.memory.generate_batch()
            advantages = np.zeros(len(rewards), dtype=np.float32)
            for t in range(len(rewards)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards)-1):
                    a_t += discount * (rewards[k] + self.config["GAMMA"] * values[k + 1] * (1 - int(dones[k])) - values[k])
                    discount = self.config["GAMMA"] * self.ppo_config["LAMBDA"]
                advantages[t] = a_t
            advantages = torch.tensor(advantages, dtype=torch.float32)
            values = torch.tensor(values, dtype=torch.float32)
            for element in batches:
                states = torch.tensor(state_batch[element], dtype=torch.float32)
                old_probabilities = torch.tensor(old_probabilities_batch[element], dtype=torch.float32)
                actions = torch.tensor(action_batch[element], dtype=torch.int32)

                distribution = self.actor(states)
                critic_value = self.critic(states)
                new_probabilities = distribution.log_prob(actions)
                probability_ratio = new_probabilities.exp() / old_probabilities.exp()
                weighted_probabilities = advantages[element] * probability_ratio
                weighted_clipped_probabilities = torch.clamp(probability_ratio, 1 - self.ppo_config["CLIP_PROBABILITY"],
                                                             1 + self.ppo_config["CLIP_PROBABILITY"])  * advantages[element]
                actor_loss = - torch.min(weighted_probabilities, weighted_clipped_probabilities).mean()

                returns = advantages[element] + values[element]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5 * critic_loss
                episode_loss =+ total_loss

                self. actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear()
        return episode_loss





