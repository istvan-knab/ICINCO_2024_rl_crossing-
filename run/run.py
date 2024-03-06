import yaml
import gymnasium as gym
import torch

from algorithms.DQN.dqn import DQNAgent
from algorithms.PPO.proximal_policy_optimization import PPOAgent
from algorithms.DQN.epsilon_greedy import EpsilonGreedy

def determine_agent(config) -> object:
    if config['algorithm'] == 'DQN':
        return DQNAgent(config)
    elif config['algorithm'] == 'PPO':
        return PPOAgent(config)
    else:
        raise NotImplementedError

def parameters() -> dict:

    with open('run/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def test(config: dict) -> None:
    #torch load model
    PATH = config["PATH_TEST"]
    agent = torch.load(PATH)
    agent.eval()

    env = gym.make(config["environment"], render_mode=config["RENDER_MODE"])
    config["EPSILON"] = 0
    action_selection = EpsilonGreedy(config, env)
    for i in range(config["EPISODES"]):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=config["DEVICE"]).unsqueeze(0)
        done = False
        episode_reward = 0
        while not done:
            action = action_selection.epsilon_greedy_selection(agent, state)
            state, reward, terminated, truncated, _ = env.step(action)
            state = torch.tensor(state, dtype=torch.float32, device=config["DEVICE"]).unsqueeze(0)
            episode_reward += reward
            if config["RENDER_MODE"] == "human":
                env.render()
            if terminated or truncated:
                done = True
                print(f"Episode finished with reward : {episode_reward}")
            if done:
                break

if __name__ == '__main__':
    config = parameters()
    agent = determine_agent(config)

    if config["mode"] == "train":
        agent.train(config)
    else:
        test(config)

    print("Done.........")





