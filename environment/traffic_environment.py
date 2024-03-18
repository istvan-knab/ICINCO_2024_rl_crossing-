import gymnasium as gym
import numpy as np
import os
import sys
import traci
import yaml

from environment.sumo.network import Network


class TrafficEnvironment(gym.Env):
    def __init__(self):

        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.config()
        self.render()
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.network = Network(self.config, self.path, self.render_mode)

    def get_state(self):
        pass
    def get_reward(self):
        pass

    def config(self):
        with open('env_config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)


    def step(self, action) -> None:
        traci.simulationStep(10.0)
        info = {}
        observation = self.get_state()
        reward = self.get_reward()
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated,  info


    def reset(self) -> None:
        traci.load(self.network.sumoCmd[1:])
        traci.gui.setSchema("View #0", "real world")

    def render(self) -> None:
        """
        This function has no influence, sumo does it
        """
        if self.config["RENDER_MODE"] == "human":
            self.render_mode = "sumo-gui"
        elif self.config["RENDER_MODE"] == None:
            self.render_mode = "sumo"



env = TrafficEnvironment()
for i in range(1000):
