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
        self.simulation_step = 0

    def action_handler(self, action, signal):
        """
        TLS incoming lanes states
        """
        observation = observation = np.array([0, 0, 0, 0])
        return observation

    def get_state(self):
        observation = observation = np.array([0, 0, 0, 0])
        return observation
    def get_reward(self):
        "The least waiting time on the whole network"
        waiting_times = 0
        lanes = self.network.instance.lanes
        for lane in lanes:
            waiting_times += traci.lane.getWaitingTime(lane)
        return waiting_times

    def is_terminal(self):
        if self.simulation_step % self.config['max_step'] == 0:
            terminated = True
        else :
            terminated = False

        return terminated

    def config(self):
        with open('../environment/env_config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)


    def step(self, action, signal) -> None:

        for seconds in range(self.config['STEPS']):
            traci.simulationStep()
            self.simulation_step += 1
        info = {}
        observation = self.action_handler(action, signal)
        reward = self.get_reward()
        terminated = self.is_terminal()
        truncated = False

        return observation, reward, terminated, truncated, info


    def reset(self) -> None:
        traci.load(self.network.sumoCmd[1:])
        traci.gui.setSchema("View #0", "real world")
        observation = np.array([0, 0, 0, 0])
        info = {}
        self.simulation_step = 0


        return observation, info

    def render(self) -> None:
        """
        This function has no influence, sumo does it
        """
        if self.config["RENDER_MODE"] == "human":
            self.render_mode = "sumo-gui"
        elif self.config["RENDER_MODE"] == None:
            self.render_mode = "sumo"


