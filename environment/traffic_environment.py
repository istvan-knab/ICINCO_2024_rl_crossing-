import gymnasium as gym
import numpy as np
import os
import sys
import traci
import yaml

from environment.sumo.network import Network


class TrafficEnvironment(gym.Env):
    def __init__(self):

        self.config()
        self.render()
        number_of_intersections = "1_intersection.sumocfg"
        self.path = os.path.dirname(os.path.abspath(__file__)) + "/sumo/intersection/" + number_of_intersections
        self.network = Network(self.config, self.path, self.render_mode)



    def config(self):
        with open('env_config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)


    def step(self, action) -> None:
        ...

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

class Observation:
    def __init__(self):
        ...

class Action:
    def __init__(self):
        ...

env = TrafficEnvironment()