import gymnasium as gym
import numpy as np
import os
import sys
import libsumo as traci
import yaml


class TrafficEnvironment(gym.Env):
    def __init__(self):

        with open('env_config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        number_of_intersections = "1_intersection.sumocfg"
        if config["RENDER_MODE"] == "human":
            self.render_mode = "sumo-gui"
        elif config["RENDER_MODE"] == None:
            self.render_mode = "sumo"

        self.path = os.path.dirname(os.path.abspath(__file__)) + "/sumo/intersection/" + number_of_intersections
        self.sumo_start()

    def sumo_start(self) -> None:
        """
            This function is responsible to build connection between gui and python
            :return: None
        """
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)

        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        self.path = self.path + ""
        self.sumoCmd = [self.render_mode, "-c", self.path, "--start",
                        "--quit-on-end", "--collision.action", "remove",
                        "--no-warnings"]
        traci.start(self.sumoCmd)
        # traci.gui.setSchema("View #0", "real world")

    def step(self, action) -> None:
        traci.load(self.sumoCmd[1:])

    def reset(self) -> None:
        pass

    def render(self) -> None:
        """
        This function has no influence, sumo does it
        """
        pass

    def get_density(self):
        pass


sumo = TrafficEnvironment()
