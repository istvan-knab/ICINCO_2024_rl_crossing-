import traci
import os
import sys
from environment.sumo.config_params import SingleIntersection,TwoIntersections, ThreeIntersections
from environment.sumo.config_params import FourIntersections,EightIntersections
class Network:
    def __init__(self, config: dict, path: str, render_mode: str) -> None:
        self.path = path
        self.config = config
        self.render_mode = render_mode
        self.select_size()
        self.start_simulation()


    def start_simulation(self):
        """
            This function is responsible to build connection between gui and python
            :return: None
        """
        sumo_home = "../sumo"
        os.environ["SUMO_HOME"] = sumo_home
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)

        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        self.sumoCmd = [self.config["RENDER_MODE"], "-c", self.instance.PATH, "--start",
                        "--quit-on-end", "--collision.action", "remove",
                        "--no-warnings"]

        traci.start(self.sumoCmd)
        if self.render_mode == 'human':
            traci.gui.setSchema("View #0", "real world")
        self.instance.config_net(lanes=list(traci.lane.getIDList()), junctions=list(traci.trafficlight.getIDList()))

    def select_size(self):
        if self.config["NUMBER_OF_INTERSECTIONS"] == 1:
            self.instance = SingleIntersection(self.path)
        if self.config["NUMBER_OF_INTERSECTIONS"] == 2:
            self.instance = TwoIntersections(self.path)
        if self.config["NUMBER_OF_INTERSECTIONS"] == 3:
            self.instance = ThreeIntersections(self.path)
        if self.config["NUMBER_OF_INTERSECTIONS"] == 4:
            self.instance = FourIntersections(self.path)
        if self.config["NUMBER_OF_INTERSECTIONS"] == 8:
            self.instance = EightIntersections(self.path)

