import traci
import os
import sys
from environment.sumo.config_params import SingleIntersection,TwoIntersections
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
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)

        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        self.sumoCmd = [self.render_mode, "-c", self.network.PATH, "--start",
                        "--quit-on-end", "--collision.action", "remove",
                        "--no-warnings"]
        traci.start(self.sumoCmd)
        # traci.gui.setSchema("View #0", "real world")
        self.network.incoming_lanes(lanes=list(traci.lane.getIDList()))

    def select_size(self):
        if self.config["NUMBER_OF_INTERSECTIONS"] == 1:
            self.network = SingleIntersection(self.path)
        if self.config["NUMBER_OF_INTERSECTIONS"] == 2:
            self.network = TwoIntersections(self.path)
        if self.config["NUMBER_OF_INTERSECTIONS"] == 4:
            self.network = FourIntersections(self.path)
        if self.config["NUMBER_OF_INTERSECTIONS"] == 8:
            self.network = EightIntersections(self.path)

