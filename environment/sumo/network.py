import libsumo as traci
import os
import sys
class Network:
    def __init__(self, config: dict, path: str, render_mode: str) -> None:
        self.path = path
        self.config = config
        self.render_mode = render_mode
        self.start_simulation()
        self.lanes = []
        self.junctions = []

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

        self.path = self.path + ""
        self.sumoCmd = [self.render_mode, "-c", self.path, "--start",
                        "--quit-on-end", "--collision.action", "remove",
                        "--no-warnings"]
        traci.start(self.sumoCmd)
        # traci.gui.setSchema("View #0", "real world")

