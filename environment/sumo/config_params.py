import traci

class ParentNetwork:
    def __init__(self):
        self.lanes = []
        self.traffic_light = []

    def incoming_lanes(self, lanes):
        for lane in lanes:
            if "+lane" in lane:
                self.lanes.append(lane)

    def get_lights(self):
        pass


class SingleIntersection(ParentNetwork):
    def __init__(self, path):
        self.PATH = path + "/sumo/intersection/1_intersection.sumocfg"
        self.lanes = []
        self.traffic_light = self.get_lights()

    def get_lights(self):
        traffic_light = ["J1"]
        return traffic_light

class TwoIntersections(ParentNetwork):
    def __init__(self,path):
        self.PATH = path + "/sumo/intersection/2_intersection.sumocfg"
        self.lanes = []
        self.traffic_light = []

    def get_lights(self):
        traffic_light = ["J6", "J8"]
        return traffic_light
class FourIntersections(ParentNetwork):
    def __init__(self, path):
        self.PATH = path + "/sumo/intersection/4_intersection.sumocfg"
        self.lanes = []
        self.traffic_light = []

    def get_lights(self):
        traffic_light = ["J1", "J3", "J2", "J4"]
        return traffic_light
class EightIntersections(ParentNetwork):
    def __init__(self, path):
        self.PATH = path + "/sumo/intersection/8_intersection.sumocfg"
        self.lanes = []
        self.traffic_light = []

    def get_lights(self):
        traffic_light = ["J1", "J3", "J7", "J10", "J2", "J5", "J8", "J9"]
        return traffic_light