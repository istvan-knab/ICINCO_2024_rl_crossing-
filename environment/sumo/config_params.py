import numpy as np
import traci

class SumoNetworkParameters:
    def __init__(self):
        self.lanes = []
        self.traffic_light = []

    def config_net(self, lanes, junctions):
        self.incoming_lanes(lanes)
        self.get_lights(junctions)
        self.set_sections()

    def incoming_lanes(self, lanes):
        for lane in lanes:
            if "lane" in lane:
                self.lanes.append(lane)

    def get_lights(self, junctions):
        for junction in junctions:
            if "TLS" in junction:
                self.traffic_light.append(junction)

    def set_sections(self):

        self.sections = []
        count = 0
        for traffic_light in range(len(self.traffic_light)):
            single_section = [" ", " ", " ", " "]
            for section in range(4):
                single_section[section] = self.lanes[count]
                count += 1
            self.sections.append(single_section)


class SingleIntersection(SumoNetworkParameters):
    def __init__(self, path):
        self.PATH = path + "/sumo/intersection/1_intersection.sumocfg"
        self.lanes = []
        self.traffic_light = []

class TwoIntersections(SumoNetworkParameters):
    def __init__(self,path):
        self.PATH = path + "/sumo/intersection/2_intersection.sumocfg"
        self.lanes = []
        self.traffic_light = []

class ThreeIntersections(SumoNetworkParameters):
    def __init__(self,path):
        self.PATH = path + "/sumo/intersection/3_intersection.sumocfg"
        self.lanes = []
        self.traffic_light = []

class FourIntersections(SumoNetworkParameters):
    def __init__(self, path):
        self.PATH = path + "/sumo/intersection/4_intersection.sumocfg"
        self.lanes = []
        self.traffic_light = []

class EightIntersections(SumoNetworkParameters):
    def __init__(self, path):
        self.PATH = path + "/sumo/intersection/8_intersection.sumocfg"
        self.lanes = []
        self.traffic_light = []
