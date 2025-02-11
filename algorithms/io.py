import os

import torch
import time
class IO(object):
    def __init__(self):
        pass
    def read_to_np_array(self, filename):
        pass

    def save_model(self, model, config, intersections, id: str):
        PATH = (config['PATH'] + '/' + id + "_" + str(config["algorithm"]) +
                 "_" + str(intersections) + ".pth")
        torch.save(model, PATH)


