import os

import torch
import time
class IO(object):
    def __init__(self):
        pass
    def read_to_np_array(self, filename):
        pass

    def save_model(self, model, config):
        PATH = (config['PATH'] + '/' + str(config["environment"]) + "_" + str(config["algorithm"]) +
                "_" + str(time.time()) + ".pth")
        torch.save(model, PATH)


