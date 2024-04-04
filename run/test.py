from collections import namedtuple
import traci
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from environment.traffic_environment import TrafficEnvironment
from algorithms.DQN.epsilon_greedy import EpsilonGreedy
class TestTraffic:
    def __init__(self):
        self.modes = ['simple', 'adaptive', 'TSC']
        self.data = namedtuple('Data',
                                ('queue_length', 'waiting_time', 'co2',
                                 'nox', 'halting_vehicles', 'arrived_vehicles'))
        self.parameters()
        self.env = TrafficEnvironment()
        self.action_selection = EpsilonGreedy(self.config, self.env)


    def parameters(self) -> dict:

        with open('config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

    def run(self):
        simple_data = self.simple()
        self.env.reset()
        actuated_data = self.actuated()
        self.env.reset()
        delay_data = self.delay_based()
        marl_data = self.marl()
        self.plot(simple_data, actuated_data, delay_data, marl_data)


    def simple(self):
        data = []
        for signal in self.env.signals:
            traci.trafficlight.setProgram(signal, "static")
        for warmup in range(self.env.config["WARMUP_STEPS"]):
            traci.simulationStep()

        steps = self.env.config["max_step"]
        for step in range(steps):
            traci.simulationStep()
            data.append(self.log_values())

        return data


    def actuated(self):
        data = []
        for signal in self.env.signals:
            traci.trafficlight.setProgram(signal, "actuated")

        for warmup in range(self.env.config["WARMUP_STEPS"]):
            traci.simulationStep()

        steps = self.env.config["max_step"]
        for step in range(steps):
            traci.simulationStep()
            data.append(self.log_values())
        return data

    def delay_based(self):
        data = []
        for signal in self.env.signals:
            traci.trafficlight.setProgram(signal, "delay")

        for warmup in range(self.env.config["WARMUP_STEPS"]):
            traci.simulationStep()

        steps = self.env.config["max_step"]
        for step in range(steps):
            traci.simulationStep()
            data.append(self.log_values())

        return data

    def marl(self):
        data = []
        PATH = self.config["PATH_TEST"]
        agent = torch.load(PATH)
        agent.eval()
        self.config["EPSILON"] = 0

        for signal in self.env.signals:
            traci.trafficlight.setProgram(signal, "static")

        for episode in range(self.config["EPISODES"]):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.config["DEVICE"]).unsqueeze(0)
            done = False
            #warmup
            for warmup in range(self.env.config["WARMUP_STEPS"]):
                traci.simulationStep()
            while not done:
                states = []
                actions = []
                for signal in self.env.network.instance.traffic_light:
                    state = self.env.get_state(signal)
                    state = torch.tensor(state, dtype=torch.float32, device=self.config["DEVICE"]).unsqueeze(0)
                    states.append(state)
                    action = self.action_selection.epsilon_greedy_selection(agent, state)
                    actions.append(action)

                observation, reward, terminated, truncated, episode_data = self.env.step(actions)
                data.append(episode_data)
                if terminated or truncated:
                    done = True

            data_shape = int((np.shape(np.array(data).flatten())[0]) / 5)
            data = np.reshape(data,(data_shape,5))
            return data

    def plot(self, static, actuated, delayed, marl):
        "This describes which data is relevant in a certain test"
        data = 3
        window_size = 50
        static = np.array([row[data] for row in static])
        actuated = np.array([row[data] for row in actuated])
        delayed = np.array([row[data] for row in delayed])
        marl = np.array([row[data] for row in marl])
        x = np.arange(len(static))

        static = pd.DataFrame(static, columns=['data'])
        actuated = pd.DataFrame(actuated, columns=['data'])
        delayed = pd.DataFrame(delayed, columns=['data'])
        marl = pd.DataFrame(marl, columns=['data'])

        static['smoothed_data'] = static['data'].rolling(window=window_size).mean()
        actuated['smoothed_data'] = actuated['data'].rolling(window=window_size).mean()
        delayed['smoothed_data'] = delayed['data'].rolling(window=window_size).mean()
        marl['smoothed_data'] = marl['data'].rolling(window=window_size).mean()

        plt.figure(figsize=[10, 5])  # a new figure window
        plt.plot(x, static["smoothed_data"], label='static')
        plt.plot(x, actuated["smoothed_data"], label='actuated')
        plt.plot(x, delayed["smoothed_data"], label='delayed')
        plt.plot(x, marl["smoothed_data"], label='marl')
        plt.legend()
        plt.show()

        print(f"Static : {np.mean(static['data'])}")
        print(f"Delayed : {np.mean(delayed['data'])}")
        print(f"Actuated : {np.mean(actuated['data'])}")
        print(f"MARL : {np.mean(marl['data'])}")

    def filter_data(self,*args):
        filtered_data = None
        return filtered_data
    def log_values(self):
        waiting_time = []
        speed = []
        co2 = []
        nox = []
        halting_vehicles = []

        for lane in self.env.network.instance.lanes:
            waiting_time.append(traci.lane.getWaitingTime(lane))
            speed.append(traci.lane.getLastStepMeanSpeed(lane))
            co2.append(traci.lane.getCO2Emission(lane))
            nox.append(traci.lane.getNOxEmission(lane))
            halting_vehicles.append(traci.lane.getLastStepHaltingNumber(lane))

        avg_waiting_time = np.mean(waiting_time)
        avg_speed = np.mean(speed)
        avg_co2 = np.mean(co2)
        avg_nox = np.mean(nox)
        avg_halting_vehicles = np.mean(halting_vehicles)

        return [avg_waiting_time, avg_speed, avg_co2, avg_nox, avg_halting_vehicles]

if __name__ == '__main__':
    test = TestTraffic()
    test.run()