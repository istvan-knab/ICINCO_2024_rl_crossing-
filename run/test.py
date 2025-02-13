from collections import namedtuple
import traci
import torch
import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
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
        self.print_results(simple_data, actuated_data, delay_data, marl_data)
        self.plot(simple_data, actuated_data, delay_data, marl_data)

    def print_results(self,simple_data_in, actuated_data_in, delay_data_in, marl_data_in):

        simple_data_in = np.array(simple_data_in)
        actuated_data_in = np.array(actuated_data_in)
        delay_data_in = np.array(delay_data_in)
        marl_data_in = np.array(marl_data_in)
        simple_data = []
        actuated_data = []
        delay_data = []
        marl_data = []
        for i in range(5):
            simple_data.append(simple_data_in[:,i])
            actuated_data.append(actuated_data_in[:, i])
            delay_data.append(delay_data_in[:, i])
            marl_data.append(marl_data_in[:,i])


        print("\n")
        print("\t \t \t Waiting time \t \t \t \t AVG speed \t \t  \t \t CO2 \t \t \t \t \t \t  "
              "NOx \t \t \t \t  \t \t  Halting Vehicles")
        print(f"Static    : {np.mean(simple_data[0])} \t \t {np.mean(simple_data[1])} \t \t \t  {np.mean(simple_data[2])} "
              f"\t \t  {np.mean(simple_data[3])} \t \t {np.sum(simple_data[4])}")
        print(f"Actuated  : {np.mean(actuated_data[0])} \t \t {np.mean(actuated_data[1])} \t \t \t  {np.mean(actuated_data[2])} "
              f"\t \t  {np.mean(actuated_data[3])} \t \t {np.sum(actuated_data[4])}")
        print(f"Delayed   : {np.mean(delay_data[0])} \t \t {np.mean(delay_data[1])} \t \t \t {np.mean(delay_data[2])} "
              f"\t \t {np.mean(delay_data[3])} \t \t {np.sum(delay_data[4])}")
        print(f"MARL      : {np.mean(marl_data[0])} \t \t {np.mean(marl_data[1])} \t \t \t {np.mean(marl_data[2])} "
              f"\t \t {np.mean(marl_data[3])} \t \t {np.sum(marl_data[4])}")

    def simple(self):
        print("Testing Simple...")
        data = []
        arrived = 0
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
        print("Testing Actuated...")
        data = []
        arrived = 0

        for signal in self.env.signals:
            traci.trafficlight.setProgram(signal, "static")

        for warmup in range(self.env.config["WARMUP_STEPS"]):
            traci.simulationStep()

        for signal in self.env.signals:
            traci.trafficlight.setProgram(signal, "actuated")

        steps = self.env.config["max_step"]
        for step in range(steps):
            traci.simulationStep()
            data.append(self.log_values())
        return data

    def delay_based(self):
        print("Testing DelayBased...")
        data = []
        arrived = 0

        for signal in self.env.signals:
            traci.trafficlight.setProgram(signal, "static")

        for warmup in range(self.env.config["WARMUP_STEPS"]):
            traci.simulationStep()

        for signal in self.env.signals:
            traci.trafficlight.setProgram(signal, "delay")

        steps = self.env.config["max_step"]
        for step in range(steps):
            traci.simulationStep()
            data.append(self.log_values())


        return data

    def marl(self):
        print("Testing MARL...")
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

            data_shape = int((np.shape(np.array(data).flatten())[0]) / 6)
            data = np.reshape(data,(data_shape,6))
            return data

    def plot(self, static, actuated, delayed, marl):
        "This describes which data is relevant in a certain test"
        data = 0
        window_size = 400
        static = np.array([row[data] for row in static])
        actuated = np.array([row[data] for row in actuated])
        delayed = np.array([row[data] for row in delayed])
        marl = np.array([row[data] for row in marl])
        x = np.arange(len(static))

        mpl.rcParams['axes.facecolor'] = '#EEF3F9'
        static = pd.DataFrame(static, columns=['data'])
        actuated = pd.DataFrame(actuated, columns=['data'])
        delayed = pd.DataFrame(delayed, columns=['data'])
        marl = pd.DataFrame(marl, columns=['data'])

        static['smoothed_data'] = static['data'].rolling(window=window_size).mean()
        actuated['smoothed_data'] = actuated['data'].rolling(window=window_size).mean()
        delayed['smoothed_data'] = delayed['data'].rolling(window=window_size).mean()
        marl['smoothed_data'] = marl['data'].rolling(window=window_size).mean()

        plt.figure(figsize=[10, 7])  # a new figure window
        plt.plot(x, static["smoothed_data"], label='Static',color='#000099')
        plt.plot(x, actuated["smoothed_data"], label='Actuated',color='#0066CC')
        plt.plot(x, delayed["smoothed_data"], label='Delay Based',color='#009999')
        plt.plot(x, marl["smoothed_data"], label='MARL',color='#f90001')
        plt.legend(fontsize='large')
        plt.ylabel("Waiting time [s]")
        plt.grid(True, linewidth=1, linestyle='-', color='#ead1dc')
        plt.show()

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
        arrived_vehicles = traci.simulation.getArrivedNumber()

        return [avg_waiting_time, avg_speed, avg_co2, avg_nox, avg_halting_vehicles, arrived_vehicles]

if __name__ == '__main__':
    print("this is a forked repo")
    a = 3
    test = TestTraffic()
    test.run()