from collections import namedtuple
import traci
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

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
        self.test_steps = 100


    def parameters(self) -> dict:

        with open('config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

    def run(self):
        simple_data = self.simple()
        actuated_data = self.actuated()
        delay_data = self.delay_based()
        marl_data = self.marl()
        print(simple_data)
        print(actuated_data)
        print(delay_data)
        print(marl_data)
        self.plot(simple_data, actuated_data, delay_data, marl_data)


    def simple(self):
        data = []
        for signal in self.env.signals:
            traci.trafficlight.setProgram(signal, "static")
        for warmup in range(self.env.config["WARMUP_STEPS"]):
            traci.simulationStep()

        steps = self.test_steps * self.env.config["STEPS"]
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

        steps = self.test_steps * self.env.config["STEPS"]
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

        steps = self.test_steps * self.env.config["STEPS"]
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

                observation, reward, terminated, truncated, _ = self.env.step(actions)
                data.append(self.log_values())
                if terminated or truncated:
                    done = True

            print(episode)
            return data

    def plot(self, static, actuated, delayed, marl):
        data = 3
        static = np.array([row[data] for row in static])
        actuated = np.array([row[data] for row in actuated])
        delayed = np.array([row[data] for row in delayed])
        marl = np.array([row[data] for row in marl])
        x = np.arange(len(static))

        plt.figure(figsize=[10, 5])  # a new figure window
        plt.plot(x, static, label='static')
        plt.plot(x, actuated, label='actuated')
        plt.plot(x, delayed, label='delayed')
        #plt.plot(x, marl, label='marl')
        plt.legend()
        plt.show()
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