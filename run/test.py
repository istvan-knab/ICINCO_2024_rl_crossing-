from collections import namedtuple
import traci
import torch
import yaml

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
        self.simple()
        #self.actuated()
        #self.delay_based()
        self.marl()
    def simple(self):

        for warmup in range(self.env.config["WARMUP_STEPS"]):
            traci.simulationStep()

        steps = self.test_steps * self.env.config["STEPS"]
        for step in range(steps):
            traci.simulationStep()


    def actuated(self):
        for signal in self.env.signals:
            traci.trafficlight.setProgramLogic(signal, "actuated")

    def delay_based(self):
        for signal in self.env.signals:
            traci.trafficlight.setProgramLogic(signal, "delay_based")

    def marl(self):
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
                if terminated or truncated:
                    done = True

            print(episode)

    def log_values(self, data):
        ...

if __name__ == '__main__':
    test = TestTraffic()
    test.run()