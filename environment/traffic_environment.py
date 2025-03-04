import gymnasium as gym
import numpy as np
import os
import sys
import traci
import yaml

from environment.sumo.network import Network


class TrafficEnvironment(gym.Env):
    def __init__(self):

        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2, seed=42)
        self.config()
        self.render()
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.network = Network(self.config, self.path, self.render_mode)
        self.simulation_step = 0
        self.signals = self.network.instance.traffic_light
        self.signal = self.signals[0]
        self.arrived = 0

    def active_lanes(self, signal):
        counter = 0
        for pointer in self.signals:
            if signal == pointer:
                return counter
            counter +=1
    def action_handler(self, action, signal):
        """
        TLS incoming lanes states
        """
        if action is None:
            print(f"[ERROR] Received None action for signal {signal}. Defaulting to 0.")
            action = 0  # Default safe action
        action = action * 2
        traci.trafficlight.setPhase(signal, action)


    def get_state(self, signal):
        index = self.active_lanes(signal)
        scope = self.network.instance.sections[index][:]
        observation = np.array([0, 0, 0, 0])
        count = 0
        for lane in scope:
            observation[count] = traci.lane.getLastStepMeanSpeed(lane)
            count += 1
        return observation
    def get_reward(self):
        "The least waiting time on the whole network"
        waiting_times = 0
        speed = 0
        lanes = self.network.instance.lanes
        for lane in lanes:
            waiting_times += traci.lane.getWaitingTime(lane)
            speed += traci.lane.getLastStepMeanSpeed(lane)
        reward = (speed / len(lanes))/ (1.0 + waiting_times)
        return reward

    def is_terminal(self):
        if self.simulation_step % self.config['max_step'] == 0:
            terminated = True
        else :
            terminated = False

        return terminated

    def config(self):
        with open('../environment/env_config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)


    def step(self, action) -> None:
        observation = []
        info = []

        count = 0
        for signal in self.network.instance.traffic_light:
            self.action_handler(action[count], signal)
            count += 1
        for step in range (self.config['STEPS']):
            traci.simulationStep()
            if self.config["mode"] == "test":
                info.append(self.log_values())
        self.simulation_step += self.config["STEPS"]
        reward = self.get_reward()
        for signal in self.network.instance.traffic_light:
            state = self.get_state(signal)
            observation.append(state)
        terminated = self.is_terminal()
        truncated = False

        return observation, reward, terminated, truncated, info


    def reset(self) -> None:
        traci.load(self.network.sumoCmd[1:])
        if self.network.config['RENDER_MODE'] == "human":
            traci.gui.setSchema("View #0", "real world")
        observation = np.array([0, 0, 0, 0])
        info = {}
        self.simulation_step = 0


        return observation, info

    def render(self) -> None:
        """
        This function has no influence, sumo does it
        """
        if self.config["RENDER_MODE"] == "human":
            self.render_mode = "sumo-gui"
        elif self.config["RENDER_MODE"] == None:
            self.render_mode = "sumo"

    def log_values(self):
        waiting_time = []
        speed = []
        co2 = []
        nox = []
        halting_vehicles = []


        for lane in self.network.instance.lanes:
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
