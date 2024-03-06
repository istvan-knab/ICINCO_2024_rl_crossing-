import random
import torch
class EpsilonGreedy:
    def __init__(self, config, env):
        self.config = config
        self.env = env

    def epsilon_greedy_selection(self, model: object, state):

        if (random.random() < self.config['EPSILON']) and (self.config['EPSILON'] > self.config['EPSILON_THRESHOLD']):
            "Random exploratory step"
            action = self.env.action_space.sample()

            return action
        else:
            "Greedy exploitation step"
            with torch.no_grad():
                model.eval()
                q_calc = model(state)
                model.train()
                action = int(torch.argmax(q_calc))

                return action

    def epsilon_update(self):
        self.config["EPSILON"] *= self.config["EPSILON_DECAY"]
