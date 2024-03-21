import numpy as np
class Memory:
    def __init__(self, batch_size):
        self.states = []
        self.probabilities = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batch(self):
        number_of_states = len(self.states)
        batch_start = np.arange(0, number_of_states, self.batch_size, dtype=np.int32)
        indices = np.arange(number_of_states, dtype=np.int32)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.probabilities),\
            np.array(self.values), np.array(self.rewards), np.array(self.dones), batches

    def push(self, state, action, probability, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probabilities.append(probability)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.probabilities = []
        self.values = []
        self.rewards = []
        self.dones = []