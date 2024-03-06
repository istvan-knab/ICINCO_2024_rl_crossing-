from collections import deque
from collections import namedtuple
import random as random


class ReplayMemory:
    def __init__(self, BUFFER_SIZE, BATCH_SIZE):

        self.BATCH_SIZE = BATCH_SIZE
        self.memory = deque([], maxlen=BUFFER_SIZE)

    def push(self, *args):
        """
        Add new MDP element to queue
        :param Transition: state, reward, next_state, done
        :return: None
        """
        Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward','done'))
        self.memory.append(Transition(*args))

    def sample(self):
        """
        Gives random samples from the memory
        :return: namedtuple containing states, actions, rewards, next_states
        """
        return random.sample(self.memory, self.BATCH_SIZE)

    def __len__(self):
        return len(self.memory)