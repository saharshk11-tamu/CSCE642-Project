from abc import ABC, abstractmethod
from RadiationGridworld import RadiationGridworld

class AbstractSolver():

    def __init__(self, env: RadiationGridworld, epsilon: float = 0.1, gamma: float = 0.1, num_episodes: int = 100, max_steps: int = 100):
        self.env = env
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.max_steps = max_steps
    
    @abstractmethod
    def train_episode(self):
        pass

    @abstractmethod
    def greedy_policy(self):
        pass

    @abstractmethod
    def epsilon_greedy_policy(self):
        pass