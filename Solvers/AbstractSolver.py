from abc import ABC, abstractmethod
from RadiationGridworld import RadiationGridworld

class AbstractSolver():
    '''
    Abstract base class for reinforcement learning solvers in the Radiation Gridworld environment.
    This class defines the common interface and attributes for all solver implementations.
    Subclasses must implement the train_episode and greedy_policy methods.
    '''

    def __init__(self, env: RadiationGridworld, epsilon: float = 0.1, gamma: float = 0.1, num_episodes: int = 100, max_steps: int = 100):
        '''
        Initializes the AbstractSolver with the given environment and parameters.
        Parameters:
        - env: RadiationGridworld, the environment in which the agent operates
        - epsilon: float, the exploration probability for epsilon-greedy policies
        - gamma: float, the discount factor for future rewards
        - num_episodes: int, the number of training episodes
        - max_steps: int, the maximum number of steps per episode
        '''
        
        self.env = env
        self.epsilon = epsilon # probability of choosing a random action
        self.gamma = gamma # discount factor
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.total_steps = 0
        self.reward = 0.0
    
    @abstractmethod
    def train_episode(self):
        pass

    @abstractmethod
    def greedy_policy(self):
        pass