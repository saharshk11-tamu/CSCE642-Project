from abc import ABC, abstractmethod
from typing import Callable
from RadiationGridworld import RadiationGridworld
import numpy as np


class AbstractSolver(ABC):
    '''
    Abstract base class for reinforcement learning solvers in the Radiation Gridworld environment.
    This class defines the common interface and attributes for all solver implementations.
    Subclasses must implement the train_episode and greedy_policy methods.
    '''

    def __init__(
            self,
            env: RadiationGridworld,
            epsilon: float = 0.1,
            gamma: float = 0.1,
            num_episodes: int = 100,
            max_steps: int = 100,
            reward_aggregator: Callable[[np.ndarray], float] | None = None
        ):
        '''
        Initializes the AbstractSolver with the given environment and parameters.
        Parameters:
        - env: RadiationGridworld, the environment in which the agents operate
        - epsilon: float, the exploration probability for epsilon-greedy policies
        - gamma: float, the discount factor for future rewards
        - num_episodes: int, the number of training episodes
        - max_steps: int, the maximum number of steps per episode
        - reward_aggregator: function to aggregate per-agent rewards to a scalar (default: sum)
        '''
        
        self.env = env
        self.num_agents = getattr(env, 'num_agents', 1)
        self.epsilon = epsilon # probability of choosing a random action
        self.gamma = gamma # discount factor
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.total_steps = 0
        self.reward_aggregator = reward_aggregator if reward_aggregator is not None else np.sum
        self.reward = 0.0
    
    @abstractmethod
    def train_episode(self):
        pass

    @abstractmethod
    def create_greedy_policy(self):
        pass

    def step(self, action):
        _, next_state, rewards, done, info = self.env.step(action)
        return next_state, rewards, done, info

    def evaluate_greedy_policy(self, num_episodes: int = 100):
        '''
        Evaluates the solver's greedy policy by running multiple rollouts in the environment.
        Parameters:
        - num_episodes: int, the number of evaluation episodes to average over
        Returns:
        - avg_return: float, the mean aggregated return over the evaluation episodes
        - std_return: float, the standard deviation of aggregated returns over the evaluation episodes
        '''
        policy_fn = self.create_greedy_policy()
        returns = []
        for _ in range(num_episodes):
            total_reward = 0.0
            state, _ = self.env.reset()
            for _ in range(self.max_steps):
                action = policy_fn(state)
                next_state, rewards, done, _ = self.step(action)
                total_reward += float(self.reward_aggregator(np.array(rewards)))
                state = next_state
                if done:
                    break
            returns.append(total_reward)
        returns = np.array(returns, dtype=np.float32)
        return float(returns.mean()), float(returns.std())
