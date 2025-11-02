from AbstractSolver import AbstractSolver
import numpy as np


def get_random_policy(num_states: int, num_actions: int):
    policy = np.zeros((num_states, num_actions))
    for s_idx in range(num_states):
        action = np.random.choice(num_actions)
        policy[s_idx, action] = 1
    
    return policy


class PolicyIteration(AbstractSolver):

    def __init__(self, env, epsilon = 0.1, gamma = 0.1, num_episodes = 100, max_steps = 100):
        super().__init__(env, epsilon, gamma, num_episodes, max_steps)

        self.V = np.zeros(env.observation_space.shape)

        self.policy = get_random_policy(self.env.size, self.env.action_space.n)

    def train_episode(self):
        return super().train_episode()
    
    def one_step_lookahead(self, state):
        pass

    def policy_eval(self):
        pass

    def create_greedy_policy(self):
        pass
        