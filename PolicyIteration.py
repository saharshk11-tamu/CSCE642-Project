from AbstractSolver import AbstractSolver

class PolicyIteration(AbstractSolver):

    def __init__(self, env, epsilon = 0.1, gamma = 0.1, num_episodes = 100, max_steps = 100):
        super().__init__(env, epsilon, gamma, num_episodes, max_steps)
        