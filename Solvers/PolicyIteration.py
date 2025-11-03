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

        self.V = np.zeros(self.env.observation_space.n)

        self.policy = get_random_policy(self.env.observation_space.n, self.env.action_space.n)

    def train_episode(self):
        self.policy_eval()

        for s in range(self.env.observation_space.n):
            self.policy[s].fill(0)
            self.policy[s, self.one_step_lookahead(s).argmax()] = 1

        self.reward = np.sum(self.V)
    
    def one_step_lookahead(self, state):
        A = np.zeros(self.env.action_space.n)
        for a in range(self.env.action_space.n):
            for prob, next_state, reward, _ in self.env.get_transition(state, a).values():
                A[a] += prob * (reward + self.gamma * self.V[next_state])

    def policy_eval(self):
        probs = np.zeros((self.env.observation_space.n, self.env.observation_space.n))
        rewards = np.zeros(self.env.observation_space.n)

        for s in range(self.env.observation_space.n):
            a = self.policy[s].argmax()

            for prob, next_state, reward, _ in self.env.get_transition(s, a).values():
                probs[s, next_state] += prob
                rewards[s] += prob * reward
        
        lhs = np.identity(self.env.observation_space.n) - self.gamma*probs
        res = np.linalg.solve(lhs, rewards)

        for s in range(res.shape[0]):
            self.V[s] = res[s]

    def create_greedy_policy(self):
        def policy_fn(state):
            return np.argmax(self.policy[state])
        
        return policy_fn
        