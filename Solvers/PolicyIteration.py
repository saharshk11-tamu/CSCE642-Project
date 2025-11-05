from Solvers.AbstractSolver import AbstractSolver
import numpy as np


def get_random_policy(num_states: int, num_actions: int):
    '''
    Provides a random policy for the given number of states and actions.
    Each state is assigned a random action with equal probability.
    Parameters:
    - num_states: int, the number of states in the environment
    - num_actions: int, the number of possible actions in the environment
    Returns:
    - policy: numpy array of shape (num_states, num_actions), the random policy
    '''
    policy = np.zeros((num_states, num_actions))
    for s_idx in range(num_states):
        action = np.random.choice(num_actions)
        policy[s_idx, action] = 1
    
    return policy


class PolicyIteration(AbstractSolver):
    '''
    Policy Iteration Solver for Reinforcement Learning
    This class implements the Policy Iteration algorithm with a linear solver to find the optimal policy for a given environment.
    It alternates between policy evaluation and policy improvement until convergence.
    Inherits from AbstractSolver.
    '''

    def __init__(self, env, epsilon = 0.1, gamma = 0.1, num_episodes = 100, max_steps = 100):
        '''
        Initializes the Policy Iteration solver with the given environment and parameters.
        Parameters:
        - env: the environment to solve
        - epsilon: float, the exploration rate (not used in Policy Iteration)
        - gamma: float, the discount factor for future rewards
        - num_episodes: int, the number of episodes to train
        - max_steps: int, the maximum number of steps per episode
        '''
        super().__init__(env, epsilon, gamma, num_episodes, max_steps)

        self.V = np.zeros(self.env.observation_space.n)

        self.policy = get_random_policy(self.env.observation_space.n, self.env.action_space.n)

    def train_episode(self):
        '''
        Trains the solver for one episode using Policy Iteration.
        This involves evaluating the current policy and then improving it.
        Updates the value function and policy accordingly.
        '''
        self.policy_eval()

        for s in range(self.env.observation_space.n):
            self.policy[s].fill(0)
            self.policy[s, self.one_step_lookahead(s).argmax()] = 1

        self.reward = np.sum(self.V)
    
    def one_step_lookahead(self, state):
        '''
        Performs a one-step lookahead to calculate the action-values for all actions in a given state.
        Parameters:
        - state: int, the current state
        Returns:
        - A: numpy array of shape (num_actions,), the action-values for each action in the state
        '''
        A = np.zeros(self.env.action_space.n)
        for a in range(self.env.action_space.n):
            for prob, next_state, reward, _ in self.env.get_transition(state, a).values():
                A[a] += prob * (reward + self.gamma * self.V[next_state])
        
        return A

    def policy_eval(self):
        '''
        Evaluates the current policy using a linear solver.
        Updates the value function V based on the current policy.
        '''
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
        '''
        Creates a greedy policy based on the current value function.
        Returns:
        - policy_fn: function, a function that takes a state and returns the best action according to the greedy policy
        '''
        def policy_fn(state):
            return np.argmax(self.policy[state])
        
        return policy_fn
    
    def policy_viz(self):
        '''
        Visualizes the current policy as a grid of action directions.
        Returns:
        - viz: numpy array of shape (grid_size, grid_size), the visual representation of the policy
        '''
        policy = self.create_greedy_policy()
        viz = []
        for s in range(self.env.observation_space.n):
            viz.append(self.env._action_to_direction_string['arrow'][policy(s)])
        
        viz = np.array(viz).reshape((self.env.size, self.env.size))
        viz = np.flip(viz, axis=0)
        return viz