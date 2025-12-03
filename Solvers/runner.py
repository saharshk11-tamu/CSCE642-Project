import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from RadiationGridworld import RadiationGridworld
from Solvers.PolicyIteration import PolicyIteration
from Solvers.DQL import DQN
from Solvers.REINFORCE import Reinforce
from Solvers.A2C import A2C


class Runner():
    '''
    Runner class to manage the training of reinforcement learning agents in the Radiation Gridworld environment.
    It initializes the environment and solver based on provided configuration dictionaries, runs the training process, and logs the results.
    '''

    def __init__(self, gridworld_config: dict, solver_config: dict):
        '''
        Initializes the Runner with configuration dictionaries.
        Parameters:
        - gridworld_config: dict, parameters for RadiationGridworld constructor
        - solver_config: dict with keys:
            - 'type': str, one of {'Policy Iteration', 'DQN', 'REINFORCE', 'A2C'}
            - 'params': dict, keyword args for the solver
        '''
        self.env = RadiationGridworld(**gridworld_config)
        self.solver_type = solver_config['type']
        params = solver_config.get('params', {})
        if self.solver_type == 'Policy Iteration':
            self.solver = PolicyIteration(self.env, **params)
        elif self.solver_type == 'DQN':
            self.solver = DQN(self.env, **params)
        elif self.solver_type == 'REINFORCE':
            self.solver = Reinforce(self.env, **params)
        elif self.solver_type == 'A2C':
            self.solver = A2C(self.env, **params)
        else:
            raise ValueError(f'Unknown solver type: {self.solver_type}')
        
        self.log = []

    def run(self, verbose: bool = True, log_path: str = None):
        '''
        Runs the training process for the solver.
        Parameters:
        - verbose: bool, whether to print progress information
        - log_path: str or None, the path to save logs and visualizations; if None, no logs are saved
        
        Logs rewards and environment info per episode when a log_path is provided.
        '''
        if verbose:
            print(f'Training {self.solver_type} Solver for {self.solver.num_episodes} episodes...')
        
        self.log = []
        self.env_info_log = []
        for episode in tqdm(range(self.solver.num_episodes), disable=not verbose):
            self.solver.train_episode()
            self.log.append(self.solver.reward)
            info = self.env._get_info()
            serialized_info = {
                k: v.tolist() if hasattr(v, 'tolist') else v
                for k, v in info.items()
            }
            self.env_info_log.append(serialized_info)

        if log_path is not None:
            os.makedirs(log_path, exist_ok=True)
            np.save(os.path.join(log_path, f'{self.solver_type}_training_log.npy'), np.array(self.log, dtype=np.float32))
            with open(os.path.join(log_path, f'{self.solver_type}_env_info.json'), 'w') as fp:
                json.dump(self.env_info_log, fp, indent=2)
    
    def plot_run(self, log_path='logs/'):
        '''
        Plot and optionally save the reward trace from the last call to run().
        Parameters:
        - log_path: str, directory to save the plot image
        '''
        if len(self.log) == 0:
            print('Call Runner.run() before plotting')
        

        plt.plot(np.arange(len(self.log)), self.log)
        plt.xlabel('Epochs')
        plt.ylabel('Reward')
        plt.title(f'{self.solver_type} Performance')
        os.makedirs(log_path, exist_ok=True)
        plt.savefig(os.path.join(log_path, f'{self.solver_type}_training_plot.png'))
        plt.close()
    
    def get_policy(self):
        '''
        Returns the learned policy from the solver.
        Returns:
        - policy: numpy array, the learned policy
        - reward: float, the total reward obtained by following the policy
        '''
        final_policy = []
        policy = self.solver.create_greedy_policy()
        obs, _ = self.env.reset()
        total_reward = 0
        for _ in range(self.solver.max_steps):
            actions = policy(obs)
            final_policy.append([
                self.env._action_to_direction_string['arrow'][a] for a in actions
            ])
            _, next_obs, rewards, done, _ = self.env.step(actions)
            total_reward += float(self.solver.reward_aggregator(np.array(rewards)))
            obs = next_obs
            if done:
                break
        return np.array(final_policy), total_reward


def main():
    '''
    Example usage of the Runner class to train a solver in the Radiation Gridworld environment.
    '''
    gridworld_config = {
        'size': 5,
        'agent_locs': [[0, 0], [4, 4]],
        'target_loc': [2, 2],
        'wall_locs': [],
        'radiation_locs': [[1, 1]],
        'radiation_consts': [[1.0, 1.0]],
        'distance_multiplier': 0.1,
        'radiation_multiplier': 0.1,
        'target_reward': 1.0,
        'transition_prob': 1.0,
        'collision_penalty': 0.05
    }
    solver_config = {
        'type': 'DQN',
        'params': {
            'num_episodes': 100,
            'max_steps': 50
        }
    }

    runner = Runner(gridworld_config, solver_config)
    log_path = 'logs/test/'
    os.makedirs(log_path, exist_ok=True)

    runner.run(verbose=True, log_path=log_path)

    print(f'Trained Solver: {runner.solver_type}')
    avg, std = runner.solver.evaluate_greedy_policy(num_episodes=10)
    print(f'Average Reward: {avg}')
    print(f'Standard Deviation of Reward: {std}')
    runner.plot_run(log_path)

if __name__ == '__main__':
    main()
