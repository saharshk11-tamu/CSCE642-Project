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
        self.metrics = {}
        target_reached_count = 0
        for _ in tqdm(range(self.solver.num_episodes), disable=not verbose):
            self.solver.train_episode()
            self.log.append(self.solver.reward)
            info = self.env._get_info()
            serialized_info = {
                k: v.tolist() if hasattr(v, 'tolist') else v
                for k, v in info.items()
            }
            reached = True
            for r in serialized_info['reached_target']:
                if not r:
                    reached = False
                    break
            if reached:
                target_reached_count += 1
            self.env_info_log.append(serialized_info)
        
        self.log = np.array(self.log, dtype=np.float32)
        self.metrics['last_100_avg_reward'] = float(np.mean(self.log[-100:]))
        self.metrics['last_100_std_reward'] = float(np.std(self.log[-100:]))
        self.metrics['episode_completion_rate'] = target_reached_count / len(self.log)

        if log_path is not None:
            os.makedirs(log_path, exist_ok=True)
            np.save(os.path.join(log_path, f'training_log.npy'), self.log)
            with open(os.path.join(log_path, f'env_info.json'), 'w') as fp:
                json.dump(self.env_info_log, fp, indent=2)
            with open(os.path.join(log_path, f'metrics.json'), 'w') as fp:
                json.dump(self.metrics, fp, indent=2)
    
    def plot_run(self, log_path='logs/'):
        '''
        Plot and optionally save the reward trace from the last call to run().
        Parameters:
        - log_path: str, directory to save the plot image
        '''
        if len(self.log) == 0:
            print('Call Runner.run() before plotting')
        
        episodes = np.arange(len(self.log))
        plt.plot(episodes, self.log, label='Reward')

        window = max(1, min(50, len(self.log)))
        kernel = np.ones(window) / window
        running_avg = np.convolve(self.log, kernel, mode='valid')
        offset = window - 1
        plt.plot(episodes[offset:], running_avg, label=f'Running Avg (window={window})', linewidth=2)

        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title(f'{self.solver_type} Performance')
        plt.legend()
        os.makedirs(log_path, exist_ok=True)
        plt.savefig(os.path.join(log_path, f'training_plot.png'))
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
        'size': 10,
        'agent_locs': [[0, 0], [9, 9]],
        'target_loc': [5, 5],
        'wall_locs': [[4, 4], [4, 5], [4, 6], [5, 4], [6, 4], [6, 5], [6, 6]],
        'radiation_locs': [[2, 2], [7, 7]],
        'radiation_consts': [[1.0, 1.0], [1.0, 1.0]],
        'distance_multiplier': 0.02,
        'radiation_multiplier': 0.8,
        'target_reward': max(5.0, 0.5 * 10),
        'transition_prob': 0.9,
        'collision_penalty': 0.1,
        'revisit_penalty': 0.2,
        'progress_bonus': 0.2,
        'stasis_penalty': 0.1
    }
    solver_config = {
        'type': 'DQN',
        'params': {
            'num_episodes': 1000,
            'max_steps': 300
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

    # To visualize a greedy rollout:
    from Solvers.visualize_episode import animate_episode
    policy = runner.solver.create_greedy_policy()
    animate_episode(runner.env, policy, out_path=os.path.join(log_path, "episode.mp4"), max_steps=50)

if __name__ == '__main__':
    main()
