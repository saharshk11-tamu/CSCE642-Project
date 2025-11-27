import json
from RadiationGridworld import RadiationGridworld
from Solvers.PolicyIteration import PolicyIteration
from Solvers.DQL import DQN
from Solvers.REINFORCE import Reinforce
from Solvers.A2C import A2C
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

class Runner():
    '''
    Runner class to manage the training of reinforcement learning agents in the Radiation Gridworld environment.
    It initializes the environment and solver based on a configuration file, runs the training process, and logs the results.
    '''

    def __init__(self, config_path: str):
        '''
        Initializes the Runner with the specified configuration file.
        Parameters:
        - config_path: str, the path to the JSON configuration file
        '''
        with open(config_path, 'r') as file:
            self.config = json.load(file)

        self.env = RadiationGridworld(**self.config['gridworld'])
        self.solver_type = self.config['solver']['type']
        if self.solver_type == 'Policy Iteration':
            self.solver = PolicyIteration(self.env, **self.config['solver']['params'])
        elif self.solver_type == 'DQN':
            self.solver = DQN(self.env, **self.config['solver']['params'])
        elif self.solver_type == 'REINFORCE':
            self.solver = Reinforce(self.env, **self.config['solver']['params'])
        elif self.solver_type == 'A2C':
            self.solver = A2C(self.env, **self.config['solver']['params'])
        
        self.log = []

    def run(self, verbose: bool = True, log_path: str = None):
        '''
        Runs the training process for the solver.
        Parameters:
        - verbose: bool, whether to print progress information
        - log_path: str or None, the path to save logs and visualizations; if None, no logs are saved
        '''
        if verbose:
            print(f'Training {self.solver_type} Solver for {self.solver.num_episodes} episodes...')
        
        self.log = []
        for episode in tqdm(range(self.solver.num_episodes), disable=not verbose):
            self.solver.train_episode()
            self.log.append(self.solver.reward)

        if log_path is not None:
            self.env.render(render_mode='radiation_map', dir=log_path+'radiation_map.png')
            with open(log_path+f'{self.solver_type}_training_log.json', 'w') as file:
                json.dump(self.log, file, indent=4)
    
    def plot_run(self, log_path='logs/'):
        if len(self.log) == 0:
            print('Call Runner.run() before plotting')
        
        plt.plot(self.log)
        plt.xlabel('Epochs')
        plt.ylabel('Reward')
        plt.title(f'{self.solver_type} Performance')
        plt.savefig(log_path+f'{self.solver_type}_training_plot.png')
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
        state, _ = self.env.reset()
        total_reward = 0
        for _ in range(self.solver.max_steps):
            action = policy(state)
            final_policy.append(self.env._action_to_direction_string['arrow'][action])
            _, next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break
        return np.array(final_policy), total_reward


def main():
    '''
    Example usage of the Runner class to train a Policy Iteration solver in the Radiation Gridworld environment.
    '''
    runner = Runner('config.json')
    log_path = 'logs/test/'
    os.makedirs(log_path, exist_ok=True)

    runner.run(verbose=True, log_path=log_path)

    print(f'Trained Solver: {runner.solver_type}')
    avg, std = runner.solver.evaluate_greedy_policy(num_episodes=100)
    print(f'Average Reward: {avg}')
    print(f'Standard Deviation of Reward: {std}')
    runner.plot_run(log_path)

    # '''
    # Example usage of the Runner class to train a DQN solver in the Radiation Gridworld environment.
    # '''
    # runner = Runner('dqn_config.json')
    # log_path = 'logs/test/'
    # os.makedirs(log_path, exist_ok=True)

    # runner.run(verbose=True, log_path=log_path)

    # print(f'Trained Solver: {runner.solver_type}')
    # avg, std = runner.solver.evaluate_greedy_policy(num_episodes=100)
    # print(f'Average Reward: {avg}')
    # print(f'Standard Deviation of Reward: {std}')

    # '''
    # Example usage of the Runner class to train a REINFORCE solver in the Radiation Gridworld environment.
    # '''
    # runner = Runner('reinforce_config.json')
    # log_path = 'logs/test/'
    # os.makedirs(log_path, exist_ok=True)

    # runner.run(verbose=True, log_path=log_path)

    # print(f'Trained Solver: {runner.solver_type}')
    # avg, std = runner.solver.evaluate_greedy_policy(num_episodes=100)
    # print(f'Average Reward: {avg}')
    # print(f'Standard Deviation of Reward: {std}')

    # '''
    # Example usage of the Runner class to train a A2C solver in the Radiation Gridworld environment.
    # '''
    # runner = Runner('a2c_config.json')
    # log_path = 'logs/test/'
    # os.makedirs(log_path, exist_ok=True)

    # runner.run(verbose=True, log_path=log_path)

    # print(f'Trained Solver: {runner.solver_type}')
    # avg, std = runner.solver.evaluate_greedy_policy(num_episodes=100)
    # print(f'Average Reward: {avg}')
    # print(f'Standard Deviation of Reward: {std}')

if __name__ == '__main__':
    main()
