import json
from RadiationGridworld import RadiationGridworld
from Solvers.PolicyIteration import PolicyIteration
from tqdm import tqdm
import os

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
            with open(log_path+'training_log.json', 'w') as file:
                json.dump(self.log, file, indent=4)

def main():
    '''
    Example usage of the Runner class to train a solver in the Radiation Gridworld environment.
    '''
    runner = Runner('config.json')
    log_path = 'logs/test/'
    os.makedirs(log_path, exist_ok=True)

    runner.run(verbose=True, log_path=log_path)

    print(f'Trained Solver: {runner.solver_type}')
    print(f'Final Policy Visualization:\n{runner.solver.policy_viz()}')

if __name__ == '__main__':
    main()