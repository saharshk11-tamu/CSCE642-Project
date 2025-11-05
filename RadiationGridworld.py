import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import json

class RadiationGridworld(gym.Env):
    '''
    Radiation Gridworld Environment
    A gridworld environment where an agent must navigate to a target while avoiding walls and minimizing radiation exposure.
    The environment is represented as a square grid of size (size x size).
    The agent can move in four directions: up, down, left, right.
    The environment contains walls that the agent cannot pass through, radiation sources that increase the agent's radiation dose, and a target location that the agent must reach.
    The agent receives rewards based on its distance to the target and the radiation dose it receives.
    '''

    def location_to_state(self, location: list[int, int] | np.ndarray):
        '''
        Converts a 2D grid location to a 1D state representation.
        Parameters:
        - location: list or numpy array of two ints, the (x, y) location in the grid
        Returns:
        - state: int, the corresponding 1D state representation
        '''
        x, y, = int(location[0]), int(location[1])
        return x + y * self.size

    def state_to_location(self, state: int):
        '''
        Converts a 1D state representation to a 2D grid location.
        Parameters:
        - state: int, the 1D state representation
        Returns:
        - location: numpy array of two ints, the corresponding (x, y) location in the grid
        '''
        x = state % self.size
        y = state // self.size
        return np.array([x, y], dtype=np.int32)

    def __init__(
            self,
            size,
            agent_loc: list[int, int],
            target_loc: list[int, int],
            wall_locs: list[list[int, int]],
            radiation_locs: list[list[int, int]],
            radiation_consts: list[list[int, int]],
            distance_multiplier: float = 0.1,
            radiation_multiplier: float = 0.1,
            target_reward: float = 1.0,
            transition_prob: float = 1.0
        ):
        '''
        Initializes the Radiation Gridworld environment.
        Parameters:
        - size: int, the size of the gridworld (size x size)
        - agent_loc: list of two ints, the starting location of the agent [x, y]
        - target_loc: list of two ints, the location of the target [x, y]
        - wall_locs: list of lists of two ints, the locations of walls [[x1, y1], [x2, y2], ...]
        - radiation_locs: list of lists of two ints, the locations of radiation sources [[x1, y1], [x2, y2], ...]
        - radiation_consts: list of lists of two ints, the radiation constants for each source [[gamma1, activity1], [gamma2, activity2], ...]
        - distance_multiplier: float, the multiplier for distance-based rewards
        - radiation_multiplier: float, the multiplier for radiation-based rewards
        - target_reward: float, the reward for reaching the target
        - transition_prob: float, the probability of taking the intended action
        '''
        self.size = size

        self._agent_start_location = np.array(agent_loc, dtype=np.int32)
        self._agent_location = self._agent_start_location
        self._target_location = np.array(target_loc, dtype=np.int32)
        self._wall_locations = np.array(wall_locs)
        self._radiation_locations = np.array(radiation_locs, dtype=np.int32)

        self._radiation_consts = np.array(radiation_consts, dtype=np.float32)
        self._radiation_vals = self._calc_rad_doses()
        self._radiation_dose = self._radiation_vals[tuple(self._agent_location)]

        self._distance_multiplier = distance_multiplier
        self._radiation_multiplier = radiation_multiplier
        self._target_reward = target_reward
        
        self.observation_space = gym.spaces.Discrete(self.size * self.size)

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]), # right
            1: np.array([0, 1]), # up
            2: np.array([-1, 0]), # left
            3: np.array([0, -1]) # down
        }

        self._action_to_direction_string = {
            'text': {
                0: 'right',
                1: 'up',
                2: 'left',
                3: 'down'
            },
            'arrow': {
                0: '→',
                1: '↑',
                2: '←',
                3: '↓'
            }
        }

        self._transition_prob = transition_prob

    def _calc_rad_doses(self):
        '''
        Calculates the radiation dose at each cell in the gridworld based on the locations and constants of the radiation sources.
        Returns:
        - doses: 2D numpy array of shape (size, size), the radiation dose at each cell
        '''
        doses = np.zeros((self.size, self.size), dtype=np.float32)
        largest = 0.0
        for x in range(self.size):
            for y in range(self.size):
                dose = 0

                wall = False
                for i in self._wall_locations:
                    if np.array_equal([x, y], i):
                        wall = True
                        break
                if not wall:
                    for i, rad_loc in enumerate(self._radiation_locations):
                        gamma, activity = self._radiation_consts[i]
                        if not np.array_equal([x, y], rad_loc):
                            dose += gamma*activity/(np.linalg.norm(np.array([x, y]) - rad_loc)**2)
                        else:
                            dose = np.inf
                            break
                    
                if dose > largest and dose != np.inf:
                    largest = dose

                doses[x, y] = dose
        
        doses /= largest
        doses[doses == np.inf] = 1.0

        return doses

    def _get_obs(self):
        return self.location_to_state(self._agent_location)
    
    def _get_info(self):
        return {
            'distance': np.linalg.norm(self._agent_location - self._target_location)
        }
    
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        self._agent_location = self._agent_start_location
        self._curr_steps = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def get_transition(self, s, a):
        '''
        Gets the transition probabilities, next states, rewards, and termination flags for taking action a in state s.
        Parameters:
        - s: int, the current state
        - a: int, the action to be taken
        Returns:
        - transitions: dict, mapping from action to (probability, next_state, reward, terminated)

        Note: This function should not be used with any model-free RL algorithms.
        '''
        x, y = self.state_to_location(s)
        location = np.array([x, y], dtype=np.int32)
        transitions = {}

        for action in range(self.action_space.n):
            # calculate probability of taking this action
            if action == a:
                prob = self._transition_prob
            else:
                prob = (1 - self._transition_prob) / (self.action_space.n - 1)
            
            # get next state
            direction = self._action_to_direction[action]
            new_location = np.clip(location + direction, 0, self.size-1)

            # don't move if agent tries to move into a wall
            for i in self._wall_locations:
                if np.array_equal(new_location, i):
                    new_location = location
                    break
            
            new_state = self.location_to_state(new_location)

            # check if new state is terminal
            terminated = np.array_equal(new_location, self._target_location)

            # calculate reward
            # positive reward if target is reached
            if terminated:
                reward = self._target_reward
            else:
                # reward based on distance to target
                distance = np.linalg.norm(new_location - self._target_location)
                reward = -self._distance_multiplier*distance
                # reward based on radiation dose received
                rad_dose = self._radiation_vals[tuple(new_location)]
                reward += -self._distance_multiplier*rad_dose
            
            transitions[action] = (prob, new_state, reward, terminated)
        
        return transitions
    
    def step(self, action):
        '''
        Takes a step in the environment using the given action.
        Parameters:
        - action: int, the action to be taken
        Returns:
        - prob: float, the probability of taking the chosen action
        - observation: int, the new state after taking the action
        - reward: float, the reward received after taking the action
        - terminated: bool, whether the episode has terminated
        - info: dict, additional information about the environment
        '''
        transitions = self.get_transition(self.location_to_state(self._agent_location), action)

        probs = [transitions[a][0] for a in transitions.keys()]
        chosen_action = np.random.choice(np.arange(self.action_space.n), p=probs)
        prob, new_state, reward, terminated = transitions[chosen_action]
        new_location = self.state_to_location(new_state)

        self._agent_location = new_location
        observation = self._get_obs()
        info = self._get_info()

        return prob, observation, reward, terminated, info
    

    def render(self, render_mode: str = 'ansi', dir: str = None):
        '''
        Provides a visualization of the current state of the environment.
        Parameters:
        - render_mode: str, the mode of rendering ('ansi' for text, 'radiation_map' for heatmap)
        - dir: str, the directory to save the radiation map image (only used for 'radiation_map' mode)
        '''
        if render_mode == 'ansi':
            for y in range(self.size-1, -1, -1):
                row = ''
                for x in range(self.size):
                    if np.array_equal([x, y], self._agent_location):
                        row += 'A '
                    elif np.array_equal([x, y], self._target_location):
                        row += 'T '
                    else:
                        rad = False
                        for i in self._radiation_locations:
                            if np.array_equal([x, y], i):
                                row += 'R '
                                rad = True
                                break
                        wall = False
                        for i in self._wall_locations:
                            if np.array_equal([x, y], i):
                                row += 'W '
                                wall = True
                                break
                        
                        if not rad and not wall:
                            row += '. '
                print(row)
            print()
        elif render_mode == 'radiation_map':
            data = np.flipud(self._radiation_vals.T)
            plt.title('Gridworld Radiation Map')
            plt.imshow(data, cmap='hot')
            plt.axis('off')
            plt.savefig(dir if dir is not None else 'radiation_map.png')
        else:
            print(f'Invalid render mode: {render_mode}')
    
    def save(self, path: str = 'radiation_gridworld_config.json'):
        '''
        Saves the current configuration of the Radiation Gridworld to a JSON file.
        Parameters:
        - path: str, the file path to save the configuration
        '''
        config = {
            'size': self.size,
            'agent_start_location': self._agent_start_location,
            'agent_location': self._agent_location,
            'target_location': self._target_location,
            'wall_locations': self._wall_locations,
            'radiation_locations': self._radiation_locations,
            'radiation_consts': self._radiation_consts,
            'radiation_vals': self._radiation_vals,
            'radiation_dose': self._radiation_dose,
            'transition_prob': self._transition_prob
        }

        with open(path, 'w') as file:
            json.dump(config, file, indent=4)
    
    def load(self, path: str = 'radiation_gridworld_config.json'):
        '''
        Loads the configuration of the Radiation Gridworld from a JSON file.
        Parameters:
        - path: str, the file path to load the configuration from
        '''
        with open(path, 'r') as file:
            config = json.load(file)
        
        self.size = config['size']
        self._agent_start_location = np.array(config['agent_start_location'], dtype=np.int32)
        self._agent_location = np.array(config['agent_location'], dtype=np.int32)
        self._target_location = np.array(config['target_location'], dtype=np.int32)
        self._wall_locations = np.array(config['wall_locations'], dtype=np.int32)
        self._radiation_locations = np.array(config['radiation_locations'], dtype=np.int32)
        self._radiation_consts = np.array(config['radiation_consts'], dtype=np.float32)
        self._radiation_vals = np.array(config['radiation_vals'], dtype=np.float32)
        self._radiation_dose = config['radiation_dose']
        self._transition_prob = config['transition_prob']
