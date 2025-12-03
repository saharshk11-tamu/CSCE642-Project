import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import json

class RadiationGridworld(gym.Env):
    '''
    Radiation Gridworld Environment (multi-agent)
    A gridworld environment where one or more agents navigate to a single target while avoiding walls and minimizing radiation exposure.
    The environment is represented as a square grid of size (size x size).
    Each agent can move in four directions: up, down, left, right.
    The environment contains walls that agents cannot pass through, radiation sources that increase the agent's radiation dose, and a target location that agents must reach.
    Agents receive rewards based on a mix of individual progress and group progress, and an episode terminates when every agent reaches the target.
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
            size: int,
            agent_locs: list[list[int, int]],
            target_loc: list[int, int],
            wall_locs: list[list[int, int]],
            radiation_locs: list[list[int, int]],
            radiation_consts: list[list[int, int]],
            distance_multiplier: float = 0.1,
            radiation_multiplier: float = 0.1,
            target_reward: float = 1.0,
            transition_prob: float = 1.0,
            collision_penalty: float = 0.05
        ):
        '''
        Initializes the Radiation Gridworld environment.
        Parameters:
        - size: int, the size of the gridworld (size x size)
        - agent_locs: list of two ints per agent, the starting locations [[x1, y1], [x2, y2], ...]
        - target_loc: list of two ints, the location of the single target [x, y]
        - wall_locs: list of lists of two ints, the locations of walls [[x1, y1], [x2, y2], ...]
        - radiation_locs: list of lists of two ints, the locations of radiation sources [[x1, y1], [x2, y2], ...]
        - radiation_consts: list of lists of two ints, the radiation constants for each source [[gamma1, activity1], [gamma2, activity2], ...]
        - distance_multiplier: float, the multiplier for distance-based rewards
        - radiation_multiplier: float, the multiplier for radiation-based rewards
        - target_reward: float, the reward for reaching the target
        - transition_prob: float, the probability of taking the intended action
        - collision_penalty: float, the penalty for attempted moves into walls or out-of-bounds
        '''
        self.size = size
        self.num_agents = len(agent_locs)

        self._agent_start_locations = np.array(agent_locs, dtype=np.int32)
        self._agent_locations = np.array(agent_locs, dtype=np.int32)
        self._target_location = np.array(target_loc, dtype=np.int32)
        self._wall_locations = np.array(wall_locs)
        self._radiation_locations = np.array(radiation_locs, dtype=np.int32)

        self._radiation_consts = np.array(radiation_consts, dtype=np.float32)
        self._radiation_vals = self._calc_rad_doses()
        self._radiation_doses = np.array(
            [self._radiation_vals[tuple(loc)] for loc in self._agent_locations],
            dtype=np.float32
        )
        self._cumulative_doses = np.array(self._radiation_doses, dtype=np.float32)

        self._distance_multiplier = distance_multiplier
        self._radiation_multiplier = radiation_multiplier
        self._target_reward = target_reward
        self._collision_penalty = collision_penalty

        self.observation_space = gym.spaces.Dict({
            'positions': gym.spaces.Box(
                low=0,
                high=self.size - 1,
                shape=(self.num_agents, 2),
                dtype=np.int32
            ),
            'doses': gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_agents,),
                dtype=np.float32
            )
        })

        self.action_space = gym.spaces.MultiDiscrete([4] * self.num_agents)

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
        self._reached_target = np.zeros(self.num_agents, dtype=bool)
        self._curr_steps = 0

    def _calc_rad_doses(self):
        '''
        Calculate normalized radiation dose at each cell based on source locations and constants.
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
        '''
        Build the current observation dict for all agents.
        Returns:
        - dict with positions and doses arrays
        '''
        return {
            'positions': np.array(self._agent_locations, dtype=np.int32),
            'doses': np.array([self._radiation_vals[tuple(loc)] for loc in self._agent_locations], dtype=np.float32)
        }
    
    def _get_info(self):
        '''
        Build auxiliary info about the current episode state.
        Returns:
        - dict of step count, distances, current and cumulative doses, and reached flags
        '''
        distances = np.linalg.norm(self._agent_locations - self._target_location, axis=1)
        return {
            'curr_step': self._curr_steps,
            'distances': distances,
            'current_doses': np.array([self._radiation_vals[tuple(loc)] for loc in self._agent_locations], dtype=np.float32),
            'cumulative_doses': np.array(self._cumulative_doses, dtype=np.float32),
            'reached_target': np.array(self._reached_target, dtype=bool)
        }
    
    def reset(self, *, seed=None, options=None):
        '''
        Reset environment to initial agent positions and recompute radiation/dose bookkeeping.
        Returns:
        - observation dict
        - info dict
        '''
        super().reset(seed=seed)

        self._agent_locations = np.array(self._agent_start_locations, dtype=np.int32)
        self._curr_steps = 0
        self._radiation_doses = np.array(
            [self._radiation_vals[tuple(loc)] for loc in self._agent_locations],
            dtype=np.float32
        )
        self._cumulative_doses = np.array(self._radiation_doses, dtype=np.float32)
        self._reached_target = np.zeros(self.num_agents, dtype=bool)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def get_transition(self, positions, actions):
        '''
        Get transition probabilities, next positions, rewards, and termination flags for a batch of actions.
        Parameters:
        - positions: np.ndarray of shape (num_agents, 2), current positions
        - actions: iterable of length num_agents, intended actions for each agent
        Returns:
        - transitions: list of dicts, one per agent, mapping action -> (probability, next_position, reward, reached_target)

        Note: This function should not be used with any model-free RL algorithms.
        '''
        positions = np.array(positions, dtype=np.int32)
        actions = np.array(actions, dtype=np.int32)
        transitions = []

        for agent_idx in range(self.num_agents):
            location = positions[agent_idx]
            intended_action = actions[agent_idx]
            agent_transitions = {}

            for action in range(4):
                if action == intended_action:
                    prob = self._transition_prob
                else:
                    prob = (1 - self._transition_prob) / 3

                direction = self._action_to_direction[action]
                proposed = location + direction
                new_location = np.clip(proposed, 0, self.size - 1)

                for i in self._wall_locations:
                    if np.array_equal(new_location, i):
                        new_location = location
                        break

                reached = np.array_equal(new_location, self._target_location)

                if reached:
                    reward = self._target_reward
                else:
                    distance = np.linalg.norm(new_location - self._target_location)
                    reward = -self._distance_multiplier * distance
                    rad_dose = self._radiation_vals[tuple(new_location)]
                    prospective_cum = self._cumulative_doses[agent_idx] + rad_dose
                    reward += -self._radiation_multiplier * prospective_cum
                    if np.array_equal(new_location, location):
                        reward -= self._collision_penalty

                agent_transitions[action] = (prob, new_location, reward, reached)

            transitions.append(agent_transitions)

        return transitions
    
    def step(self, actions):
        '''
        Execute a joint action for all agents and advance the environment state.
        Parameters:
        - actions: iterable of length num_agents, the actions to be taken
        Returns:
        - probs: np.ndarray of shape (num_agents,), probability of each chosen action
        - observation: dict, containing positions and current doses for all agents
        - rewards: np.ndarray of shape (num_agents,), reward received by each agent
        - terminated: bool, whether all agents have reached the target
        - info: dict, additional information about the environment
        '''
        actions = np.array(actions, dtype=np.int32)
        if actions.shape != (self.num_agents,):
            raise ValueError(f'actions must have shape ({self.num_agents},), got {actions.shape}')

        transitions = self.get_transition(self._agent_locations, actions)

        chosen_probs = np.zeros(self.num_agents, dtype=np.float32)
        individual_rewards = np.zeros(self.num_agents, dtype=np.float32)

        for agent_idx, agent_transitions in enumerate(transitions):
            if np.isclose(self._transition_prob, 1.0):
                chosen_action = actions[agent_idx]
            else:
                action_ids = np.array(list(agent_transitions.keys()))
                probs = np.array([agent_transitions[a][0] for a in action_ids], dtype=np.float64)
                probs /= probs.sum()
                chosen_action = np.random.choice(action_ids, p=probs)

            prob, new_location, reward, reached = agent_transitions[chosen_action]
            rad_dose = self._radiation_vals[tuple(new_location)]

            self._agent_locations[agent_idx] = new_location
            self._radiation_doses[agent_idx] = rad_dose
            self._cumulative_doses[agent_idx] += rad_dose
            self._reached_target[agent_idx] = self._reached_target[agent_idx] or reached

            chosen_probs[agent_idx] = prob
            individual_rewards[agent_idx] = reward

        distances = np.linalg.norm(self._agent_locations - self._target_location, axis=1)
        group_reward = -self._distance_multiplier * np.mean(distances)
        group_reward += -self._radiation_multiplier * np.mean(self._cumulative_doses)
        if np.all(self._reached_target):
            group_reward += self._target_reward

        rewards = individual_rewards + group_reward

        terminated = bool(np.all(self._reached_target))
        self._curr_steps += 1
        observation = self._get_obs()
        info = self._get_info()

        return chosen_probs, observation, rewards, terminated, info
    

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
                    cell = np.array([x, y])
                    agent_idxs = [idx for idx, loc in enumerate(self._agent_locations) if np.array_equal(cell, loc)]
                    if agent_idxs:
                        row += f'A{agent_idxs[0]} '
                    elif np.array_equal(cell, self._target_location):
                        row += 'T '
                    else:
                        rad = False
                        for i in self._radiation_locations:
                            if np.array_equal(cell, i):
                                row += 'R '
                                rad = True
                                break
                        wall = False
                        for i in self._wall_locations:
                            if np.array_equal(cell, i):
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
            plt.close()
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
            'num_agents': self.num_agents,
            'agent_start_locations': self._agent_start_locations.tolist(),
            'agent_locations': self._agent_locations.tolist(),
            'target_location': self._target_location.tolist(),
            'wall_locations': self._wall_locations.tolist(),
            'radiation_locations': self._radiation_locations.tolist(),
            'radiation_consts': self._radiation_consts.tolist(),
            'radiation_vals': self._radiation_vals.tolist(),
            'radiation_doses': self._radiation_doses.tolist(),
            'cumulative_doses': self._cumulative_doses.tolist(),
            'reached_target': self._reached_target.tolist(),
            'transition_prob': self._transition_prob,
            'collision_penalty': self._collision_penalty
        }

        with open(path, 'w') as file:
            json.dump(config, file, indent=4)
    
    def load(self, path: str = 'radiation_gridworld_config.json'):
        '''
        Load a configuration from a JSON file and reset environment state accordingly.
        Parameters:
        - path: str, the file path to load the configuration from
        '''
        with open(path, 'r') as file:
            config = json.load(file)
        
        self.size = config['size']
        self.num_agents = config.get('num_agents', len(config['agent_start_locations']))
        self._agent_start_locations = np.array(config['agent_start_locations'], dtype=np.int32)
        self._agent_locations = np.array(config['agent_locations'], dtype=np.int32)
        self._target_location = np.array(config['target_location'], dtype=np.int32)
        self._wall_locations = np.array(config['wall_locations'], dtype=np.int32)
        self._radiation_locations = np.array(config['radiation_locations'], dtype=np.int32)
        self._radiation_consts = np.array(config['radiation_consts'], dtype=np.float32)
        self._radiation_vals = np.array(config.get('radiation_vals', self._calc_rad_doses()), dtype=np.float32)
        self._radiation_doses = np.array(config.get('radiation_doses', [self._radiation_vals[tuple(loc)] for loc in self._agent_locations]), dtype=np.float32)
        self._cumulative_doses = np.array(config.get('cumulative_doses', self._radiation_doses.tolist()), dtype=np.float32)
        self._reached_target = np.array(config.get('reached_target', [False] * self.num_agents), dtype=bool)
        self._transition_prob = config['transition_prob']
        self._collision_penalty = config['collision_penalty']

        self.observation_space = gym.spaces.Dict({
            'positions': gym.spaces.Box(
                low=0,
                high=self.size - 1,
                shape=(self.num_agents, 2),
                dtype=np.int32
            ),
            'doses': gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_agents,),
                dtype=np.float32
            )
        })

        self.action_space = gym.spaces.MultiDiscrete([4] * self.num_agents)


    def set_random(self, seed=0):
        '''
        Randomly reset agent starting locations (avoiding walls and the target) and reset the environment.
        '''
        rng = np.random.default_rng(seed)
        free_cells = []
        for x in range(self.size):
            for y in range(self.size):
                cell = np.array([x, y], dtype=np.int32)
                if any(np.array_equal(cell, w) for w in self._wall_locations):
                    continue
                if np.array_equal(cell, self._target_location):
                    continue
                free_cells.append(cell)

        if len(free_cells) < self.num_agents:
            raise ValueError('Not enough free cells to place all agents.')

        rng.shuffle(free_cells)
        self._agent_start_locations = np.array(free_cells[:self.num_agents], dtype=np.int32)
        return self.reset()
