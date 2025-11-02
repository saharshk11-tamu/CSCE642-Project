import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import json

class RadiationGridworld(gym.Env):

    def __init__(
            self,
            size,
            agent_loc: list[int, int],
            target_loc: list[int, int],
            wall_locs: list[list[int, int]],
            radiation_locs: list[list[int, int]],
            radiation_consts: list[list[int, int]],
            render_mode: str = 'ansi'
        ):
        self.size = size

        self._agent_start_location = np.array(agent_loc, dtype=np.int32)
        self._agent_location = self._agent_start_location
        self._target_location = np.array(target_loc, dtype=np.int32)
        self._wall_locations = np.array(wall_locs)
        self._radiation_locations = np.array(radiation_locs, dtype=np.int32)

        self._radiation_consts = np.array(radiation_consts)
        self._radiation_vals = self._calc_rad_doses()
        self._radiation_dose = self._radiation_vals[tuple(self._agent_location)]
        
        # represent agent position as a single discrete state: index = x + y*size
        self.observation_space = gym.spaces.Discrete(self.size * self.size)

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1])
        }

        self.render_mode = render_mode

    def _calc_rad_doses(self):
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
                        dose += gamma*activity/(np.linalg.norm(np.array([x, y]) - rad_loc)**2)
                    
                if dose > largest and dose != np.inf:
                    largest = dose

                doses[x, y] = dose
        
        doses /= largest
        doses[doses == np.inf] = 1.0

        return doses

    def _get_obs(self):
        x, y = int(self._agent_location[0]), int(self._agent_location[1])
        return {'agent_location': x + y * self.size}
    
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
    
    def step(self, action):
        direction = self._action_to_direction[action]
        
        # update agent location; don't move if action tries to go out of grid
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size)
        # don't move if agent tries to move into a wall
        for i in self._wall_locations:
            if np.array_equal(self._agent_location, i):
                self._agent_location -= direction
                break
        
        # check if target is reached
        terminated = np.array_equal(self._agent_location, self._target_location)

        # calculate reward
        # positive reward if target is reached
        if terminated:
            reward = 1
        else:
            # reward based on distance to target
            distance = np.linalg.norm(self._agent_location - self._target_location) # distance to target
            reward = -0.01*distance
            # reward based on radiation
            self._radiation_dose = self._calc_rad_dose()
            reward += -0.01*self._radiation_dose

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, info
    

    def render(self, dir: str = None):
        if self.render_mode == 'ansi':
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
        elif self.render_mode == 'radiation_map':
            data = np.flip(self._radiation_vals, axis=0)
            print(data)
            plt.title('Gridworld Radiation Map')
            plt.imshow(data, cmap='hot')
            plt.axis('off')
            plt.savefig(dir if dir is not None else 'radiation_map.png')
    
    def save(self, path: str = 'radiation_gridworld.json'):
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
            'render_mode': self.render_mode
        }

        with open(path, 'w') as file:
            json.dump(config, file, indent=4)
