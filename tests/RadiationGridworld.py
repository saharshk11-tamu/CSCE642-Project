import numpy as np
import gymnasium as gym

class RadiationGridworld(gym.Env):

    def __init__(
            self, size, agent_loc: list[int, int],
            target_loc: list[int, int],
            wall_locs: list[list[int, int]],
            radiation_locs: list[list[int, int]],
            radiation_consts: list[list[int, int]],
            step_limit: int = 100,
            transition_prob: float = 0.9
        ):
        self.size = size

        self._agent_start_location = np.array(agent_loc, dtype=np.int32)
        
        self._agent_location = self._agent_start_location
        self._target_location = np.array(target_loc, dtype=np.int32)
        self._wall_locations = np.array(wall_locs)
        self._radiation_locations = np.array(radiation_locs, dtype=np.int32)
        self._radiation_consts = np.array(radiation_consts)
        self._radiation_dose = self._calc_rad_dose()
        
        self.observation_space = gym.spaces.Dict(
            {
                'agent': gym.spaces.Box(low=0, high=size, shape=(2,), dtype=int),
                'target': gym.spaces.Box(low=0, high=size, shape=(2,), dtype=int)
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1])
        }

        self.step_limit = step_limit
        self._curr_steps = 0
        self._transition_prob = transition_prob
    
    def _calc_rad_dose(self):
        dose = 0
        for i, rad_loc in enumerate(self._radiation_locations):
            gamma, activity = self._radiation_consts[i]
            dose += gamma*activity/(np.linalg.norm(self._agent_location - rad_loc)**2)
        
        return dose

    def _get_obs(self):
        return {'agent': self._agent_location, 'target': self._target_location, 'radiation_dose': self._radiation_dose}
    
    def _get_info(self):
        return {
            'distance': {
                'manhattan': np.linalg.norm(self._agent_location - self._target_location, ord=1),
                'euclidean': np.linalg.norm(self._agent_location - self._target_location)
            },
            'step': self._curr_steps,
            'radiation_dose': self._radiation_dose
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

        # end episode if step limit is reached
        self._curr_steps += 1
        truncated = self._curr_steps == self.step_limit

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

        return observation, reward, terminated, truncated, info
    


