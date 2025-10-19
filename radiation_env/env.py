import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .config import *

class RadiationEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.grid_size = GRID_SIZE
        self.goal_pos = np.array(GOAL_POS)
        self.radiation_zones = np.array(RADIATION_ZONES)
        self.max_steps = MAX_STEPS
        self.step_count = 0

        # Define spaces
        # Actions: up, down, left, right
        self.action_space = spaces.Discrete(4)

        # Observations: (x, y, radiation, distance_to_goal)
        low = np.array([0, 0, 0, 0], dtype=np.float32)
        high = np.array([self.grid_size - 1, self.grid_size - 1, 10.0, np.sqrt(2) * self.grid_size])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.agent_pos = None
        self.reset()

    def _radiation_at(self, pos):
        """Compute radiation intensity based on proximity to radiation zones"""
        dist = np.linalg.norm(self.radiation_zones - pos, axis=1)
        return np.sum(np.exp(-dist)) * RADIATION_STRENGTH

    def _noisy_observation(self):
        """Return observation with noise proportional to radiation"""
        radiation = self._radiation_at(self.agent_pos)
        noise = np.random.normal(0, NOISE_STD * radiation, size=2)
        noisy_pos = np.clip(self.agent_pos + noise, 0, self.grid_size - 1)
        distance_to_goal = np.linalg.norm(self.goal_pos - self.agent_pos)
        return np.array([*noisy_pos, radiation, distance_to_goal], dtype=np.float32)

    def step(self, action):
        # Move agent
        if action == 0:   # up
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 1: # down
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 2: # left
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 3: # right
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)

        self.step_count += 1
        radiation = self._radiation_at(self.agent_pos)

        # Reward: encourage goal proximity, penalize radiation
        distance_to_goal = np.linalg.norm(self.goal_pos - self.agent_pos)
        reward = -0.1 * radiation - 0.05 * distance_to_goal

        # Bonus for reaching goal
        terminated = np.array_equal(np.round(self.agent_pos), self.goal_pos)
        if terminated:
            reward += 10.0

        truncated = self.step_count >= self.max_steps
        obs = self._noisy_observation()
        return obs, reward, terminated, truncated

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0.0, 0.0])
        self.step_count = 0
        return self._noisy_observation(), {}

    def compute_optimal_policy(self):
        pass
    
    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = '.'
        for rz in self.radiation_zones:
            grid[int(rz[1]), int(rz[0])] = 'R'
        gx, gy = map(int, self.goal_pos)
        grid[gy, gx] = 'G'
        ax, ay = map(int, self.agent_pos)
        grid[ay, ax] = 'A'
        print("\nGrid\n" + "\n".join([" ".join(row) for row in grid[::-1]]))
