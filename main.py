from argparse import ArgumentParser
import json
import os
import random
from itertools import product
import math

import numpy as np
import torch
from tqdm import tqdm

from Solvers.runner import Runner

WALL_DENSITIES = {
    'low': 0.05,
    'medium': 0.1,
    'high': 0.2,
}
RADIATION_SOURCE_DENSITIES = {
    'low': 0.0005,
    'medium': 0.001,
    'high': 0.002,
}
RADIATION_STRENGTH_RANGES = {
    'low': (0.2, 0.8),
    'medium': (0.8, 1.5),
    'high': (1.5, 3.0),
}
SOLVER_TYPES = ('DQN', 'A2C', 'REINFORCE')
TARGET_REWARD_PER_UNIT = 10.0


def sample_positions(rng: np.random.Generator, count: int, size: int, forbidden: set[tuple[int, int]]):
    """Sample unique grid positions while avoiding forbidden coordinates."""
    available = [(x, y) for x in range(size) for y in range(size) if (x, y) not in forbidden]
    if count > len(available):
        count = len(available)
    if count == 0:
        return []
    idx = rng.choice(len(available), size=count, replace=False)
    return [list(available[i]) for i in idx]


def build_grid_config(size: int, num_agents: int, wall_level: str, rad_level: str, strength_level: str, rng: np.random.Generator):
    """Create a randomized gridworld configuration with controllable density and strength ranges."""
    target_loc = [size - 1, size - 1]
    forbidden = set()
    # Avoid target unless the grid is 1x1; otherwise allow the only available cell.
    if size > 1:
        forbidden.add(tuple(target_loc))

    agent_locs = sample_positions(rng, num_agents, size, forbidden)
    if len(agent_locs) < num_agents and size == 1:
        agent_locs = [target_loc]
    forbidden.update({(loc[0], loc[1]) for loc in agent_locs})

    desired_walls = max(0, int(size * size * WALL_DENSITIES[wall_level]))
    desired_rad_sources = max(1, int(size * size * RADIATION_SOURCE_DENSITIES[rad_level]))

    max_cells = size * size
    available_cells = max_cells - len(forbidden)
    desired_rad_sources = min(desired_rad_sources, max(1, available_cells - desired_walls))
    desired_walls = min(desired_walls, max(0, available_cells - desired_rad_sources))

    wall_locs = sample_positions(rng, desired_walls, size, forbidden)
    forbidden.update({(x, y) for x, y in wall_locs})

    rad_sources = sample_positions(rng, desired_rad_sources, size, forbidden)
    rad_low, rad_high = RADIATION_STRENGTH_RANGES[strength_level]
    rad_consts = [[float(rng.uniform(rad_low, rad_high)), float(rng.uniform(rad_low, rad_high))] for _ in rad_sources]

    grid_config = {
        'size': size,
        'agent_locs': agent_locs,
        'target_loc': target_loc,
        'wall_locs': wall_locs,
        'radiation_locs': rad_sources,
        'radiation_consts': rad_consts,
        'distance_multiplier': 0.1,
        'radiation_multiplier': 0.1,
        'target_reward': max(1.0, TARGET_REWARD_PER_UNIT * size),
        'transition_prob': 0.9,
        'collision_penalty': 0.05,
    }
    return grid_config


def main(sizes, seed, log_dir, num_agents, num_episodes):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    rng = np.random.default_rng(seed)
    os.makedirs(log_dir, exist_ok=True)

    difficulty_levels = ('low', 'medium', 'high')
    num_episodes = max(1, num_episodes)

    for size in sizes:
        max_steps_for_size = max(1, math.ceil(2.5 * size))
        solver_params = {
            'num_episodes': num_episodes,
            'max_steps': max_steps_for_size,
        }
        agents_for_size = max(1, min(num_agents, size))
        if num_agents > size:
            print(f"Requested {num_agents} agents for size {size}; using maximum allowed {agents_for_size}.")
        size_dir = os.path.join(log_dir, f"size_{size}")
        os.makedirs(size_dir, exist_ok=True)

        grid_variants = list(product(difficulty_levels, repeat=3))
        grid_bar = tqdm(grid_variants, desc=f"Gridworlds (size {size})", leave=False)
        for wall_level, rad_level, strength_level in grid_bar:
            grid_dir = os.path.join(size_dir, f"grid_w-{wall_level}_r-{rad_level}_s-{strength_level}")
            os.makedirs(grid_dir, exist_ok=True)

            grid_config = build_grid_config(size, agents_for_size, wall_level, rad_level, strength_level, rng)
            grid_bar.set_postfix({
                'walls': wall_level,
                'rad': rad_level,
                'strength': strength_level,
                'agents': len(grid_config['agent_locs']),
                'wall_ct': len(grid_config['wall_locs']),
                'rad_ct': len(grid_config['radiation_locs']),
            })
            with open(os.path.join(grid_dir, "grid_config.json"), "w") as fp:
                json.dump(grid_config, fp, indent=2)

            runner = None
            for solver_idx, solver_type in enumerate(SOLVER_TYPES):
                solver_dir = os.path.join(grid_dir, solver_type)
                os.makedirs(solver_dir, exist_ok=True)
                solver_config = {'type': solver_type, 'params': solver_params}
                runner = Runner(grid_config, solver_config)
                if solver_idx == 0:
                    runner.env.render(render_mode='radiation_map', dir=os.path.join(grid_dir, "radiation_map.png"))
                runner.run(verbose=False, log_path=solver_dir)
                runner.plot_run(log_path=solver_dir)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog='RadiationGridworldRuns',
        description='Performs randomized RL Training on Radiation Gridworld Environments',
    )

    parser.add_argument(
        '--sizes',
        help='The gridworld sizes to test',
        type=int,
        nargs='+',
        default=[5, 10, 50, 100]
    )
    parser.add_argument(
        '--seed',
        help='Random seed',
        type=int,
        default=123
    )
    parser.add_argument(
        '--log-dir',
        help='Folder name to log results',
        default='runs'
    )
    parser.add_argument(
        '--num-agents',
        help='Number of agents to place in each gridworld (will be clamped to [1, size])',
        type=int,
        default=3
    )
    parser.add_argument(
        '--num-episodes',
        help='Number of training episodes per solver',
        type=int,
        default=1000
    )
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    
    main(
        sizes=args.sizes,
        seed=args.seed,
        log_dir=args.log_dir,
        num_agents=args.num_agents,
        num_episodes=args.num_episodes,
    )
