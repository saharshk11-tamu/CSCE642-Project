import os
import json
from typing import Dict, Any, List

import numpy as np

from RadiationGridworld import RadiationGridworld
from Solvers.DQL import DQN
from Solvers.REINFORCE import Reinforce
from Solvers.A2C import A2C


# 1. Define a standard set of environments
#    You can add/remove scenarios as needed.
STANDARD_ENVIRONMENTS: List[Dict[str, Any]] = [
    {
        "name": "env_baseline_single_source",
        "gridworld": {
            "size": 10,
            "agent_loc": [0, 0],
            "target_loc": [9, 9],
            "wall_locs": [[3, 2], [7, 5], [8, 2], [4, 4], [0, 3]],
            "radiation_locs": [[2, 3]],
            "radiation_consts": [[1.0, 1.0]],
            "distance_multiplier": 0.01,
            "radiation_multiplier": 0.1,
            "target_reward": 1.0,
            "transition_prob": 1.0,
            # optional – uses the defaults from your updated env if omitted:
            # "wall_attenuation_coeff": 0.5,
            # "wall_thickness": 1.0,
        },
    },
    {
        "name": "env_two_sources_shielded_path",
        "gridworld": {
            "size": 10,
            "agent_loc": [0, 0],
            "target_loc": [9, 9],
            "wall_locs": [
                [4, 0], [4, 1], [4, 2], [4, 3], [4, 4],  # vertical wall
                [5, 4], [6, 4], [7, 4]                   # horizontal extension
            ],
            "radiation_locs": [[2, 7], [7, 1]],
            "radiation_consts": [[1.0, 1.0], [1.0, 1.0]],
            "distance_multiplier": 0.01,
            "radiation_multiplier": 0.1,
            "target_reward": 1.0,
            "transition_prob": 1.0,
        },
    },
    {
        "name": "env_close_source_with_wall_shadow",
        "gridworld": {
            "size": 10,
            "agent_loc": [0, 0],
            "target_loc": [9, 0],
            "wall_locs": [
                [3, 0], [3, 1], [3, 2], [3, 3]   # wall that can be used as shielding
            ],
            "radiation_locs": [[5, 2]],
            "radiation_consts": [[1.0, 1.0]],
            "distance_multiplier": 0.01,
            "radiation_multiplier": 0.1,
            "target_reward": 1.0,
            "transition_prob": 1.0,
        },
    },
]


# 2. Solver specs (you can tune these; these roughly mirror your JSON configs)
SOLVER_SPECS = {
    "DQN": {
        "class": DQN,
        "params": {
            "epsilon": 0.1,
            "gamma": 0.1,
            "num_episodes": 1000,
            "max_steps": 100,
            "layers": [64, 64],
            "replay_buffer_size": 2000,
            "batch_size": 64,
            "update_target_every": 1000,
        },
    },
    "REINFORCE": {
        "class": Reinforce,
        "params": {
            "epsilon": 0.1,
            "gamma": 0.1,
            "num_episodes": 1000,
            "max_steps": 100,
            "layers": [64, 64],
            "lr": 1e-3,
        },
    },
    "A2C": {
        "class": A2C,
        "params": {
            "epsilon": 0.1,
            "gamma": 0.1,
            "num_episodes": 1000,
            "max_steps": 100,
            "layers": [64, 64],
            "lr": 1e-3,
        },
    },
}


def evaluate_policy_metrics(env: RadiationGridworld, solver, num_episodes: int = 100):
    """
    Evaluate the final greedy policy with extra metrics:
    - average return
    - success rate (reaching the target)
    - average steps per episode
    - average cumulative radiation dose (using env._radiation_vals)
    """
    policy = solver.create_greedy_policy()

    returns = []
    successes = 0
    steps_list = []
    dose_list = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        total_dose = 0.0
        steps = 0
        done = False

        while not done and steps < solver.max_steps:
            action = policy(state)
            _, next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Approximate per-step dose from current agent location
            ax, ay = int(env._agent_location[0]), int(env._agent_location[1])
            total_dose += float(env._radiation_vals[ax, ay])

            state = next_state
            steps += 1

        returns.append(total_reward)
        steps_list.append(steps)
        dose_list.append(total_dose)

        # Success = ended on the target
        if np.array_equal(env._agent_location, env._target_location):
            successes += 1

    returns = np.array(returns, dtype=np.float32)
    steps_arr = np.array(steps_list, dtype=np.float32)
    dose_arr = np.array(dose_list, dtype=np.float32)

    avg_return = float(returns.mean())
    std_return = float(returns.std())
    success_rate = successes / num_episodes
    avg_steps = float(steps_arr.mean())
    avg_dose = float(dose_arr.mean())

    return {
        "eval_avg_return": avg_return,
        "eval_std_return": std_return,
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_cumulative_dose": avg_dose,
    }


def run_experiments(output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)

    summary = []

    for env_spec in STANDARD_ENVIRONMENTS:
        env_name = env_spec["name"]
        grid_cfg = env_spec["gridworld"]

        for solver_name, solver_info in SOLVER_SPECS.items():
            print(f"\n=== Running {solver_name} on {env_name} ===")

            # Fresh env for this (env, solver) combo
            env = RadiationGridworld(**grid_cfg)

            SolverClass = solver_info["class"]
            solver_params = dict(solver_info["params"])  # shallow copy
            solver = SolverClass(env, **solver_params)

            # Training loop (same idea as Runner.run)
            episode_rewards = []
            for ep in range(solver.num_episodes):
                solver.train_episode()
                episode_rewards.append(float(solver.reward))

            # Evaluation
            eval_metrics = evaluate_policy_metrics(env, solver, num_episodes=100)

            run_result = {
                "environment": env_name,
                "solver": solver_name,
                "training_rewards": episode_rewards,
                **eval_metrics,
            }

            # Per-run log
            log_path = os.path.join(
                output_dir, f"{env_name}_{solver_name}_results.json"
            )
            with open(log_path, "w") as f:
                json.dump(run_result, f, indent=2)

            summary.append(run_result)

    # Global summary for quick loading in notebooks
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll experiments complete. Summary written to {summary_path}")


if __name__ == "__main__":
    run_experiments()
