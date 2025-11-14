# CSCE642-Project

### Install Requirements
`pip install -r requirements.txt`

### Basic Usage
`python -m Solvers.runner.py`

### config.json Format
The runner expects a JSON file with the following top-level structure:

```jsonc
{
  "gridworld": { /* environment configuration */ },
  "solver": {   /* training configuration */ }
}
```
The `gridworld` section specifies the radiation gridworld hyperparameters:

- `gridworld.size` *(int)*: Number of rows/columns in the square grid.
- `gridworld.agent_loc` *(2-item list[int, int])*: Starting `[x, y]` coordinate of the agent; origin is bottom-left.
- `gridworld.target_loc` *(2-item list[int, int])*: Goal coordinate the agent must reach.
- `gridworld.wall_locs` *(list[list[int, int]])*: Obstacles that block movement.
- `gridworld.radiation_locs` *(list[list[int, int]])*: Cells that emit radiation.
- `gridworld.radiation_consts` *(list[list[float, float]])*: `[gamma, activity]` constants for each radiation source, ordered to match `radiation_locs`.
- `gridworld.distance_multiplier` *(float, optional)*: Weight on distance-based penalty (default `0.1`).
- `gridworld.radiation_multiplier` *(float, optional)*: Weight on radiation penalty (default `0.1`).
- `gridworld.target_reward` *(float, optional)*: Reward received upon reaching the target (default `1.0`)
- `gridworld.transition_prob` *(float, optional)*: Probability that the chosen action is executed (default `1.0`).

The `solver` section specifies which algorithm to run and its hyperparameters:

- `solver.type` *(string)*: Currently `"Policy Iteration"` is supported.
- `solver.params.epsilon` *(float, optional)*: Included for API symmetry; ignored by policy iteration.
- `solver.params.gamma` *(float)*: Discount factor in `[0, 1)`.
- `solver.params.num_episodes` *(int)*: Number of policy-iteration sweeps to perform.
- `solver.params.max_steps` *(int)*: Reserved for future solvers; unused in policy iteration.

A complete example (matching the default `config.json`):

```json
{
  "gridworld": {
    "size": 10,
    "agent_loc": [0, 0],
    "target_loc": [9, 9],
    "wall_locs": [[3, 2], [7, 5], [8, 2], [4, 4], [3, 0]],
    "radiation_locs": [[2, 3], [7, 1], [6, 6]],
    "radiation_consts": [[1, 1], [1, 1], [1, 1]],
    "distance_multiplier": 0.01,
    "radiation_multiplier": 0.01,
    "transition_prob": 1.0
  },
  "solver": {
    "type": "Policy Iteration",
    "params": {
      "epsilon": 0.1,
      "gamma": 0.1,
      "num_episodes": 10,
      "max_steps": 100
    }
  }
}
```
