# Radiation Gridworld Experiment Runner

## Quick start

### With uv
```bash
# Ensure uv is installed (https://docs.astral.sh/uv/)
uv venv .venv
uv sync
```

### With pip
```bash
pip install -r requirements.txt
```

## What it does
`main.py` generates randomized Radiation Gridworld environments and trains multiple reinforcement learning agents on each variation. It automates creating grid configurations, logging results, and saving visualizations.
- For each requested grid size, builds 27 grid variations (wall density × radiation source density × radiation strength level).
- Randomly samples wall locations, radiation sources, and radiation strengths (seeded for reproducibility).
- Supports multi-agent setups; agent starts are sampled uniquely per grid.
- Trains three solvers on every grid (`DQN`, `A2C`, `REINFORCE`) for the specified number of episodes/steps.
- Logs each run under `--log-dir`, saving training rewards (`*.npy`) and plots, plus one radiation heatmap per grid.

## Usage
```bash
python main.py --sizes 5 10 --num-agents 3 --seed 123 --log-dir runs --num-episodes 50 --max-steps 200
```

### Arguments
- `--sizes`: one or more grid sizes (ints).
- `--num-agents`: agents per grid (clamped to `[1, size]` per grid).
- `--seed`: global seed for numpy/random/torch.
- `--log-dir`: base directory for outputs (defaults to `runs`).
- `--num-episodes`: training episodes per solver (min 1).
- `--max-steps`: max steps per episode (min 1).

## Outputs
- `runs/size_<N>/grid_w-<wall>_r-<rad>_s-<strength>/grid_config.json`: the generated grid definition.
- `.../radiation_map.png`: heatmap of radiation for that grid (one per grid).
- `.../<SOLVER>/`: per-solver logs (training rewards `.npy` and plots).

## Notes
- Radiation source counts scale with grid area via density thresholds; walls/radiation are sampled to avoid agent/target collisions.
- If `--num-agents` exceeds the grid size, it is capped and a message is printed.***
