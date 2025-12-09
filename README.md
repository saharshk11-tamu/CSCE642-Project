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
- Supports multi-agent setups; agent starts are sampled uniquely per grid and clamped to `[1, size]` per grid.
- Trains three solvers on every grid (`DQN`, `A2C`, `REINFORCE`) for the specified number of episodes; max steps per episode is `ceil(2.5 * grid_size)`.
- Logs each run under `--log-dir`, saving training rewards (`*.npy`), env info (`*_env_info.json`), and plots, plus one radiation heatmap per grid.

## Usage

### Example
```bash
python main.py --sizes 10 20 --num-agents 3 --seed 42 --log-dir runs --num-episodes 500
```

### Reproduce Experiment
```bash
python main.py
```

### Arguments
- `--sizes`: one or more grid sizes (ints). Default: `10`.
- `--num-agents`: agents per grid (clamped to `[1, size]` per grid). Default: `3`.
- `--seed`: global seed for numpy/random/torch. Default: `123`.
- `--log-dir`: base directory for outputs. Default: `runs`.
- `--num-episodes`: training episodes per solver (min 1). Default: `1000`.
- Max steps per episode is automatically set to `ceil(2.5 * grid_size)` per run.

## Outputs
- `log-dir/size_<N>/grid_w-<wall>_r-<rad>_s-<strength>/grid_config.json`: the generated grid definition.
- `.../radiation_map.png`: heatmap of radiation for that grid (one per grid).
- `.../<SOLVER>/`: per-solver logs (training rewards `.npy`, env info `.json`, training plot, episode animation).

## Notes
- Radiation source counts scale with grid area via density thresholds; walls/radiation are sampled to avoid agent/target collisions.
- If `--num-agents` exceeds the grid size, it is capped and a message is printed. Progress is shown via `tqdm` during grid generation/training.***
