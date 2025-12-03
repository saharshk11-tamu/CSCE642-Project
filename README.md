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
- Trains three solvers on every grid (`DQN`, `A2C`, `REINFORCE`) for the specified number of episodes/steps.
- Logs each run under `--log-dir`, saving training rewards (`*.npy`), env info (`*_env_info.json`), and plots, plus one radiation heatmap per grid.

## Usage

### Example
```bash
python main.py --sizes 8 16 --num-agents 2 --seed 42 --log-dir demo_runs --num-episodes 30 --max-steps 150
```

### Arguments
- `--sizes`: one or more grid sizes (ints). Default: `5 10 50 100`.
- `--num-agents`: agents per grid (clamped to `[1, size]` per grid). Default: `1`.
- `--seed`: global seed for numpy/random/torch. Default: `123`.
- `--log-dir`: base directory for outputs. Default: `runs`.
- `--num-episodes`: training episodes per solver (min 1). Default: `20`.
- `--max-steps`: max steps per episode (min 1). Default: `100`.

## Outputs
- `runs/size_<N>/grid_w-<wall>_r-<rad>_s-<strength>/grid_config.json`: the generated grid definition.
- `.../radiation_map.png`: heatmap of radiation for that grid (one per grid).
- `.../<SOLVER>/`: per-solver logs (training rewards `.npy`, env info `.json`, training plot).

## Notes
- Radiation source counts scale with grid area via density thresholds; walls/radiation are sampled to avoid agent/target collisions.
- If `--num-agents` exceeds the grid size, it is capped and a message is printed. Progress is shown via `tqdm` during grid generation/training.***
