"""
Utilities to visualize a single episode in RadiationGridworld using matplotlib.
Generates a heatmap background (radiation), overlays walls/target/agents, and
produces an MP4 (ffmpeg) or GIF (PillowWriter fallback).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def rollout_states(env, policy, max_steps=None):
    """
    Roll out one episode using a policy; record positions and reached flags per step.
    Returns a list of dicts with keys: positions (np.ndarray), reached (np.ndarray).
    """
    states = []
    obs, _ = env.reset()
    # Record initial state
    states.append({
        "positions": np.array(env._agent_locations, copy=True),
        "reached": np.array(env._reached_target, copy=True)
    })
    done = False
    steps = 0
    while not done and (max_steps is None or steps < max_steps):
        actions = policy(obs)
        _, obs, _, done, _ = env.step(actions)
        steps += 1
        # Record post-step state (captures final frame when episode ends)
        states.append({
            "positions": np.array(env._agent_locations, copy=True),
            "reached": np.array(env._reached_target, copy=True)
        })
    return states


def animate_episode(env, policy, out_path="episode.mp4", fps=4, dpi=150, max_steps=None):
    """
    Create an animation of a single episode and save to MP4 (if ffmpeg available) or GIF.
    env: RadiationGridworld instance
    policy: callable mapping obs -> actions
    out_path: output filename (extension determines first attempt)
    fps: frames per second
    dpi: rendering DPI
    max_steps: optional cap on steps to render
    Returns the path of the saved file (MP4 or GIF).
    """
    states = rollout_states(env, policy, max_steps=max_steps)
    size = env.size

    # Background heatmap: flip y-axis for bottom-left origin
    background = np.flipud(env._radiation_vals.T)

    # Wider figure to leave space for legend outside the grid.
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title("Radiation Gridworld")
    heat = ax.imshow(background, cmap="magma", origin="lower", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    # Gridlines removed for a cleaner look; adjust if you want cell boundaries visible.
    # ax.grid(True, color="white", alpha=0.3, linewidth=0.5)
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)

    # Walls/target overlays
    if len(env._wall_locations):
        wx, wy = env._wall_locations.T
        ax.scatter(wx, wy, marker="s", s=300, c="gray", label="Wall")
    tx, ty = env._target_location
    ax.scatter([tx], [ty], marker="*", s=300, c="cyan", edgecolor="k", label="Target")

    agent_scat = ax.scatter([], [], c="lime", edgecolor="k", s=200, label="Agent")
    text_step = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                        color="white", fontsize=10)

    # Place legend outside the grid with extra spacing to avoid overlap.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.5, labelspacing=1.0, borderpad=0.8, handlelength=2.5, handletextpad=0.8)
    fig.subplots_adjust(right=0.8)

    def init():
        agent_scat.set_offsets(np.empty((0, 2)))
        text_step.set_text("")
        return agent_scat, text_step

    def update(frame_idx):
        state = states[frame_idx]
        pos = state["positions"]
        agent_scat.set_offsets(pos)
        text_step.set_text(f"Step {frame_idx + 1}/{len(states)}")
        # Color agents that reached target differently
        colors = ["lime" if not reached else "gold" for reached in state["reached"]]
        agent_scat.set_color(colors)
        return agent_scat, text_step

    anim = animation.FuncAnimation(
        fig, update, frames=len(states), init_func=init,
        interval=1000 / fps, blit=True
    )

    # Try requested writer; fallback to GIF
    saved_path = out_path
    try:
        anim.save(out_path, writer="ffmpeg", fps=fps, dpi=dpi)
    except Exception:
        from matplotlib.animation import PillowWriter
        gif_path = out_path.rsplit(".", 1)[0] + ".gif"
        anim.save(gif_path, writer=PillowWriter(fps=fps))
        saved_path = gif_path

    plt.close(fig)
    return saved_path


__all__ = ["rollout_states", "animate_episode"]
