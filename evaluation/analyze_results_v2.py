"""
Analyze experiment results comparing obs_mode and sigma and create plots

Usage:
    python -m evaluation.analyze_results_v2 results/experiments/<timestamp>
    python -m evaluation.analyze_results_v2 results/experiments/<timestamp> --obs_mode both

Output in <run_dir>/analysis/:
    paths_obs_modes.pdf: path visualizations for obs_modes
    paths_sigma.pdf: path visualizations for sigma values
    convergence_sigma.pdf: convergence plot (reward over episodes) for all sigmas 

Note: Requires SAVE_MODELS=True during the experiment run (need the saved models for this script)
Note: best obs_mode is used for sigma and convergence plots, but can override it with --obs_mode to set specific one
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "world"))

from evaluation.evaluate_model import load_agent
from world.environment import Environment
from world.state import ObservationBuilder

# plot settings
AGENT_COLORS    = {"dqn": "steelblue", "ppo": "darkorange"}
OBS_LINESTYLES  = {"xy": "solid", "sensors": "dashed", "both": "dotted"}
SMOOTHING_WINDOW = 50 # smoothen to reduce noise in convergence plots


def smooth(values, window):
    result = np.empty(len(values))
    for i in range(len(values)):
        result[i] = np.mean(values[max(0, i - window + 1): i + 1])
    return result


def draw_env_background(ax, env):
    for obs_x, obs_y, obs_w, obs_h in env.obstacles:
        ax.add_patch(mpatches.Rectangle((obs_x, obs_y), obs_w, obs_h, facecolor="black"))
    ax.add_patch(plt.Circle(env.target_pos, env.target_radius, facecolor="green", alpha=0.5))
    ax.set_xlim(0, env.x_max)
    ax.set_ylim(0, env.y_max)
    ax.set_aspect("equal")


def get_path(model_path, space_path, max_steps, sigma, random_seed=0, sensor_range=10.0):
    """Run one greedy episode on a saved model and return (env, path) for path plots"""
    agent = load_agent(model_path)
    env = Environment(space_path=space_path, no_gui=True, sigma=sigma, random_seed=random_seed)
    obs_builder = ObservationBuilder(env, agent.obs_mode, sensor_range)
    env.reset()
    path = [env.agent_pos]
    state = obs_builder.build(env.agent_pos)
    for _ in range(max_steps):
        action = agent.take_action(state)
        _, _, done, _ = env.step(action)
        state = obs_builder.build(env.agent_pos)
        path.append(env.agent_pos)
        if done:
            break
    return env, path


def best_obs_mode_for(results_df, agent):
    """Return obs_mode of the single best-performing experiment for this agent."""
    agent_df = results_df[results_df["agent"] == agent]
    return agent_df.loc[agent_df["eval_cumulative_reward"].idxmax(), "obs_mode"]


def get_experiment(results_df, agent, obs_mode, sigma):
    """Return the best experiment row for a given (agent, obs_mode, sigma) combination"""
    matches = results_df[
        (results_df["agent"] == agent) &
        (results_df["obs_mode"] == obs_mode) &
        np.isclose(results_df["sigma"], sigma)
    ]
    if matches.empty:
        return None
    return matches.loc[matches["eval_cumulative_reward"].idxmax()]


def get_best_experiment(results_df, agent, obs_mode):
    """Return the best experiment row for (agent, obs_mode) across all sigma values"""
    matches = results_df[
        (results_df["agent"] == agent) &
        (results_df["obs_mode"] == obs_mode)
    ]
    if matches.empty:
        return None
    return matches.loc[matches["eval_cumulative_reward"].idxmax()]


def saved_model_path(run_dir, exp_id):
    return run_dir / "saved_models" / f"exp_{int(exp_id):04d}.pt"


def plot_paths_obs_modes(results_df, run_dir, out_dir, space_path):
    """Best model per (agent, obs_mode) with sigma=0 in evaluation"""
    max_steps = int(results_df["max_steps"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 8))
    env_drawn = False
    legend_handles = []

    for agent in sorted(results_df["agent"].unique()):
        for obs_mode, linestyle in OBS_LINESTYLES.items():
            row = get_best_experiment(results_df, agent, obs_mode)
            if row is None:
                continue
            model_file = saved_model_path(run_dir, row["exp_id"])
            if not model_file.exists():
                print(f"  Skipping {model_file.name} (not found)")
                continue

            env, path = get_path(model_file, space_path, max_steps, sigma=0.0)
            if not env_drawn:
                draw_env_background(ax, env)
                env_drawn = True

            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            line, = ax.plot(xs, ys, color=AGENT_COLORS[agent], linestyle=linestyle, linewidth=1.5, alpha=0.85)
            legend_handles.append((line, f"{agent.upper()} — {obs_mode}"))

    ax.legend([h for h, _ in legend_handles], [lbl for _, lbl in legend_handles], fontsize=9)
    ax.set_title("Path by obs_mode  (best model per obs_mode, σ=0 evaluation)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out_dir / "paths_obs_modes.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved paths_obs_modes.pdf")


def plot_paths_sigma(results_df, run_dir, out_dir, space_path, obs_mode_override=None):
    """All sigma values, both agents, best obs_mode per agent (or set obs_mode value)"""
    sigmas = sorted(results_df["sigma"].unique())
    sigma_linestyles = ["solid", "dashed", "dotted"]
    max_steps = int(results_df["max_steps"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 8))
    env_drawn = False
    legend_handles = []

    for agent in sorted(results_df["agent"].unique()):
        obs_mode = obs_mode_override or best_obs_mode_for(results_df, agent)
        print(f"  {agent.upper()}: obs_mode='{obs_mode}'")

        for sigma, linestyle in zip(sigmas, sigma_linestyles):
            row = get_experiment(results_df, agent, obs_mode, sigma)
            if row is None:
                continue
            model_file = saved_model_path(run_dir, row["exp_id"])
            if not model_file.exists():
                print(f"  Skipping {model_file.name} (not found)")
                continue

            env, path = get_path(model_file, space_path, max_steps, sigma)
            if not env_drawn:
                draw_env_background(ax, env)
                env_drawn = True

            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            line, = ax.plot(xs, ys, color=AGENT_COLORS[agent], linestyle=linestyle, linewidth=1.5, alpha=0.85)
            legend_handles.append((line, f"{agent.upper()} — σ={sigma}"))

    obs_modes_used = {agent: (obs_mode_override or best_obs_mode_for(results_df, agent))
                      for agent in sorted(results_df["agent"].unique())}
    obs_label = ", ".join(f"{a.upper()}={m}" for a, m in obs_modes_used.items())

    ax.legend([h for h, _ in legend_handles], [lbl for _, lbl in legend_handles], fontsize=9)
    ax.set_title(f"Path by sigma (obs_mode: {obs_label})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out_dir / "paths_sigma.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved paths_sigma.pdf")


def plot_convergence(results_df, curves_df, out_dir, obs_mode_override=None, show_episodes_until=None):
    """(Smoothed) training reward over episodes, best obs_mode per agent, for different sigmas"""
    if curves_df is None or curves_df.empty:
        print("No training curves found, skipping.")
        return

    sigmas = sorted(results_df["sigma"].unique())
    sigma_linestyles = ["solid", "dashed", "dotted"]

    fig, ax = plt.subplots(figsize=(10, 5))
    legend_handles = []

    for agent in sorted(results_df["agent"].unique()):
        obs_mode = obs_mode_override or best_obs_mode_for(results_df, agent)

        for sigma, linestyle in zip(sigmas, sigma_linestyles):
            row = get_experiment(results_df, agent, obs_mode, sigma)
            if row is None:
                continue

            episode_data = curves_df[curves_df["exp_id"] == int(row["exp_id"])].sort_values("episode")
            if episode_data.empty:
                continue

            smoothed = smooth(episode_data["episode_reward"].tolist(), SMOOTHING_WINDOW)
            line, = ax.plot(np.arange(len(smoothed)), smoothed,
                            color=AGENT_COLORS[agent], linestyle=linestyle, linewidth=1.5)
            legend_handles.append((line, f"{agent.upper()} — σ={sigma} ({obs_mode})"))

    ax.legend([h for h, _ in legend_handles], [lbl for _, lbl in legend_handles], fontsize=9)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward (smoothed, window={SMOOTHING_WINDOW})")
    ax.set_title("Training convergence by sigma  (best obs_mode per agent)")
    ax.grid(linestyle="--", alpha=0.3)
    if show_episodes_until is not None:
        ax.set_xlim(0, show_episodes_until)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_sigma.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved convergence_sigma.pdf")


def plot_convergence_obs_modes(results_df, curves_df, out_dir, show_episodes_until=None):
    """Smoothed training reward over episodes for best run per (agent, obs_mode) combination"""
    fig, ax = plt.subplots(figsize=(10, 5))
    legend_handles = []

    for agent in sorted(results_df["agent"].unique()):
        for obs_mode, linestyle in OBS_LINESTYLES.items():
            row = get_best_experiment(results_df, agent, obs_mode)
            if row is None:
                continue

            episode_data = curves_df[curves_df["exp_id"] == int(row["exp_id"])].sort_values("episode")
            if episode_data.empty:
                continue

            smoothed = smooth(episode_data["episode_reward"].tolist(), SMOOTHING_WINDOW)
            line, = ax.plot(np.arange(len(smoothed)), smoothed,
                            color=AGENT_COLORS[agent], linestyle=linestyle, linewidth=1.5)
            legend_handles.append((line, f"{agent.upper()} — {obs_mode}"))

    ax.legend([h for h, _ in legend_handles], [lbl for _, lbl in legend_handles], fontsize=9)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward (smoothed, window={SMOOTHING_WINDOW})")
    ax.set_title("Training convergence by obs_mode  (best run per obs_mode)")
    ax.grid(linestyle="--", alpha=0.3)
    if show_episodes_until is not None:
        ax.set_xlim(0, show_episodes_until)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_obs_modes.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved convergence_obs_modes.pdf")


def main(run_dir_arg, obs_mode_override=None, show_episodes_until=None):
    run_dir = Path(run_dir_arg)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    results_csv = run_dir / "experiment_results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"experiment_results.csv not found in {run_dir}")

    results_df = pd.read_csv(results_csv)
    curves_csv = run_dir / "training_curves.csv"
    curves_df = pd.read_csv(curves_csv) if curves_csv.exists() else None

    print(f"Loaded {len(results_df)} experiments from {run_dir}")

    space_stem = results_df["space"].iloc[0]
    space_path = Path("spaces") / f"{space_stem}.pickle"
    if not space_path.exists():
        raise FileNotFoundError(f"Space file not found: {space_path}")

    print("\nGenerating paths by obs_mode...")
    plot_paths_obs_modes(results_df, run_dir, out_dir, space_path)

    print("\nGenerating paths by sigma...")
    plot_paths_sigma(results_df, run_dir, out_dir, space_path, obs_mode_override)

    print("\nGenerating convergence by sigma plot...")
    plot_convergence(results_df, curves_df, out_dir, obs_mode_override, show_episodes_until)

    print("\nGenerating convergence by obs_mode plot...")
    plot_convergence_obs_modes(results_df, curves_df, out_dir, show_episodes_until)

    print(f"\nAll outputs saved in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment results (obs_mode and sigma comparison)")
    parser.add_argument("run_dir", help="Path to experiment run directory")
    parser.add_argument("--obs_mode", default=None, choices=["xy", "sensors", "both"],
                        help="Override obs_mode for sigma/convergence plots (default: best per agent)")
    parser.add_argument("--show_episodes_until", type=int, default=None,
                        help="Clip the convergence plot x-axis at this episode number (default: show all)")
    args = parser.parse_args()
    main(args.run_dir, obs_mode_override=args.obs_mode, show_episodes_until=args.show_episodes_until)
