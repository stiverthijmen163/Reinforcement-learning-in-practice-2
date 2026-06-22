"""
Overlay plot for aggregate_eval.aggregate_runs() output: one map, every trajectory drawn
on top of each other, color = agent, line style = obs_mode.

Usage:
    python -m evaluation.plot_eval
"""

from pathlib import Path

import matplotlib.pyplot as plt

from evaluation.aggregate_eval import build_environment

AGENT_COLORS = {"dqn": "steelblue", "ppo": "darkorange"}
OBS_LINESTYLES = {"sensors": "-", "both": ":", "xy": "--"}


def output_dir_for(run_dir, base=Path("results/eval_plots")):
    """results/eval_plots/<run_dir name>, so different experiment runs don't collide."""
    return Path(base) / Path(run_dir).name


def save_figure(fig, out_dir, name):
    """Save fig as PDF + PNG (150 dpi) under out_dir. Returns the saved paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for ext, kwargs in [("pdf", {}), ("png", {"dpi": 150})]:
        fp = out_dir / f"{name}.{ext}"
        fig.savefig(fp, bbox_inches="tight", **kwargs)
        saved.append(fp)
    return saved


def _draw_map(ax, env):
    for ox, oy, w, h in env.obstacles:
        ax.add_patch(plt.Rectangle((ox, oy), w, h, facecolor="black"))
    ax.add_patch(plt.Circle(env.target_pos, env.target_radius, facecolor="green", alpha=0.5))
    ax.set_xlim(0, env.x_max)
    ax.set_ylim(0, env.y_max)
    ax.set_aspect("equal")


def _plot_runs(ax, env, runs, label_fn):
    handles, labels = [], []
    for agent, obs_mode, agent_path in runs:
        xs = [p[0] for p in agent_path]
        ys = [p[1] for p in agent_path]
        line, = ax.plot(xs, ys, color=AGENT_COLORS[agent], linestyle=OBS_LINESTYLES[obs_mode], linewidth=1.5)

        label = label_fn(agent, obs_mode)
        if label not in labels:
            handles.append(line)
            labels.append(label)

        ax.add_patch(plt.Circle((xs[0], ys[0]), env.agent_radius, facecolor="#f2d352",
                                edgecolor="black", linewidth=0.5, zorder=10))

    ax.legend(handles, labels, fontsize=9, loc="upper right")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_overlay(space_path, runs, title=None):
    """
    One map, every trajectory overlaid. Color = agent, line style = obs_mode.

    :param space_path: path to the space .pickle file
    :param runs: list of (agent, obs_mode, agent_path) tuples
    :param title: optional figure title
    :return: the matplotlib Figure
    """
    env = build_environment(space_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_map(ax, env)
    _plot_runs(ax, env, runs, label_fn=lambda agent, obs_mode: f"{agent.upper()} — {obs_mode}")

    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_overlay_by_agent(space_path, runs, title=None):
    """
    Two subplots side by side, one per agent, each overlaying that agent's trajectories.
    Line style = obs_mode.

    :param space_path: path to the space .pickle file
    :param runs: list of (agent, obs_mode, agent_path) tuples
    :param title: optional figure title
    :return: the matplotlib Figure
    """
    env = build_environment(space_path)
    agents = sorted({agent for agent, _, _ in runs})
    fig, axes = plt.subplots(1, len(agents), figsize=(8 * len(agents), 8))

    for ax, agent in zip(axes, agents):
        _draw_map(ax, env)
        agent_runs = [r for r in runs if r[0] == agent]
        _plot_runs(ax, env, agent_runs, label_fn=lambda a, obs_mode: obs_mode)
        ax.set_title(agent.upper())

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import pandas as pd

    from evaluation.aggregate_eval import aggregate_runs

    run_dir = Path("results/experiments/symmetric_ablation_2")
    space_path = Path("spaces/symmetric_2_space.pickle")
    manifest = pd.read_csv(run_dir / "experiment_results.csv").sort_values(["agent", "obs_mode"])
    starts = [(5.0, 3.0), (15.0, 3.0)]

    runs = []
    for _, row in manifest.iterrows():
        model_path = run_dir / "saved_models" / f"exp_{int(row.exp_id):04d}.pt"
        for start in starts:
            _, paths, _ = aggregate_runs(model_path, space_path, [start], [0], sigma=0.0, max_steps=500)
            runs.append((row.agent, row.obs_mode, paths[(start, 0)]["agent_path"]))

    fig = plot_overlay(space_path, runs, title="DQN vs PPO, sensors vs both (symmetric_2_space)")
    saved = save_figure(fig, output_dir_for(run_dir), "trajectories_overlay")

    fig_split = plot_overlay_by_agent(space_path, runs, title="DQN vs PPO, sensors vs both (symmetric_2_space)")
    saved += save_figure(fig_split, output_dir_for(run_dir), "trajectories_overlay_split")

    for fp in saved:
        print(fp)
