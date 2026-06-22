"""
Run repeated evaluations (multiple start positions x seeds) for a single saved model
and collect results for downstream plotting and metrics.

Usage:
    python -m evaluation.aggregate_eval

Note: Environment.evaluate_agent() only returns the final world_stats dict, it does not
expose per-step agent_path/collision_path, and it unconditionally builds a path plot (and
writes a results .txt file) on every call. To collect trajectories for many runs without
modifying evaluate_agent or the visualizers, _run_episode() below reimplements the same
single-episode loop evaluate_agent uses internally (same Environment/ObservationBuilder/
agent.take_action/env.step primitives), so the resulting world_stats are equivalent to what
evaluate_agent would report for the same model/space/start_pos/seed/sigma/sensor_range/max_steps.
"""

from pathlib import Path
import pandas as pd
from evaluation.evaluate_model import load_agent
from world.environment import Environment
from world.state import ObservationBuilder

WORLD_STATS_FIELDS = [
    "cumulative_reward",
    "total_steps",
    "total_agent_moves",
    "total_failed_moves",
    "total_targets_reached",
    "total_collision",
    "targets_remaining",
]


def build_environment(space_path, sigma=0.0, start_pos=None, seed=0):
    """Construct and reset an Environment for a single evaluation episode (or for plotting)."""
    env = Environment(
        space_path=space_path,
        no_gui=True,
        sigma=sigma,
        agent_start_pos=start_pos,
        target_fps=-1,
        random_seed=seed,
    )
    env.reset()
    return env


def _run_episode(agent, space_path, start_pos, seed, sensor_range, sigma, max_steps):
    """Run a single greedy episode, mirroring Environment.evaluate_agent's internal loop.

    :return: (world_stats, agent_path, collision_path)
    """
    env = build_environment(space_path, sigma=sigma, start_pos=start_pos, seed=seed)
    obs_builder = ObservationBuilder(env, agent.obs_mode, sensor_range)
    state = obs_builder.build(env.agent_pos)

    agent_path = [env.agent_pos]
    collision_path = [False]
    info = {"target_reached": False}

    for _ in range(max_steps):
        action = agent.take_action(state)
        _, _, terminated, info = env.step(action)
        state = obs_builder.build(env.agent_pos)
        agent_path.append(env.agent_pos)
        collision_path.append(info["collided"])
        if terminated:
            break

    env.world_stats["targets_remaining"] = 0 if info["target_reached"] else 1
    return dict(env.world_stats), agent_path, collision_path


def aggregate_runs(model_path, space_path, start_positions, seeds,
                    sensor_range=10.0, sigma=0.0, max_steps=250, model_id=None):
    """
    Run a (start_pos x seed) evaluation grid for a single saved model.

    The agent is loaded once via load_agent() and reused across every run -- evaluate_agent
    already accepts a pre-loaded agent object (its `agent` param), so this requires no change
    there.

    :param model_path: path to the saved .pt model
    :param space_path: path to the space .pickle file
    :param start_positions: list of (x, y) tuples
    :param seeds: list of ints
    :param sensor_range: sensor range, must match training
    :param sigma: environment stochasticity
    :param max_steps: max steps per episode
    :param model_id: label stored in summary rows (default: model_path stem)

    :return: (summary_df, paths, positions_by_start)
        summary_df: pandas DataFrame, one row per (start_pos, seed)
        paths: dict {(start_pos, seed): {"agent_path": [...], "collision_path": [...]}}
        positions_by_start: dict {start_pos: [flat (x, y) list across all seeds]},
                            ready to pass as `all_positions` to visualize_heatmap
    """
    agent = load_agent(model_path)
    model_id = model_id or Path(model_path).stem

    rows = []
    paths = {}
    positions_by_start = {start_pos: [] for start_pos in start_positions}

    for start_pos in start_positions:
        for seed in seeds:
            stats, agent_path, collision_path = _run_episode(
                agent, space_path, start_pos, seed, sensor_range, sigma, max_steps,
            )

            row = {field: stats[field] for field in WORLD_STATS_FIELDS}
            row["start_x"] = start_pos[0]
            row["start_y"] = start_pos[1]
            row["start_pos"] = f"{start_pos[0]},{start_pos[1]}"
            row["seed"] = seed
            row["model_id"] = model_id
            rows.append(row)

            paths[(start_pos, seed)] = {
                "agent_path": agent_path,
                "collision_path": collision_path,
            }
            positions_by_start[start_pos].extend(agent_path)

    summary_df = pd.DataFrame(rows)
    return summary_df, paths, positions_by_start


def compute_metrics(summary_df, group_col="start_pos"):
    """
    Compute success rate, steps-to-goal, reward, and collision metrics grouped by group_col.

    - success_rate: fraction of runs with targets_remaining == 0
    - steps_to_goal mean/std: total_steps, restricted to successful runs only
      (targets_remaining == 0) so failed/timed-out episodes don't pollute it
    - cumulative_reward mean/std: over all runs (success and failure)
    - total_collision mean: over all runs

    :param summary_df: DataFrame as returned by aggregate_runs
    :param group_col: column to group by (e.g. "start_pos" or "model_id")
    :return: pandas DataFrame, one row per group value
    """
    records = []
    for group_value, group_df in summary_df.groupby(group_col):
        successful = group_df[group_df["targets_remaining"] == 0]

        records.append({
            group_col: group_value,
            "n_runs": len(group_df),
            "success_rate": len(successful) / len(group_df),
            "steps_to_goal_mean": successful["total_steps"].mean() if not successful.empty else float("nan"),
            "steps_to_goal_std": successful["total_steps"].std() if not successful.empty else float("nan"),
            "cumulative_reward_mean": group_df["cumulative_reward"].mean(),
            "cumulative_reward_std": group_df["cumulative_reward"].std(),
            "total_collision_mean": group_df["total_collision"].mean(),
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    # 20260618_003533 is excluded on purpose: its checkpoints don't reproduce their own
    # manifest (save/load mismatch), so symmetric_ablation_2 is the known-good demo run.
    model_path = Path("results/experiments/symmetric_ablation_2/saved_models/exp_0001.pt")
    space_path = Path("spaces/symmetric_2_space.pickle")
    start_positions = [(5.0, 3.0), (15.0, 3.0)]
    seeds = [0]

    summary_df, paths, positions_by_start = aggregate_runs(
        model_path, space_path, start_positions, seeds,
    )

    print(summary_df)
    print()
    print(compute_metrics(summary_df))
