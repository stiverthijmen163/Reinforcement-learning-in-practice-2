"""
Analyze and visualize experiment results from run_experiments.py.

Usage:
    python -m evaluation.analyze_results results/experiments/TIMESTAMP

Output in <run_dir>/analysis/:
    effect_<param>.png          - bar charts showing each parameter's impact
    training_curves_<...>.png   - smoothed training reward over episodes
    best_configs.csv            - best config per (agent, space) based on eval reward

    TODO: Add additional evaluation function to do something with the heatmaps
    Or do we want to keep that per run in training loop?
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


EVAL_METRICS = {
    "eval_cumulative_reward": "Eval reward",
    "eval_total_targets_reached": "Target reached",
    "eval_total_steps": "Steps to finish",
    "eval_total_collision": "Collisions",
}

PARAM_LABELS = {
    # Shared
    "sigma": "Sigma (σ)",
    "episodes": "Episodes",
    "max_steps": "Max steps",
    "learning_rate": "Learning rate (α)",
    "gamma": "Gamma (γ)",
    "batch_size": "Batch size",
    # DQN-specific
    "replay_capacity": "Replay capacity",
    "target_update_freq": "Target update freq",
    "epsilon": "Initial epsilon (ε)",
    "min_epsilon": "Min epsilon",
    "patience": "Patience",
    "min_delta": "Min delta",
    # PPO-specific
    "rollout_size": "Rollout size",
    "gae_lambda": "GAE lambda (λ)",
    "clip_epsilon": "Clip epsilon (ε)",
    "update_epochs": "Update epochs",
}

SMOOTHING_WINDOW = 50   # Episodes to smooth over for training curve plots

# Load the results from run_experiments.py
def load_results(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    results_csv = run_dir / "experiment_results.csv"
    curves_csv  = run_dir / "training_curves.csv"

    if not results_csv.exists():
        raise FileNotFoundError(f"experiment_results.csv not found in {run_dir}")

    results_df = pd.read_csv(results_csv)
    curves_df  = pd.read_csv(curves_csv) if curves_csv.exists() else None
    return results_df, curves_df

# Detect which parameters vary across the experiments to know which plots to make
def detect_varying_params(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return (column, label) for every known parameter that has >= 2 distinct values."""
    return [
        (col, label)
        for col, label in PARAM_LABELS.items()
        if col in df.columns and df[col].dropna().nunique() >= 2
    ]

### Some helper functions:

def smooth(values: list[float], window: int) -> np.ndarray:
    """averages the last window values
    """
    result = np.empty(len(values))
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result[i] = np.mean(values[start : i + 1])
    return result


def save_fig(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_parameter_effect(df: pd.DataFrame, param: str, param_label: str,
                          out_dir: Path) -> None:
    """Grouped bar chart showing how param values affect each eval metric

    One subplot per metric, bars grouped by agent
    """
    sub = df[df[param].notna()].copy()
    if sub.empty:
        return

    param_values = sorted(sub[param].unique())
    agents       = sorted(sub["agent"].unique())
    metrics      = [(col, lbl) for col, lbl in EVAL_METRICS.items() if col in df.columns]
    if not metrics:
        return

    n_metrics = len(metrics)
    n_values  = len(param_values)
    colors    = plt.cm.tab10.colors
    bar_width = 0.7 / n_values
    x         = np.arange(len(agents))

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    fig.suptitle(f"Effect of {param_label}", fontsize=12, fontweight="bold")

    for ax_idx, (ax, (metric_col, metric_label)) in enumerate(zip(axes, metrics)):
        # Average over all other hyperparameters so only param drives the difference
        agg = (
            sub.groupby(["agent", param])[metric_col]
            .mean()
            .reset_index()
        )

        for i, val in enumerate(param_values):
            val_rows = agg[agg[param] == val]
            heights  = [
                val_rows.loc[val_rows["agent"] == a, metric_col].values[0]
                if not val_rows.loc[val_rows["agent"] == a].empty else np.nan
                for a in agents
            ]
            offset = (i - (n_values - 1) / 2) * bar_width
            ax.bar(x + offset, heights, width=bar_width, label=str(val),
                   color=colors[i % len(colors)], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(agents, fontsize=9)
        ax.set_title(metric_label, fontsize=10)
        ax.set_ylabel("Mean value", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Legend only on the last subplot
        if ax_idx == len(axes) - 1:
            ax.legend(title=param_label, fontsize=8)

    save_fig(fig, out_dir / f"effect_{param}.png")


# Plot training curves for each varying parameter, grouped by (agent, space):

def plot_training_curves(curves_df: pd.DataFrame, results_df: pd.DataFrame,
                         varying_params: list[tuple[str, str]], out_dir: Path) -> None:
    """For each varying parameter, plot mean training curve per param value.

    The shaded area shows the std across different runs with the same param value
    Curves are grouped by (space, agent) to compare combinations
    """
    if curves_df is None or curves_df.empty:
        print("  No training curve data found, skipping.")
        return

    param_names = [p for p, _ in varying_params]
    config_cols = ["exp_id"] + [p for p in param_names if p in results_df.columns]
    enriched    = curves_df.merge(results_df[config_cols], on="exp_id", how="left")

    colors = plt.cm.tab10.colors
    spaces = enriched["space"].dropna().unique() if "space" in enriched.columns else [None]
    agents = enriched["agent"].dropna().unique() if "agent" in enriched.columns else [None]

    for space in spaces:
        for agent in agents:
            base_mask = pd.Series(True, index=enriched.index)
            if space is not None:
                base_mask &= enriched["space"] == space
            if agent is not None:
                base_mask &= enriched["agent"] == agent
            sub = enriched[base_mask]
            if sub.empty:
                continue

            for param, param_label in varying_params:
                if param not in sub.columns:
                    continue

                unique_vals = sorted(sub[param].dropna().unique())
                color_map   = {v: colors[i % len(colors)] for i, v in enumerate(unique_vals)}

                fig, ax = plt.subplots(figsize=(10, 5))

                for val in unique_vals:
                    val_sub      = sub[sub[param] == val]
                    all_smoothed = []

                    for _, group in val_sub.groupby("exp_id"):
                        rewards  = group.sort_values("episode")["episode_reward"].tolist()
                        all_smoothed.append(smooth(rewards, SMOOTHING_WINDOW))

                    if not all_smoothed:
                        continue

                    # Align curves by shortest length and plot mean std
                    min_len    = min(len(s) for s in all_smoothed)
                    arr        = np.array([s[:min_len] for s in all_smoothed])
                    mean_curve = arr.mean(axis=0)
                    std_curve  = arr.std(axis=0)
                    episodes   = np.arange(min_len)

                    color = color_map[val]
                    n_runs = arr.shape[0]
                    ax.fill_between(episodes,
                                    mean_curve - std_curve,
                                    mean_curve + std_curve,
                                    color=color, alpha=0.2)
                    ax.plot(episodes, mean_curve, color=color, linewidth=2.0,
                            label=f"{param_label} = {val} (n={n_runs})")

                from matplotlib.patches import Patch
                handles, labels = ax.get_legend_handles_labels()
                handles.append(Patch(facecolor="gray", alpha=0.3))
                labels.append("± std across experiments")
                ax.legend(handles=handles, labels=labels, fontsize=9)

                ax.set_xlabel("Episode")
                ax.set_ylabel(f"Reward (smoothed, window = {SMOOTHING_WINDOW})")
                ax.set_title(f"Training curves by {param_label} — {agent} on {space}")
                ax.grid(linestyle="--", alpha=0.3)

                fname = f"training_curves_{param}_{agent}_{space}.png".replace(" ", "_")
                save_fig(fig, out_dir / fname)

# Save the best config per (agent, space) to csv:

def save_best_configs(df: pd.DataFrame, out_dir: Path) -> None:
    """Save the best-performing config per (agent, space) to CSV."""
    primary = "eval_cumulative_reward" # best is based on highest eval cum. reward
    if primary not in df.columns: # if not available, skip
        return

    param_cols  = [p for p in PARAM_LABELS if p in df.columns]
    metric_cols = [col for col in EVAL_METRICS if col in df.columns]

    rows = []
    for (agent, space), group in df.groupby(["agent", "space"]):
        best = group.loc[group[primary].idxmax()]
        rows.append({"agent": agent, "space": space,
                     **{p: best[p] for p in param_cols if pd.notna(best.get(p))},
                     **{m: round(best[m], 3) for m in metric_cols if pd.notna(best.get(m))}})

    path = out_dir / "best_configs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved {path.name}  ({len(rows)} rows)")


def visualize_plot_matrix(results_df: pd.DataFrame, varying: list[tuple[str, str]],
                             out_dir: Path) -> None:
    # Get the best hyperparameter settings for each agent
    best_df = results_df.copy().loc[results_df.copy().groupby("agent")["eval_cumulative_reward"].idxmax()]
    agents = best_df["agent"].copy().unique().tolist()
    colors = ["blue", "orange"]
    markers = ["o", "^"]
    spaces = results_df["space"].copy().unique().tolist()

    fig, axes = plt.subplots(nrows=len(spaces), ncols=len(varying), figsize=(10, 5))
    # Share x within columns
    for j in range(len(varying)):
        master = axes[0, j]
        for i in range(1, len(spaces)):
            axes[i, j].sharex(master)

    # Share y within rows
    for i in range(len(spaces)):
        master = axes[i, 0]
        for j in range(1, len(varying)):
            axes[i, j].sharey(master)

    for i in range(len(spaces)):
        for j in range(len(varying)):
            for agent in agents:
                data = results_df.copy()
                data = data[(data["agent"] == agent) & (data["space"] == spaces[i])]

                for hyp in varying:
                    if hyp != varying[j]:
                        best = best_df[best_df["agent"] == agent][hyp[0]].tolist()[0]
                        if pd.notna(best):
                            data = data[data[hyp[0]] == best]

                if not pd.notna(data[varying[j][0]].tolist()[0]):
                    continue

                x = data[varying[j][0]].tolist()
                y = data["eval_cumulative_reward"].tolist()

                axes[i, j].plot(np.arange(len(x)), y, color=colors[agents.index(agent)],
                                          label=agent, marker=markers[agents.index(agent)])
                axes[i, j].set_xticks(np.arange(len(x)))
                axes[i, j].set_xticklabels(x)
                axes[0, j].set_title(varying[j][1])
                axes[len(spaces)-1, j].set_xlabel("Value")
                axes[i, 0].set_ylabel(f"{spaces[i]}\nCum. Reward")

                if best_df[best_df["agent"] == agent][varying[j][0]].tolist()[0] in x:
                    axes[i, j].axvline(
                        x.index(best_df[best_df["agent"] == agent][varying[j][0]].tolist()[0]),
                        color=colors[agents.index(agent)],
                        linestyle="--",
                        alpha=0.3
                    )

    for ax in axes[:-1, :].flat:
        ax.tick_params(labelbottom=False)

    for ax in axes[:, 1:].flat:
        ax.tick_params(labelleft=False)

    legend_handles = []

    for k, agent in enumerate(agents):
        legend_handles.append(
            Line2D([0], [0],
                   color=colors[k],
                   marker=markers[k],
                   linestyle='-',
                   label=agent)
        )

        legend_handles.append(
            Line2D([0], [0],
                   color=colors[k],
                   linestyle='--',
                   alpha=0.3,
                   label=f"{agent} default")
        )

    fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles), frameon=False,
               bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Agent's Cumulative Reward by Space Layout and Parameter", y=0.98, size="xx-large")
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_fig(fig, out_dir / f"plot_matrix.png", False)


def main(run_dir_arg: str) -> None:
    run_dir = Path(run_dir_arg)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading results from: {run_dir}")
    results_df, curves_df = load_results(run_dir)
    print(f"  {len(results_df)} experiments loaded")

    varying = detect_varying_params(results_df)
    print(f"  Varying parameters: {[p for p, _ in varying]}\n")

    print("Generating parameter effect plots...")
    for param, label in varying:
        plot_parameter_effect(results_df, param, label, out_dir)

    print("\nGenerating training curve plots...")
    plot_training_curves(curves_df, results_df, varying, out_dir)

    print("\nSaving best configs...")
    save_best_configs(results_df, out_dir)

    visualize_plot_matrix(results_df, varying, out_dir)

    print(f"\nAll outputs saved in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("run_dir", help="Path to experiment run directory")
    args = parser.parse_args()
    main(args.run_dir)