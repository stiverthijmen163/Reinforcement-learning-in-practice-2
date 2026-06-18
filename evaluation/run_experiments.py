"""
Run all experiments defined in evaluation/config.py and save results to csv

Usage:  
    python -m evaluation.run_experiments               # parallel (a lot faster)
    python -m evaluation.run_experiments --sequential  # sequential (can be nice to debug)

Output in results/experiments/<timestamp>/:
    experiment_results.csv has one row per experiment: config + final eval metrics
    training_curves.csv has one row per episode reward/length per experiment

To disable images (path plots + heatmaps), set SAVE_IMAGES = False in config.py.
To suppress all training output and only see experiment progress, set VERBOSE = False.

For adding a new agent:
1. Add "agent_name" to AGENTS in config.py
2. Import its train module below and add to AGENT_TRAINERS
3. Add the code for it in build_experiments()
4. Add the code for it in run_experiment()
"""

import contextlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add world/ to path so environment imports work in subprocesses
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.config import Config
from world.reward_functions import REWARD_FUNCTIONS


### Agent train functions
# To add a new agent: import its module and add an entry here
import train_dqn
import train_ppo

AGENT_TRAINERS = {
    "dqn": train_dqn.main,
    "ppo": train_ppo.main,
}


### Experiment builder

def build_experiments() -> list[dict]:
    """Build all (agent, space, hyperparams) combinations from Config.

    Each returned dict is a flat set of parameters for one training run.
    """
    experiments = []

    for space_path, space_cfg in Config.SPACES.items():
        for agent in Config.AGENTS:
            shared = {
                "agent":      agent,
                "space_path": Path(space_path),
                "start_pos":  space_cfg["start_pos"],
            }

            if agent == "dqn":
                for (sigma, episodes, max_steps, lr, gamma,
                     batch_size, replay_cap, target_upd_freq,
                     epsilon, min_epsilon, anneal_steps, obs_mode, reward_scale, reward_fn,
                ) in product(
                    Config.SIGMAS,
                    Config.EPISODES,
                    Config.MAX_STEPS,
                    Config.LEARNING_RATES,
                    Config.GAMMAS,
                    Config.BATCH_SIZES,
                    Config.REPLAY_CAPACITIES,
                    Config.TARGET_UPDATE_FREQS,
                    Config.EPSILONS,
                    Config.MIN_EPSILONS,
                    Config.EPSILON_ANNEAL_STEPS,
                    Config.OBS_MODES,
                    Config.REWARD_SCALES,
                    Config.REWARD_FNs
                ):
                    experiments.append({
                        **shared,
                        "sigma":                sigma,
                        "episodes":             episodes,
                        "max_steps":            max_steps,
                        "learning_rate":        lr,
                        "gamma":                gamma,
                        "batch_size":           batch_size,
                        "replay_capacity":      replay_cap,
                        "target_update_freq":   target_upd_freq,
                        "epsilon":              epsilon,
                        "min_epsilon":          min_epsilon,
                        "epsilon_anneal_steps": anneal_steps,
                        "obs_mode":             obs_mode,
                        "reward_scale":         reward_scale,
                        "reward_fn":            reward_fn
                    })

            elif agent == "ppo":
                for (sigma, episodes, max_steps, lr, gamma,
                     batch_size, rollout_size, gae_lambda,
                     clip_epsilon, update_epochs, obs_mode, reward_fn
                ) in product(
                    Config.SIGMAS,
                    Config.EPISODES,
                    Config.MAX_STEPS,
                    Config.LEARNING_RATES,
                    Config.GAMMAS,
                    Config.BATCH_SIZES,
                    Config.ROLLOUT_SIZES,
                    Config.GAE_LAMBDAS,
                    Config.CLIP_EPSILONS,
                    Config.UPDATE_EPOCHS,
                    Config.OBS_MODES,
                    Config.REWARD_FNs
                ):
                    experiments.append({
                        **shared,
                        "sigma":         sigma,
                        "episodes":      episodes,
                        "max_steps":     max_steps,
                        "learning_rate": lr,
                        "gamma":         gamma,
                        "batch_size":    batch_size,
                        "rollout_size":  rollout_size,
                        "gae_lambda":    gae_lambda,
                        "clip_epsilon":  clip_epsilon,
                        "update_epochs": update_epochs,
                        "obs_mode":      obs_mode,
                        "reward_fn":     reward_fn
                    })

    return experiments


### Single experiment runner

def run_experiment(experiment: dict, run_dir: Path, exp_id: int) -> tuple[dict, list[dict]]:
    """Train and evaluate one agent config

    Returns:
        summary_row: config + final eval metrics (one row for experiment_results.csv)
        curve_rows:  per-episode data (rows for training_curves.csv)
    """
    agent   = experiment["agent"]
    train   = AGENT_TRAINERS[agent]

    # Format start_pos: train_dqn.main expects "x,y" string or None
    start_pos = experiment["start_pos"]
    if start_pos is not None and not isinstance(start_pos, str):
        start_pos = f"{start_pos[0]},{start_pos[1]}"

    if agent == "dqn":
        results = train(
            grid_paths           = [experiment["space_path"]],
            no_gui               = True,
            sigma                = experiment["sigma"],
            fps                  = 30,
            random_seed          = Config.RANDOM_SEED,
            start_pos            = start_pos,
            episodes             = experiment["episodes"],
            max_steps            = experiment["max_steps"],
            learning_rate        = experiment["learning_rate"],
            gamma                = experiment["gamma"],
            epsilon              = experiment["epsilon"],
            min_epsilon          = experiment["min_epsilon"],
            epsilon_anneal_steps = experiment["epsilon_anneal_steps"],
            batch_size           = experiment["batch_size"],
            replay_capacity      = experiment["replay_capacity"],
            target_update_freq   = experiment["target_update_freq"],
            eval_freq            = Config.EVAL_FREQ,
            eval_episodes        = Config.EVAL_EPISODES,
            obs_mode             = experiment["obs_mode"],
            reward_scale         = experiment["reward_scale"],
            save_path            = run_dir,
            save_image           = Config.SAVE_IMAGES,
            save_model           = Config.SAVE_MODELS,
            experiment_name      = f"exp_{exp_id:04d}",
            reward_fn            = REWARD_FUNCTIONS[experiment["reward_fn"]],
        )

    elif agent == "ppo":
        results = train(
            grid_paths           = [experiment["space_path"]],
            no_gui               = True,
            sigma                = experiment["sigma"],
            fps                  = 30,
            random_seed          = Config.RANDOM_SEED,
            start_pos            = start_pos,
            episodes             = experiment["episodes"],
            max_steps            = experiment["max_steps"],
            lr                   = experiment["learning_rate"],
            gamma                = experiment["gamma"],
            gae_lambda           = experiment["gae_lambda"],
            clip_epsilon         = experiment["clip_epsilon"],
            update_epochs        = experiment["update_epochs"],
            batch_size           = experiment["batch_size"],
            rollout_size         = experiment["rollout_size"],
            eval_freq            = Config.EVAL_FREQ,
            eval_episodes        = Config.EVAL_EPISODES,
            obs_mode             = experiment["obs_mode"],
            save_path            = run_dir,
            save_image           = Config.SAVE_IMAGES,
            save_model           = Config.SAVE_MODELS,
            experiment_name      = f"exp_{exp_id:04d}",
            reward_fn=REWARD_FUNCTIONS[experiment["reward_fn"]],
        )

    results = results or {}
    episode_rewards = results.pop("episode_rewards", [])
    episode_lengths = results.pop("episode_lengths", [])

    summary_row = {
        "exp_id":    exp_id,
        "agent":     experiment["agent"],
        "space":     experiment["space_path"].stem,
        "start_pos": str(experiment["start_pos"]),
        **{k: v for k, v in experiment.items()
           if k not in ("agent", "space_path", "start_pos")},
        **results,
    }

    curve_rows = [
        {
            "exp_id":         exp_id,
            "agent":          experiment["agent"],
            "space":          experiment["space_path"].stem,
            "episode":        ep,
            "episode_reward": reward,
            "episode_length": episode_lengths[ep] if ep < len(episode_lengths) else None,
        }
        for ep, reward in enumerate(episode_rewards)
    ]

    return summary_row, curve_rows


### Parallel worker (runs in subprocess)

def _worker(args: tuple) -> tuple[dict, list[dict]]:
    os.environ["TQDM_DISABLE"] = "1"
    if not Config.VERBOSE:
        # Silence stdout/stderr for the entire subprocess so training prints stay silent
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    i, experiment, run_dir = args
    return run_experiment(experiment, run_dir, i)


### Main function

def main(sequential: bool = False) -> None:
    experiments = build_experiments()
    print(experiments)
    n = len(experiments)
    print(f"Built {n} experiments")
    print(f"Images: {'enabled' if Config.SAVE_IMAGES else 'disabled'} "
          f"(change SAVE_IMAGES in evaluation/config.py)")
    print(f"Verbose: {'enabled' if Config.VERBOSE else 'disabled'} "
          f"(change VERBOSE in evaluation/config.py)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results") / "experiments" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results → {run_dir}\n")

    all_summaries: list[dict] = []
    all_curves:    list[dict] = []

    if sequential:
        for i, exp in enumerate(experiments):
            print(f"[{i + 1}/{n}] Running experiment {i}")
            # Silence training output if VERBOSE=False, but still show progress bar and experiment info:
            # Added this as it was impossible to see how far the training script was with all the outputs
            # So this feels cleaner
            with contextlib.redirect_stdout(open(os.devnull, "w", encoding="utf-8") if not Config.VERBOSE else sys.stdout), \
                 contextlib.redirect_stderr(open(os.devnull, "w", encoding="utf-8") if not Config.VERBOSE else sys.stderr):
                summary, curves = run_experiment(exp, run_dir, i)
            all_summaries.append(summary)
            all_curves.extend(curves)
    else:
        # run in parallel using ProcessPoolExecutor. Each worker runs one experiment and returns its summary + curves
        # This speeds things up a lot!
        worker_args = [(i, exp, run_dir) for i, exp in enumerate(experiments)]
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(_worker, arg): arg[0] for arg in worker_args}
            for future in tqdm(as_completed(futures), total=n, desc="Experiments"):
                try:
                    summary, curves = future.result()
                    all_summaries.append(summary)
                    all_curves.extend(curves)
                except Exception as exc:
                    idx = futures[future]
                    print(f"\nExperiment {idx} failed: {exc}")

    # Save results to CSV
    results_df = pd.DataFrame(all_summaries)
    results_df.to_csv(run_dir / "experiment_results.csv", index=False)
    print(f"\nSaved experiment_results.csv  ({len(results_df)} rows)")

    if all_curves:
        curves_df = pd.DataFrame(all_curves)
        curves_df.to_csv(run_dir / "training_curves.csv", index=False)
        print(f"Saved training_curves.csv     ({len(curves_df)} rows)")

    print(f"\nAnalyze with:")
    print(f"  python -m evaluation.analyze_results {run_dir}")
    print(f"or with   python -m evaluation.analyze_results_v2 {run_dir}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RL experiments from config.py")
    parser.add_argument("--sequential", action="store_true",
                        help="Run sequentially instead of in parallel (useful for debugging)")
    args = parser.parse_args()
    main(sequential=args.sequential)
