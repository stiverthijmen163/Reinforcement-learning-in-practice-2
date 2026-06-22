"""
Trains the symmetric-room ablation (agent x obs_mode, seed 0) by reusing
run_experiment() from evaluation.run_experiments, same training code and
manifest format, this just builds the experiment list and drives it.

Dry run (just prints the plan) by default; pass --run to actually train.
See --help for the agent/obs_mode/space/run-dir/early-stop options.

To recreate the 4-cell ablation used for the eval plots (results/eval_plots/
symmetric_ablation_2), including the dqn/both early-stop fix baked into
EARLY_STOP_OVERRIDES below:

    python -m evaluation.run_symmetric_ablation --run \\
        --space spaces/symmetric_2_space.pickle \\
        --run-dir results/experiments/symmetric_ablation_2
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from evaluation.run_experiments import run_experiment, _worker

DEFAULT_SPACE_PATH = Path("spaces/symmetric_room_space.pickle")
DEFAULT_RUN_DIR = Path("results/experiments/symmetric_ablation")
START_POS_POOL = [(5.0, 3.0), (15.0, 3.0)]
SEED = 0
DEFAULT_AGENTS = ["ppo", "dqn"]
DEFAULT_OBS_MODES = ["sensors", "both"]

# Best hyperparameters per agent, matching the configs already used in
# results/experiments/20260621_015914/experiment_results.csv.
PPO_HPARAMS = dict(
    episodes=20000, max_steps=500, learning_rate=0.0001, gamma=0.999,
    batch_size=64, rollout_size=512, gae_lambda=0.99, clip_epsilon=0.2,
    update_epochs=4, reward_fn="default",
)
DQN_HPARAMS = dict(
    episodes=3000, max_steps=500, learning_rate=0.001, gamma=0.98,
    batch_size=64, replay_capacity=50000, target_update_freq=500,
    epsilon=1.0, min_epsilon=0.02, epsilon_anneal_steps=250000,
    reward_scale=100.0, reward_fn="dqn",
)

# Known fix from the symmetric_2_space ablation: DQN/both's periodic eval only ever checks
# the default start (5,3), so it converged and self-stopped (early_stop_counter maxed out)
# while still failing from the held-out mirror start (15,3), the eval metric never saw
# that failure to begin with. Disabling early-stop let training keep going through more of
# its epsilon-anneal exploitation phase, which fixed it (see symmetric_ablation_3 vs _2).
EARLY_STOP_OVERRIDES = {("dqn", "both"): False}


def build_experiments(space_path, agents=DEFAULT_AGENTS, obs_modes=DEFAULT_OBS_MODES, early_stop=None):
    """
    :param early_stop: None (default) applies EARLY_STOP_OVERRIDES per (agent, obs_mode)
                       cell, defaulting to True elsewhere. Pass True/False to force that
                       value uniformly across every included cell instead.
    """
    all_hparams = {"ppo": PPO_HPARAMS, "dqn": DQN_HPARAMS}
    experiments = []
    for agent in agents:
        for obs_mode in obs_modes:
            cell_early_stop = (EARLY_STOP_OVERRIDES.get((agent, obs_mode), True)
                               if early_stop is None else early_stop)
            experiments.append({
                "agent": agent,
                "space_path": space_path,
                "start_pos": None,
                "start_pos_pool": START_POS_POOL,
                "sigma": 0.0,
                "obs_mode": obs_mode,
                "seed": SEED,
                "early_stop": cell_early_stop,
                **all_hparams[agent],
            })
    return experiments


def print_plan(experiments, run_dir):
    print(f"{len(experiments)} runs -> {run_dir}")
    print(f"space={experiments[0]['space_path']}")
    print(f"start_pos_pool={START_POS_POOL} (even episode -> pool[0], odd -> pool[1])\n")
    for i, exp in enumerate(experiments):
        save_path = run_dir / "saved_models" / f"exp_{i:04d}.pt"
        print(f"[{i:2d}] agent={exp['agent']:4s} obs_mode={exp['obs_mode']:8s} seed={exp['seed']:4d} "
              f"sigma={exp['sigma']} episodes={exp['episodes']:6d} early_stop={exp['early_stop']} -> {save_path}")
    print(f"\nManifest with agent/obs_mode/seed per exp_id will be written to: "
          f"{run_dir / 'experiment_results.csv'}")


def main(run: bool, sequential: bool, space_path: Path, run_dir: Path,
         agents=DEFAULT_AGENTS, obs_modes=DEFAULT_OBS_MODES, early_stop=None):
    experiments = build_experiments(space_path, agents=agents, obs_modes=obs_modes, early_stop=early_stop)
    print_plan(experiments, run_dir)

    if not run:
        print("\nDry run only (no training). Re-run with --run to actually train.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    n = len(experiments)
    all_summaries, all_curves, all_paths, all_colls = [], [], [], []

    if sequential:
        for i, exp in enumerate(experiments):
            print(f"\n[{i + 1}/{n}] Running experiment {i}")
            summary, curves, agent_path, coll_path = run_experiment(exp, run_dir, i)
            all_summaries.append(summary)
            all_curves.extend(curves)
            all_paths.extend(agent_path)
            all_colls.extend(coll_path)
    else:
        worker_args = [(i, exp, run_dir) for i, exp in enumerate(experiments)]
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(_worker, arg): arg[0] for arg in worker_args}
            for future in tqdm(as_completed(futures), total=n, desc="Symmetric ablation"):
                try:
                    summary, curves, agent_path, coll_path = future.result()
                    all_summaries.append(summary)
                    all_curves.extend(curves)
                    all_paths.extend(agent_path)
                    all_colls.extend(coll_path)
                except Exception as exc:
                    idx = futures[future]
                    print(f"\nExperiment {idx} failed: {exc}")

    pd.DataFrame(all_summaries).to_csv(run_dir / "experiment_results.csv", index=False)
    print(f"\nSaved {run_dir / 'experiment_results.csv'}  ({len(all_summaries)} rows)")
    if all_curves:
        pd.DataFrame(all_curves).to_csv(run_dir / "training_curves.csv", index=False)
        print(f"Saved {run_dir / 'training_curves.csv'}  ({len(all_curves)} rows)")
    if all_paths:
        pd.DataFrame(all_paths).to_csv(run_dir / "agent_paths.csv", index=False)
    if all_colls:
        pd.DataFrame(all_colls).to_csv(run_dir / "collisions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the symmetric-room perceptual-aliasing ablation")
    parser.add_argument("--run", action="store_true",
                        help="Actually train (default: dry run, prints configs only)")
    parser.add_argument("--sequential", action="store_true",
                        help="Run one experiment at a time instead of in parallel")
    parser.add_argument("--space", type=Path, default=DEFAULT_SPACE_PATH,
                        help=f"Space pickle to train on (default: {DEFAULT_SPACE_PATH})")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                        help=f"Output directory (default: {DEFAULT_RUN_DIR})")
    parser.add_argument("--agents", type=lambda s: s.split(","), default=DEFAULT_AGENTS,
                        help=f"Comma-separated agents to include (default: {','.join(DEFAULT_AGENTS)})")
    parser.add_argument("--obs-modes", type=lambda s: s.split(","), default=DEFAULT_OBS_MODES,
                        help=f"Comma-separated obs_modes to include (default: {','.join(DEFAULT_OBS_MODES)})")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Force early stopping off for every included cell, overriding "
                             f"EARLY_STOP_OVERRIDES (default: apply per-cell overrides, "
                             f"currently {EARLY_STOP_OVERRIDES})")
    args = parser.parse_args()
    main(run=args.run, sequential=args.sequential, space_path=args.space, run_dir=args.run_dir,
         agents=args.agents, obs_modes=args.obs_modes,
         early_stop=False if args.no_early_stop else None)
