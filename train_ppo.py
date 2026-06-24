"""Train PPO Agent.

This file can be used in two ways:
1. Directly from the command line for quick PPO runs.
2. Through evaluation/run_experiments.py, which calls main(...) directly.

The agent receives one of three observation modes:
- xy: robot coordinates
- sensors: 8 lidar sensors
- both: coordinates and sensor readings
"""
import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

sys.path.insert(0, str(Path(__file__).parent / "world"))

from agents.ppo_agent import PPOAgent
from world.environment import Environment
from world.helpers import ACTIONS
from world.state import ObservationBuilder


def parse_args():
    p = ArgumentParser(description="PPO Trainer.")

    p.add_argument("GRID", type=Path,
                   help="Paths to the grid file to use.")
    p.add_argument("--no_gui", action="store_true",
                   help="Open debug viewer during training")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if --gui is set.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--start_pos", type=str, default=None,
                   help="Agent start position as x,y (e.g. 2.0,3.0)")

    # PPO argumenst
    p.add_argument("--episodes", type=int, default=300,
                   help="Number of episodes to train")
    p.add_argument("--max_steps", type=int, default=300,
                   help="Max number of steps to train each episode")

    p.add_argument(
        "--obs_mode", choices=["xy", "sensors", "both"],
        default="both",
        help="Observation mode of the robot. Coordinates, Lidar sensors or both"
    )

    p.add_argument("--sensor_range", type=float, default=10.0,
                   help="Max range of each sensor")

    p.add_argument("--rollout_size", type=int, default=512,
                   help="Number of collected transitions before we perform an update")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="learning rate")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor")
    p.add_argument("--gae_lambda", type=float, default=0.95,
                   help="Controls bias variance tradeoff in advantage estimation")
    p.add_argument("--clip_epsilon", type=float, default=0.2,
                   help="Clipping parameter. Limits how much the policy can change in a single update")
    p.add_argument("--update_epochs", type=int, default=4,
                   help="Reuse the same rollout data N times")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Mini batch size used for updates")
    p.add_argument("--eval_freq", type=int, default=50,
                   help="Evaluate every N episodes")
    p.add_argument("--eval_episodes", type=int, default=10,
                   help="Number of evaluation episodes")
    p.add_argument("--save_model", action="store_true",
                   help="Save trained model weights after training")
    p.add_argument("--early_stop", action="store_true",
               help="Stop training when greedy evaluation reward has converged.")

    return p.parse_args()


def parse_start_pos(start_pos):
    """Parse a command-line start position of the form 'x,y'.

    Returns None when no fixed start position is provided.
    """
    if start_pos is None:
        return None

    x, y = start_pos.split(",")
    return float(x), float(y)


def set_seed(seed):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args):
    set_seed(args.random_seed)

    start_pos = parse_start_pos(args.start_pos)

    env = Environment(
        args.GRID,
        no_gui=not args.gui,
        sigma=args.sigma,
        target_fps=args.fps,
        agent_start_pos=start_pos,
        random_seed=args.random_seed,
    )

    state = env.reset()

    obs_builder = ObservationBuilder(
        env,
        obs_mode=args.obs_mode,
        sensor_range=args.sensor_range,
    )

    state_dim = obs_builder.get_state_dim()
    action_dim = len(ACTIONS)

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
    )

    print("PPO training started")
    print("obs_mode:", args.obs_mode)
    print("state_dim:", state_dim)
    print("action_dim:", action_dim)

    # REWARD SCALING
    reward_scale = 1000.0
    
    recent_rewards = []

    for episode in trange(args.episodes):
        state = env.reset(agent_start_pos=start_pos)
        obs = obs_builder.build(state)

        episode_reward = 0
        done = False

        for step in range(args.max_steps):
            action, log_prob, value = agent.choose_action(obs)

            next_state, reward, terminated, info = env.step(action)

            next_obs = obs_builder.build(next_state)

            done = terminated

            agent.remember(
                state=obs,
                action=action,
                log_prob=log_prob,
                reward=reward / reward_scale, # REWARD SCALING
                done=done,
                value=value,
            )

            episode_reward += reward
            obs = next_obs

            if len(agent.rewards) >= args.rollout_size:
                if done:
                    last_value = 0
                else:
                    last_value = agent.get_value(obs)

                agent.learn(last_value)

            if done:
                break

        if len(agent.rewards) > 0:
            if done:
                last_value = 0
            else:
                last_value = agent.get_value(obs)

            agent.learn(last_value=last_value)

        recent_rewards.append(episode_reward)
        recent_rewards = recent_rewards[-50:]

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(recent_rewards)
            print(
                f"Episode {episode + 1} | "
                f"reward={episode_reward:.1f} | "
                f"avg_reward_50={avg_reward:.1f}"
            )

    return agent

def run_greedy_evaluation(
    space_path,
    agent,
    max_steps,
    eval_episodes,
    obs_mode,
    sensor_range,
    agent_start,
    random_seed,
    reward_fn=None,
):
    """Evaluate the current greedy policy.

    PPO samples actions during training, but for evaluation we use the greedy
    action with the highest policy probability. This makes the evaluation
    deterministic with respect to the learned policy.

    The environment stochasticity is set to sigma=0.0 so that early stopping
    measures whether the learned policy itself has stabilized, not whether
    random drift happened to help or hurt the agent.

    Returns:
        mean evaluation reward and success rate over eval_episodes.
    """
    eval_rewards = []
    successes = []

    rng_state = random.getstate()

    old_training = getattr(agent, "training", None)

    if hasattr(agent, "set_training"):
        agent.set_training(False)

    for eval_idx in range(eval_episodes):
        eval_env = Environment(
            space_path=space_path,
            no_gui=True,
            sigma=0.0,
            agent_start_pos=agent_start,
            target_fps=-1,
            random_seed=random_seed + eval_idx,
            reward_fn=reward_fn,
        )

        eval_env.reset()
        obs_builder = ObservationBuilder(eval_env, obs_mode, sensor_range)
        state = obs_builder.build(eval_env.agent_pos)

        episode_reward = 0.0
        info = {"target_reached": False}

        for _ in range(max_steps):
            action = agent.choose_greedy_action(state)
            _, reward, done, info = eval_env.step(action)

            state = obs_builder.build(eval_env.agent_pos)
            episode_reward += reward

            if done:
                break

        eval_rewards.append(episode_reward)
        successes.append(1.0 if info.get("target_reached", False) else 0.0)

    if old_training is not None and hasattr(agent, "set_training"):
        agent.set_training(old_training)

    random.setstate(rng_state)

    return float(np.mean(eval_rewards)), float(np.mean(successes))


def main(grid_paths, no_gui, sigma, fps, random_seed, start_pos,
         episodes, max_steps, lr, gamma, gae_lambda, clip_epsilon,
         update_epochs, batch_size, rollout_size,
         eval_freq, eval_episodes,
         obs_mode="both", sensor_range=10.0,
         save_path=None, save_image=True, experiment_name=None,
         save_model=False, reward_fn=None, early_stop=False,
         start_pos_pool=None):
    """Train PPO and return metrics used by the experiment pipeline.

    This function can be called by running this file or by
    evaluation/run_experiments.py.

    Args:
        grid_paths: List of environment files.
        no_gui: Whether to disable visualization during training.
        sigma: Probability of stochastic drift during training.
        start_pos: Fixed start position, or None to use the environment default.
        start_pos_pool: Optional list of start positions. If provided, training
            alternates through these start positions.
        obs_mode: One of {"xy", "sensors", "both"}.

    Returns:
        Dictionary containing training rewards, episode lengths, early stopping
        information, and final evaluation metrics.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    agent_start = None
    if start_pos is not None:
        if isinstance(start_pos, str):
            x_str, y_str = start_pos.split(",")
            agent_start = (float(x_str), float(y_str))
        else:
            agent_start = start_pos

    env = Environment(
        space_path=grid_paths[0],
        no_gui=no_gui,
        sigma=sigma,
        agent_start_pos=agent_start,
        target_fps=fps,
        random_seed=random_seed,
        reward_fn=reward_fn,
    )

    obs_builder = ObservationBuilder(env, obs_mode, sensor_range)
    state_dim = obs_builder.get_state_dim()
    action_dim = len(ACTIONS)

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        update_epochs=update_epochs,
        batch_size=batch_size,
    )

    # REWARD SCALING
    reward_scale = 1000.0
    
    # Early stopping constants
    EARLY_STOP_PATIENCE = 10
    EARLY_STOP_DELTA = 1.0
    MIN_EVAL_REWARD = 0.0
    
    best_eval_reward = -float("inf")
    early_stop_counter = 0
    stopped_early = False
    
    episode_rewards = []
    episode_lengths = []
    all_training_positions = []

    for episode in trange(episodes, desc="PPO Training"):
        episode_start = start_pos_pool[episode % len(start_pos_pool)] if start_pos_pool else agent_start
        env.reset(agent_start_pos=episode_start)
        obs = obs_builder.build(env.agent_pos)
        all_training_positions.append(env.agent_pos)

        episode_reward = 0.0
        done = False

        for step in range(max_steps):
            action, log_prob, value = agent.choose_action(obs)
            _, reward, terminated, _ = env.step(action)
            next_obs = obs_builder.build(env.agent_pos)
            all_training_positions.append(env.agent_pos)
            done = terminated

            agent.remember(
                state=obs,
                action=action,
                log_prob=log_prob,
                reward=reward / reward_scale, # REWARD SCALING
                done=done,
                value=value,
            )

            episode_reward += reward
            obs = next_obs

            if len(agent.rewards) >= rollout_size:
                last_value = 0 if done else agent.get_value(obs)
                agent.learn(last_value)

            if done:
                break

        if len(agent.rewards) > 0:
            last_value = 0 if done else agent.get_value(obs)
            agent.learn(last_value=last_value)

        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        # Early stopping and periodic evaluation
        if (episode + 1) % eval_freq == 0:
            mean_eval, success_rate = run_greedy_evaluation(
                space_path=grid_paths[0],
                agent=agent,
                max_steps=max_steps,
                eval_episodes=eval_episodes,
                obs_mode=obs_mode,
                sensor_range=sensor_range,
                agent_start=agent_start,
                random_seed=random_seed,
                reward_fn=reward_fn,
            )

            print(
                f"\nEpisode {episode + 1:4d} | "
                f"Eval reward: {mean_eval:8.2f} | "
                f"Success rate: {success_rate:.2f} | "
                f"Train reward (last {eval_freq}): {np.mean(episode_rewards[-eval_freq:]):8.2f}"
            )

            if early_stop:
                if mean_eval <= MIN_EVAL_REWARD:
                    early_stop_counter = 0

                elif mean_eval > best_eval_reward + EARLY_STOP_DELTA:
                    best_eval_reward = mean_eval
                    early_stop_counter = 0

                else:
                    early_stop_counter += 1

                print(
                    f"Early stopping check: "
                    f"{early_stop_counter}/{EARLY_STOP_PATIENCE} "
                    f"(best eval reward: {best_eval_reward:.2f})"
                )

                if early_stop_counter >= EARLY_STOP_PATIENCE:
                    print(
                        f"\nEarly stopping triggered at episode {episode + 1}. "
                        f"Greedy evaluation reward has converged."
                    )
                    stopped_early = True
                    break

    # Move to CPU before eval and save so that evaluate_model (always CPU) reproduces
    # the exact same result as GPU and CPU use different summation orders
    # which can make Q values slightly different and cause differences in evaluation
    agent.model.cpu()
    agent.device = torch.device("cpu")

    eval_stats = Environment.evaluate_agent(
        space_fp=grid_paths[0],
        agent=agent,
        max_steps=max_steps,
        sigma=0.0,
        agent_start_pos=agent_start,
        random_seed=random_seed,
        training_positions=all_training_positions,
        save_path=save_path,
        save_name=experiment_name,
        save_image=save_image,
        obs_mode=obs_mode,
        sensor_range=sensor_range,
    )

    if save_model and save_path is not None:
        models_dir = Path(save_path) / "saved_models"
        models_dir.mkdir(exist_ok=True)
        agent.save(models_dir / f"{experiment_name}.pt", obs_mode=obs_mode)
        print(f"Model saved in {models_dir / f'{experiment_name}.pt'}")

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "stopped_early": stopped_early,
        "best_eval_reward": best_eval_reward,
        "early_stop_counter": early_stop_counter,
        **{f"eval_{k}": v for k, v in eval_stats.items()},
    }


if __name__ == "__main__":
    args = parse_args()
    save_path = None
    experiment_name = None
    if args.save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path("results") / "saved_models"
        save_path.mkdir(parents=True, exist_ok=True)
        experiment_name = f"ppo_{timestamp}"
    main(
        grid_paths=[args.GRID],
        no_gui=args.no_gui,
        sigma=args.sigma,
        fps=args.fps,
        random_seed=args.random_seed,
        start_pos=args.start_pos,
        episodes=args.episodes,
        max_steps=args.max_steps,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
        rollout_size=args.rollout_size,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        obs_mode=args.obs_mode,
        sensor_range=args.sensor_range,
        save_path=save_path,
        experiment_name=experiment_name,
        save_model=args.save_model,
        early_stop=args.early_stop
    )