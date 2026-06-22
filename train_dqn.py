"""Train DQN Agent.

This script trains a DQN agent to navigate a restaurant environment
using continuous state representation with LiDAR sensors.
"""
import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from tqdm import trange
import numpy as np

# world/helpers.py is imported as "from helpers import *" inside environment.py.
# Adding world/ to sys.path lets that bare import resolve when we run from the project root.
sys.path.insert(0, str(Path(__file__).parent / "world"))

from agents.dqn_agent import DQNAgent
from world.debug_viewer import DebugViewer
from world.environment import Environment
from world.state import ObservationBuilder

# 24 actions: 8 directions × 3 step sizes (0.2, 0.5, 1.0), as defined in world/helpers.py ACTIONS.
N_ACTIONS = 32

def parse_args():
    """Parse command-line arguments."""
    p = ArgumentParser(description="DQN Training for Continuous Delivery Robot")

    # Environment arguments
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to space configuration files")
    p.add_argument("--gui", action="store_true",
                   help="Open debug viewer during training")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Environmental stochasticity (0-1)")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second for rendering")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed for reproducibility")
    p.add_argument("--start_pos", type=str, default=None,
                   help="Start position as x,y floats (e.g., 5.0,6.0)")
    # Training arguments
    p.add_argument("--episodes", type=int, default=1000,
                   help="Number of training episodes")
    p.add_argument("--max_steps", type=int, default=500,
                   help="Maximum steps per episode")

    # DQN hyperparameters
    p.add_argument("--learning_rate", type=float, default=0.001,
                   help="Learning rate for Q-network updates")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor")
    p.add_argument("--epsilon", type=float, default=1.0,
                   help="Initial exploration rate")
    p.add_argument("--min_epsilon", type=float, default=0.01,
                   help="Minimum exploration rate")
    p.add_argument("--epsilon_anneal_steps", type=int, default=None,
                   help="Steps to anneal epsilon (default: episodes × max_steps // 2)")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size for training")
    p.add_argument("--replay_capacity", type=int, default=10000,
                   help="Replay buffer capacity")
    p.add_argument("--target_update_freq", type=int, default=1000,
                   help="Steps between target network updates")
    p.add_argument("--reward_scale", type=float, default=None,
               help="Divide rewards by this before DQN updates. "
                    "If None, defaults to max_steps.")

    # Evaluation arguments
    p.add_argument("--obs_mode", choices=["xy", "sensors", "both"], default="both",
                   help="Observation mode: coordinates, sensors, or both")
    p.add_argument("--eval_freq", type=int, default=50,
                   help="Evaluate every N episodes")
    p.add_argument("--eval_episodes", type=int, default=10,
                   help="Number of evaluation episodes")
    p.add_argument("--save_model", action="store_true",
                   help="Save trained model weights after training")
    p.add_argument("--early_stop", action="store_true",
               help="Stop training when greedy evaluation reward has converged.")
    return p.parse_args()

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
            action = agent.greedy_action(state)
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
         episodes, max_steps, learning_rate, gamma, epsilon,
         min_epsilon, epsilon_anneal_steps, batch_size, replay_capacity, target_update_freq,
         eval_freq, eval_episodes,
         obs_mode="both", sensor_range=10.0,
         save_path=None, save_image=True, experiment_name=None, reward_scale=None,
         save_model=False, reward_fn = None, early_stop=False,
         start_pos_pool=None):
    """Main training loop.

    Extra params for run_experiments.py:
      save_path:        where to save images/results (None = results/)
      save_image:       whether to save path + heatmap images
      experiment_name:  filename stem for saved files (None = timestamp)

    Returns a dict with episode_rewards, episode_lengths, final_epsilon,
    training_steps, and eval_* metrics from the final greedy evaluation.
    """
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Parse continuous start position
    agent_start = None
    if start_pos is not None:
        x_str, y_str = start_pos.split(",")
        agent_start = (float(x_str), float(y_str))

    # Initialise environment
    env = Environment(
        space_path=grid_paths[0],
        no_gui=no_gui,
        sigma=sigma,
        agent_start_pos=agent_start,
        target_fps=fps,
        random_seed=random_seed,
        reward_fn=reward_fn,
    )

    env = DebugViewer(env)
    obs_builder = ObservationBuilder(env, obs_mode, sensor_range)

    # Annealing epsilon over the first half of the training budget so exploration dominates
    # early and exploitation dominates late. Can still be set manually via --epsilon_anneal_steps.
    if epsilon_anneal_steps is None:
        epsilon_anneal_steps = episodes * max_steps // 2

    # Normalise rewards by max_steps so the step penalty stays meaningful relative to
    # the target reward. Dividing by the normal target reward (4000) makes step=-0.00025,
    # which is so small that the Q-values seems to collapse and learning fails.
    if reward_scale is None:
        reward_scale = float(max_steps)

    # Initialise DQN agent
    agent = DQNAgent(
        n_actions=N_ACTIONS,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        min_epsilon=min_epsilon,
        epsilon_anneal_steps=epsilon_anneal_steps,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        input_size=obs_builder.get_state_dim(),
        hidden_size=128
    )

    print("DQN Agent Configuration:")
    print(f"  Obs mode         : {obs_mode} ({obs_builder.get_state_dim()}-D state)")
    print(f"  Actions          : {N_ACTIONS} (8 directions × 3 step sizes)")
    print(f"  Learning rate    : {learning_rate}")
    print(f"  Gamma            : {gamma}")
    print(f"  Initial epsilon  : {epsilon}")
    print(f"  Min epsilon      : {min_epsilon}")
    print(f"  Epsilon anneal   : {epsilon_anneal_steps} steps")
    print(f"  Batch size       : {batch_size}")
    print(f"  Replay capacity  : {replay_capacity}")
    print(f"  Target update    : every {target_update_freq} steps")
    print(f"  Reward scale     : {reward_scale:.0f}")

    episode_rewards = []
    episode_lengths = []

    # Track all positions visited during each episode for heatmap visualization
    all_training_positions = []
    
    # Early stopping constants
    EARLY_STOP_PATIENCE = 10
    EARLY_STOP_DELTA = 1.0
    MIN_EVAL_REWARD = 0.0
    
    best_eval_reward = -float("inf")
    early_stop_counter = 0
    stopped_early = False

    for episode in trange(episodes, desc="Training"):
        episode_start = start_pos_pool[episode % len(start_pos_pool)] if start_pos_pool else agent_start
        env.reset(agent_start_pos=episode_start)
        all_training_positions.append(env.agent_pos)
        state = obs_builder.build(env.agent_pos)
        total_reward = 0.0

        for step in range(max_steps):
            action = agent.take_action(state)
            _, reward, done, _ = env.step(action)
            next_state = obs_builder.build(env.agent_pos)
            all_training_positions.append(env.agent_pos)
            
            #REWARD SCALING
            unscaled_reward = reward
            scaled_reward = reward / reward_scale

            agent.update(state, action, scaled_reward, next_state, done)
            max_q = float(np.max(agent.q_network.forward(state)))
            env.update_metrics(epsilon=agent.epsilon, loss=agent.last_loss, max_q=max_q)
            state = next_state
            total_reward += unscaled_reward

            if done:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)

        # Periodic greedy evaluation and early stopping
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
                f"Train reward (last {eval_freq}): {np.mean(episode_rewards[-eval_freq:]):8.2f} | "
                f"Epsilon: {agent.epsilon:.4f}"
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


    print("\nTraining completed.")
    print(f"Final epsilon    : {agent.epsilon:.4f}")
    print(f"Training steps   : {agent.training_step}")

    agent.set_training(False)
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
        "episode_rewards":  episode_rewards,
        "episode_lengths":  episode_lengths,
        "stopped_early": stopped_early,
        "best_eval_reward": best_eval_reward,
        "early_stop_counter": early_stop_counter,
        "final_epsilon":    agent.epsilon,
        "training_steps":   agent.training_step,
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
        experiment_name = f"dqn_{timestamp}"
    main(
        args.GRID,
        not args.gui,
        args.sigma,
        args.fps,
        args.random_seed,
        args.start_pos,
        args.episodes,
        args.max_steps,
        args.learning_rate,
        args.gamma,
        args.epsilon,
        args.min_epsilon,
        args.epsilon_anneal_steps,
        args.batch_size,
        args.replay_capacity,
        args.target_update_freq,
        args.eval_freq,
        args.eval_episodes,
        obs_mode=args.obs_mode,
        reward_scale=args.reward_scale,
        save_path=save_path,
        experiment_name=experiment_name,
        save_model=args.save_model,
        early_stop=args.early_stop,
    )