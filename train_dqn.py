"""Train DQN Agent.

This script trains a DQN agent to navigate a restaurant environment
using continuous state representation with LiDAR sensors.
"""
import sys
from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np

# world/helpers.py is imported as "from helpers import *" inside environment.py.
# Adding world/ to sys.path lets that bare import resolve when we run from the project root.
sys.path.insert(0, str(Path(__file__).parent / "world"))

from agents.dqn_agent import DQNAgent
from world.environment import Environment

# 24 actions: 8 directions × 3 step sizes (0.2, 0.5, 1.0), as defined in world/helpers.py ACTIONS.
N_ACTIONS = 24


def _reset_env(env: Environment) -> np.ndarray:
    """Reset the environment for a new episode.

    Environment.reset() is not yet implemented, so we initialise
    the required internal state manually.
    """
    env.terminal_state = False
    env.info = {}
    env.world_stats = {
        "total_steps": 0,
        "total_agent_moves": 0,
        "total_collision": 0,
        "cumulative_reward": 0,
        "total_targets_reached": 0,
    }
    env._initialize_agent_pos()
    return _get_state(env)


def _get_state(env: Environment) -> np.ndarray:
    """Build the normalised 10-D state vector [x/W, y/H, d1..d8].

    LiDAR sensor readings (d1..d8) are not yet implemented in environment.py;
    they are set to 1.0 (maximum normalised range) as a placeholder until
    ray-casting is added.
    """
    x, y = env.agent_pos
    sensors = [1.0] * 8
    return np.array([x / env.x_max, y / env.y_max] + sensors, dtype=np.float32)


def parse_args():
    """Parse command-line arguments."""
    p = ArgumentParser(description="DQN Training for Continuous Delivery Robot")

    # Environment arguments
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to space configuration files")
    p.add_argument("--no_gui", action="store_true",
                   help="Disable rendering for faster training")
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
    p.add_argument("--epsilon_anneal_steps", type=int, default=100000,
                   help="Number of training steps to linearly anneal epsilon")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size for training")
    p.add_argument("--replay_capacity", type=int, default=10000,
                   help="Replay buffer capacity")
    p.add_argument("--target_update_freq", type=int, default=1000,
                   help="Steps between target network updates")

    # Evaluation arguments
    p.add_argument("--eval_freq", type=int, default=50,
                   help="Evaluate every N episodes")
    p.add_argument("--eval_episodes", type=int, default=10,
                   help="Number of evaluation episodes")

    return p.parse_args()


def main(grid_paths, no_gui, sigma, fps, random_seed, start_pos,
         episodes, max_steps, learning_rate, gamma, epsilon,
         min_epsilon, epsilon_anneal_steps, batch_size, replay_capacity, target_update_freq,
         eval_freq, eval_episodes):
    """Main training loop.
    
    Args:
        grid_paths: List of grid file paths.
        no_gui: Whether to disable rendering.
        sigma: Environmental stochasticity.
        fps: Frames per second for rendering.
        random_seed: Random seed.
        start_pos: Starting position tuple (or None).
        episodes: Number of training episodes.
        max_steps: Maximum steps per episode.
        learning_rate: DQN learning rate.
        gamma: Discount factor.
        epsilon: Initial exploration rate.
        epsilon_decay: Epsilon decay per episode.
        min_epsilon: Minimum exploration rate.
        batch_size: Batch size for training.
        replay_capacity: Replay buffer capacity.
        target_update_freq: Target network update frequency.
        eval_freq: Evaluation frequency.
        eval_episodes: Number of evaluation episodes.
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
    )

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
        input_size=10,  # Continuous state: [x, y, d1-d8]
        hidden_size=128
    )

    print("DQN Agent Configuration:")
    print(f"  Actions          : {N_ACTIONS} (8 directions × 3 step sizes)")
    print(f"  Learning rate    : {learning_rate}")
    print(f"  Gamma            : {gamma}")
    print(f"  Initial epsilon  : {epsilon}")
    print(f"  Min epsilon      : {min_epsilon}")
    print(f"  Epsilon anneal   : {epsilon_anneal_steps} steps")
    print(f"  Batch size       : {batch_size}")
    print(f"  Replay capacity  : {replay_capacity}")
    print(f"  Target update    : every {target_update_freq} steps")

    episode_rewards = []
    episode_lengths = []

    for episode in trange(episodes, desc="Training"):
        state = _reset_env(env)
        total_reward = 0.0

        for step in range(max_steps):
            action = agent.take_action(state)
            _, reward, done, _ = env.step(action)
            next_state = _get_state(env)

            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)

        # Periodic greedy evaluation
        if (episode + 1) % eval_freq == 0:
            agent.set_training(False)
            eval_rewards = []

            for _ in range(eval_episodes):
                s = _reset_env(env)
                ep_r = 0.0
                for _ in range(max_steps):
                    a = agent.greedy_action(s)
                    _, r, d, _ = env.step(a)
                    s = _get_state(env)
                    ep_r += r
                    if d:
                        break
                eval_rewards.append(ep_r)

            agent.set_training(True)
            print(
                f"\nEpisode {episode + 1:4d} | "
                f"Eval reward: {np.mean(eval_rewards):8.2f} | "
                f"Train reward (last {eval_freq}): {np.mean(episode_rewards[-eval_freq:]):8.2f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    print("\nTraining completed.")
    print(f"Final epsilon    : {agent.epsilon:.4f}")
    print(f"Training steps   : {agent.training_step}")


if __name__ == "__main__":
    args = parse_args()
    main(
        args.GRID,
        args.no_gui,
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
    )