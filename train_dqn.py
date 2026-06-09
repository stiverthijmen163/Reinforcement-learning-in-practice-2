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
from world.state import get_state

# 24 actions: 8 directions × 3 step sizes (0.2, 0.5, 1.0), as defined in world/helpers.py ACTIONS.
N_ACTIONS = 24

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
    p.add_argument("--epsilon_anneal_steps", type=int, default=1000000,
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
    p.add_argument("--patience", type=int, default=20,
                   help="Stop if eval reward has not improved for this many evaluations")
    p.add_argument("--min_delta", type=float, default=10.0,
                   help="Minimum improvement in mean eval reward to count as progress")

    return p.parse_args()


def main(grid_paths, no_gui, sigma, fps, random_seed, start_pos,
         episodes, max_steps, learning_rate, gamma, epsilon,
         min_epsilon, epsilon_anneal_steps, batch_size, replay_capacity, target_update_freq,
         eval_freq, eval_episodes, patience, min_delta):
    """Main training loop."""
    
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

    # Track all positions visited during each episode for heatmap visualization
    all_training_positions = []

    # Early stopping state: track the best mean eval reward and how many
    # consecutive evaluations have passed without a meaningful improvement.
    best_eval = -float("inf")
    evals_without_improvement = 0

    for episode in trange(episodes, desc="Training"):
        env.reset()
        all_training_positions.append(env.agent_pos)
        state = get_state(env)
        total_reward = 0.0

        for step in range(max_steps):
            action = agent.take_action(state)
            _, reward, done, _ = env.step(action)
            next_state = get_state(env)
            all_training_positions.append(env.agent_pos)

            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)

        # Periodic greedy evaluation
        if (episode + 1) % eval_freq == 0:
            agent.set_training(False)
            eval_rewards = []

            for _ in range(eval_episodes):
                env.reset()
                s = get_state(env)
                ep_r = 0.0
                for _ in range(max_steps):
                    a = agent.greedy_action(s)
                    _, r, d, _ = env.step(a)
                    s = get_state(env)
                    ep_r += r
                    if d:
                        break
                eval_rewards.append(ep_r)

            agent.set_training(True)
            mean_eval = np.mean(eval_rewards)
            print(
                f"\nEpisode {episode + 1:4d} | "
                f"Eval reward: {mean_eval:8.2f} | "
                f"Train reward (last {eval_freq}): {np.mean(episode_rewards[-eval_freq:]):8.2f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

            # Check for improvement: reset counter if reward improved by at least
            # min_delta, otherwise increment. Stop when patience is exhausted.
            if mean_eval > best_eval + min_delta:
                best_eval = mean_eval
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
                if evals_without_improvement >= patience:
                    print(f"\nEarly stop: no improvement for {patience} evaluations.")
                    break

    print("\nTraining completed.")
    print(f"Final epsilon    : {agent.epsilon:.4f}")
    print(f"Training steps   : {agent.training_step}")

    agent.set_training(False)
    Environment.evaluate_agent(
        space_fp=grid_paths[0],
        agent=agent,
        max_steps=max_steps,
        sigma=0.,
        agent_start_pos=agent_start,
        random_seed=random_seed,
        training_positions=all_training_positions,
    )


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
        args.patience,
        args.min_delta,
    )