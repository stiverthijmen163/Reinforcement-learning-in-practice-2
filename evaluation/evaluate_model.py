"""
Evaluate a saved model on a space and optionally save path/heatmap images.

Usage:
    python -m evaluation.evaluate_model <model_path> <space_path> [options]

Examples:
    python -m evaluation.evaluate_model results/saved_models/dqn_20260617.pt spaces/u_path_space.pickle
    python -m evaluation.evaluate_model results/saved_models/dqn_20260617.pt spaces/u_path_space.pickle --start_pos 5.0,6.0

Agent type and obs_mode are read automatically from the saved model file

It saves the results and image in results/eval_saved_models/ by default
but can be changed with --save_path
"""

import sys
import torch
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "world"))

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from world.environment import Environment
import os


def load_agent(model_path: Path):
    """Load a DQN or PPO agent from a .pt file"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    agent_type = checkpoint["agent_type"]
    if agent_type == "dqn":
        return DQNAgent.load(model_path)
    elif agent_type == "ppo":
        return PPOAgent.load(model_path)


def evaluate_model(
    model_path: Path,
    space_path: Path,
    sensor_range: float = 10.0,
    max_steps: int = 250, # TODO: need to think about evalaution setings here (use same as training or certain value for evaluation?)
    sigma: float = 0.0,
    start_pos: str = None,
    random_seed: int = 0,
    save_image: bool = False,
    save_path: Path = None,
    save_name: str = None,
) -> dict:
    """Load a saved model and evaluate it on a space

    obs_mode is read automatically from the model file as this is saved
    Returns the stats from evaluate_agent() function
    """
    agent = load_agent(model_path)

    agent_start = None
    if start_pos is not None:
        x, y = start_pos.split(",")
        agent_start = (float(x), float(y))

    print(f"Start position: {agent_start if agent_start is not None else 'from space file'}")

    stats = Environment.evaluate_agent(
        space_fp = space_path,
        agent = agent,
        max_steps = max_steps,
        sigma = 0.,
        agent_start_pos = agent_start,
        random_seed = random_seed,
        save_path = save_path,
        save_name = save_name,
        save_image = save_image,
        obs_mode = agent.obs_mode,
        sensor_range = sensor_range,
    )

    return stats


def parse_args():
    p = ArgumentParser(description="Evaluate a saved DQN or PPO model on a space")
    p.add_argument("model_path", type=Path,
                   help="Path to saved model file (.pt)")
    p.add_argument("space_path", type=Path,
                   help="Path to space .pickle file")
    p.add_argument("--sensor_range", type=float, default=10.0,
                   help="Sensor range (must match training)")
    p.add_argument("--max_steps", type=int, default=250,
                   help="Maximum steps per evaluation episode")
    p.add_argument("--sigma", type=float, default=0.0,
                   help="Environment stochasticity")
    p.add_argument("--start_pos", type=str, default=None,
                   help="Agent start position as x,y (e.g. 5.0,6.0)")
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--no_image", action="store_true",
                   help="Skip saving path and heatmap images")
    p.add_argument("--save_path", type=Path, default=None,
                   help="Directory to save images (default: results/eval/)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    save_image = not args.no_image
    save_path = args.save_path
    if save_image and save_path is None:
        save_path = Path("results") / "eval_saved_models"
        save_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # save_name = f"{args.model_path.stem}_{timestamp}"

    print(f"Model : {args.model_path}")
    print(f"Space : {args.space_path}")
    print(f"sigma={args.sigma}  max_steps={args.max_steps}\n")

    if args.model_path.is_file():
        model_paths = [args.model_path]
        save_names = [args.model_path.stem]
    else:
        model_paths = [Path(os.path.join(str(args.model_path), f)) for f in os.listdir(str(args.model_path))]
        save_names = [Path(f).stem for f in model_paths]

    for model_path, save_name in zip(model_paths, save_names):
        stats = evaluate_model(
            model_path   = model_path,
            space_path   = args.space_path,
            sensor_range = args.sensor_range,
            max_steps    = args.max_steps,
            sigma        = args.sigma,
            start_pos    = args.start_pos,
            random_seed  = args.random_seed,
            save_image   = save_image,
            save_path    = save_path,
            save_name    = save_name,
        )

        for k, v in stats.items():
            print(f"  {k}: {v}")