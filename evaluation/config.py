"""
Experiment configuration for the continuous environment.

run_experiments.py reads this file and sweeps all combinations of the listed
value. Note that this leads to combinatorial explosion, so watch out adding too many

== How to add a new agent ==
1. Add the agent name string to AGENTS (e.g. "ppo")
2. Import its train module in run_experiments.py and add an entry to AGENT_TRAINERS
3. Add its hyperparameter lists to this Config class
4. Add a build block for it in build_experiments() in run_experiments.py
"""


class Config:

    # start_pos: None means use the position stored in the space file
    SPACES = {
        "spaces/easy_space.pickle": {"start_pos": None},
        # "spaces/test_space.pickle": {"start_pos": None},
    }

    AGENTS = ["dqn", "ppo"]

    # Shared parameters
    SIGMAS      = [0.1]   # Stochasticity
    RANDOM_SEED = 0             # Fixed seed for reproducibility across all runs

    # Shared training parameters (used by both DQN and PPO)
    EPISODES      = [10]  # TODO: set higher for real experiments
    MAX_STEPS     = [200]
    LEARNING_RATES = [0.001]
    GAMMAS        = [0.99]
    BATCH_SIZES   = [32]

    # Observation mode ("xy", "sensors", or "both")
    OBS_MODES = ["xy", "sensors", "both"]

    # DQN-specific hyperparameters
    REPLAY_CAPACITIES    = [10000]
    TARGET_UPDATE_FREQS  = [1000]
    EPSILONS             = [1.0]
    MIN_EPSILONS         = [0.01]
    EPSILON_ANNEAL_STEPS = [None]   # None: default: episodes × max_steps // 2

    # PPO-specific hyperparameters
    ROLLOUT_SIZES  = [512]
    GAE_LAMBDAS    = [0.95]
    CLIP_EPSILONS  = [0.2]
    UPDATE_EPOCHS  = [4]

    # Evaluation parameters during training
    EVAL_FREQ     = 50    # Run a greedy eval every N training episodes
    EVAL_EPISODES = 10    # Number of greedy episodes per eval checkpoint

    # Set SAVE_IMAGES = False to skip path/heatmap images
    SAVE_IMAGES = False

    # Set VERBOSE = False to suppress all output from training scripts
    # and only show the progress bar from run_experiments.py itself
    VERBOSE = False
