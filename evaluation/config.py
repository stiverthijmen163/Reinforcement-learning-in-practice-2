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

    AGENTS = ["dqn"]

    # Shared parameters
    SIGMAS      = [0.0, 0.1]   # Stochasticity
    RANDOM_SEED = 0             # Fixed seed for reproducibility across all runs

    # DQN hyperparameters + perhaps also some for PPO TODO: write this correct 
    EPISODES             = [10, 20] #TODO: Set these normal for real experiments, this was to test quickly
    MAX_STEPS            = [200]
    LEARNING_RATES       = [0.001]
    GAMMAS               = [0.99]
    BATCH_SIZES          = [32]
    REPLAY_CAPACITIES    = [10000]
    TARGET_UPDATE_FREQS  = [1000]
    EPSILONS             = [1.0]
    MIN_EPSILONS         = [0.01]
    EPSILON_ANNEAL_STEPS = [None]   # None: default: episodes × max_steps // 2
    PATIENCE             = [20]
    MIN_DELTA            = [10.0]

    # Evaluation parameters during training
    EVAL_FREQ     = 50    # Run a greedy eval every N training episodes
    EVAL_EPISODES = 10    # Number of greedy episodes per eval checkpoint

    # Set SAVE_IMAGES = False to skip path/heatmap images
    SAVE_IMAGES = False

    # Set VERBOSE = False to suppress all output from training scripts
    # and only show the progress bar from run_experiments.py itself
    VERBOSE = False
