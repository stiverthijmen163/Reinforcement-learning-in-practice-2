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
        # "spaces/easy_space.pickle": {"start_pos": None},
        # "spaces/test_space.pickle": {"start_pos": None},
        # "spaces/restaurant_1_space.pickle": {"start_pos": None}, # 50*50 map
        # "spaces/restaurant_2_space.pickle": {"start_pos": (17.5, 5.0) }
        "spaces/u_path_space.pickle": {"start_pos": None},
    }
    # "dqn" and, or "ppo"
    AGENTS = ["dqn"]

    # Shared parameters
    SIGMAS      = [0.0]   # Stochasticity
    RANDOM_SEED = 0             # Fixed seed for reproducibility across all runs

    # Shared training parameters (used by both DQN and PPO)
    EPISODES      = [10000]  # TODO: set higher for real experiments
    MAX_STEPS     = [250]
    LEARNING_RATES = [0.0001] # 0.001 for PPO worked okay
    GAMMAS        = [0.99]
    BATCH_SIZES   = [128]

    # Observation mode ("xy", "sensors", or "both")
    OBS_MODES = ["both"]

    # DQN-specific hyperparameters
    REPLAY_CAPACITIES    = [30000]
    TARGET_UPDATE_FREQS  = [1000]
    EPSILONS             = [1.0]
    MIN_EPSILONS         = [0.05]
    EPSILON_ANNEAL_STEPS = [1500000]   # None: default: episodes × max_steps // 2
    
    REWARD_SCALES = [100.0]

    # PPO-specific hyperparameters
    ROLLOUT_SIZES  = [2048]
    GAE_LAMBDAS    = [0.95]
    CLIP_EPSILONS  = [0.2]
    UPDATE_EPOCHS  = [4]

    # Evaluation parameters during training
    EVAL_FREQ     = 50    # Run a greedy eval every N training episodes
    EVAL_EPISODES = 10    # Number of greedy episodes per eval checkpoint

    # Set SAVE_IMAGES = False to skip path/heatmap images
    SAVE_IMAGES = True

    # Set VERBOSE = False to suppress all output from training scripts
    # and only show the progress bar from run_experiments.py itself
    VERBOSE = True