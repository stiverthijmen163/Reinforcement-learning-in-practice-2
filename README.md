
Welcome to Data Intelligence Challenge-2AMC15!
This is the repository containing the challenge environment code.

## Quickstart

1. Create a virtual environment for this course with Python >= 3.10. Using conda, you can do: `conda create -n dic2025 python=3.11`. Use `conda activate dic2025` to activate it `conda deactivate` to deactivate it.
2. Clone this repository into the local directory you prefer `git clone https://github.com/RL-In-Practice/2AMC15-2026.git`.
3. Install the required packages `pip install -r requirements.txt`. Now, you are ready to use the simulation environment! :partying_face:	
4. Run `$ python train.py spaces/easy_space.pickle` to start training!

`train.py` is just an example training script. Inside this file, initialize the agent you want to train and evaluate. Feel free to modify it as necessary. Its usage is:

```bash
usage: train.py [-h] [--no_gui] [--sigma SIGMA] [--fps FPS] [--iter ITER]
                [--random_seed RANDOM_SEED] [--start_pos START_POS]
                GRID [GRID ...]

DIC Reinforcement Learning Trainer.

positional arguments:
  GRID                  Paths to the grid file to use. There can be more than
                        one.
options:
  -h, --help                 show this help message and exit
  --no_gui                   Disables rendering to train faster (boolean)
  --sigma SIGMA              Sigma value for the stochasticity of the environment. (float, default=0.1, should be in [0, 1])
  --fps FPS                  Frames per second to render at. Only used if no_gui is not set. (int, default=30)
  --iter ITER                Number of iterations to go through. Should be integer. (int, default=1000)
  --random_seed RANDOM_SEED  Random seed value for the environment. (int, default=0)
  --start_pos START_POS      Agent start position as col,row (e.g. 2,3). If not set, the GUI lets you click to place it. In no_gui mode, defaults to random placement.
```

## Report Results

Commands to reproduce the runs and figures used in the report.

### General Experiments

_(commands and `config.py` settings for the main experiment sweep — TODO)_

### Symmetric Ablation

Tests whether an agent that only sees LiDAR `sensors` can still solve the task when the room is mirror-symmetric, compared against `obs_mode="both"`. Produces `trajectories_overlay_split.png`, the only figure from this test used in the report. This script doesn't read `evaluation/config.py`, its hyperparameters are set directly in `run_symmetric_ablation.py`.

1. Build the room (`spaces/symmetric_2_space.pickle` is already committed, so this step is only needed if you want to regenerate it):

```bash
cd world
python create_symmetric_2_space.py
cd ..
```

2. Train all 4 agent/obs_mode combinations:

```bash
python -m evaluation.run_symmetric_ablation --run --space spaces/symmetric_2_space.pickle --run-dir results/experiments/symmetric_ablation_2
```

3. Generate the plots:

```bash
python -m evaluation.plot_eval
```

This writes both `trajectories_overlay.{pdf,png}` and `trajectories_overlay_split.{pdf,png}` to `results/eval_plots/symmetric_ablation_2/`. Only `trajectories_overlay_split` is the one used in the report.

## Code guide

The code is made up of 2 modules: 

1. `agent`
2. `world`

### The `agent` module

The `agent` module contains the `BaseAgent` class as well as some benchmark agents you may want to test against.

The `BaseAgent` is an abstract class and all RL agents for DIC must inherit from/implement it.
If you know/understand class inheritence, skip the following section:

#### `BaseAgent` as an abstract class
Here you can find an explanation about abstract classes [Geeks for Geeks](https://www.geeksforgeeks.org/abstract-classes-in-python/).

Think of this like how all models in PyTorch start like 

```python
class NewModel(nn.Module):
    def __init__(self):
        super().__init__()
    ...
```

In this case, `NewModel` inherits from `nn.Module`, which gives it the ability to do back propagation, store parameters, etc. without you having to manually code that every time.
It also ensures that every class that inherits from `nn.Module` contains _at least_ the `forward()` method, which allows a forward pass to actually happen.

In the case of your RL agent, inheriting from `BaseAgent` guarantees that your agent implements `update()` and `take_action()`.
This ensures that no matter what RL agent you make and however you code it, the environment and training code can always interact with it in the same way.
Check out the benchmark agents to see examples.

### The `world` module

The world module contains the environment itself:
- `space.py` — defines a room layout (bounds, obstacles, start, target) and saves/loads it as a `.pickle` file
- `environment.py` — the `Environment` class the agent acts in
- `state.py` — builds the agent's observation (`xy`, LiDAR `sensors`, or `both`)
- `helpers.py` — actions, collision checks, and other geometry helpers
- `debug_viewer.py` — live pygame visualization (see Debug Viewer below)
- `create_*_space.py` — scripts that build the space files under `spaces/`

#### The Environment

The `Environment` is very important because it contains everything we hold dear, including ourselves [^1].
It is also the name of the class which our RL agent will act within. Most of the action happens in there.

The main interaction with `Environment` is through the methods:

- `Environment()` to initialize the environment
- `reset()` to reset the environment
- `step()` to actually take a time step with the environment
- `Environment().evaluate_agent()` to evaluate the agent after training.

[^1]: In case you missed it, this sentence is a joke. Please do not write all your code in the `Environment` class.

## Running Agents and Experiments

Below are example commands for running each implemented agent script.

### DQN Agent

Train a single DQN agent on a space:

```bash
python train_dqn.py spaces/easy_space.pickle --episodes 1000 --sigma 0.1
```

The agent can observe its `xy` position, 8 LiDAR-style sensor readings, or both (`--obs_mode`). Some other useful flags:

- `--obs_mode {xy,sensors,both}` — state of the agent, what it observes
- `--gui` — open the live debug viewer while training (see below)
- `--save_model` — save the trained weights after training
- `--early_stop` — stop once the greedy evaluation reward has converged

### Debug Viewer

Pass `--gui` to `train_dqn.py` to open a live pygame window while training:

```bash
python train_dqn.py spaces/easy_space.pickle --gui
```

It shows the agent's path, its LiDAR rays, and a HUD with episode/step/reward/epsilon. Controls: `Space` pause/resume, `T`/`C` auto-pause on target reached / on collision, `← →` scrub back through recent steps, drag the speed slider to change fps.

### Evaluating a Saved Model — `evaluation/evaluate_model.py`

Load a previously saved `.pt` model and run a greedy evaluation episode on any space. The agent type and `obs_mode` are read automatically from the model file, so you do not need to specify them:

```bash
python -m evaluation.evaluate_model results/saved_models/dqn_20260617.pt spaces/u_path_space.pickle
```

To override the starting position:

```bash
python -m evaluation.evaluate_model results/saved_models/dqn_20260617.pt spaces/u_path_space.pickle --start_pos 5.0,6.0
```

Passing a directory instead of a single file evaluates every `.pt` file in that directory:

```bash
python -m evaluation.evaluate_model results/saved_models/ spaces/u_path_space.pickle
```

Some other useful flags:

- `--sigma SIGMA` — environment stochasticity during evaluation (default: `0.0`)
- `--max_steps N` — maximum steps per evaluation episode (default: `250`)
- `--sensor_range R` — sensor range, must match training (default: `10.0`)
- `--random_seed SEED` — random seed for reproducibility (default: `0`)
- `--no_image` — skip saving path and heatmap images
- `--save_path PATH` — directory to save images (default: `results/eval_saved_models/`)

### Inspecting Spaces

Print the contents of every space file in `spaces/` and view them side by side:

```bash
python spaces/inspect_spaces.py
```

For each space it prints the bounds, start position, target position, and obstacle list, and opens a matplotlib window with one panel per space.

---

## Evaluation System

The `evaluation/` folder contains scripts for running many experiments at once and analyzing the results.

### 1. Configure experiments — `evaluation/config.py`

Edit this file to define what to run. The key settings are:

- **`SPACES`** — which space files to train on
- **`AGENTS`** — which agents to run
- Hyperparameter lists — any parameter with multiple values will be swept (all combinations are run)
- **`SAVE_IMAGES = False`** — set `True` to save path and heatmap images per experiment
- **`VERBOSE = False`** — set `True` to see full training output; `False` shows only the progress bar

### 2. Run experiments — `evaluation/run_experiments.py`

```bash
# Run all experiments in parallel (fastest)
python -m evaluation.run_experiments

# Run sequentially (useful for debugging)
python -m evaluation.run_experiments --sequential
```

Results are saved to `results/experiments/<timestamp>/`:
- `experiment_results.csv` — one row per experiment with config + final eval metrics
- `training_curves.csv` — episode reward and length per episode per experiment
- `agent_paths.csv` — agent (x, y) position per step per experiment
- `collisions.csv` — collision status per step per experiment
- `exp_NNNN_heatmap.pdf` / `exp_NNNN.pdf` — path and heatmap images (if `SAVE_IMAGES = True`)

### 3. Analyze results — `evaluation/analyze_results.py`

```bash
python -m evaluation.analyze_results results/experiments/<timestamp>
```

Output is saved to `results/experiments/<timestamp>/analysis/`:
- `effect_<param>.png` — bar charts showing how each varying hyperparameter affects eval metrics
- `training_curves_<param>_<agent>_<space>.png` — mean training curves with ± std band per parameter value
- `best_configs.csv` — best-performing config per (agent, space) combination

### 4. Compare obs_mode and stochasticity — `evaluation/analyze_results_v2.py`

This script is specifically designed to compare how **observation mode** (`xy`, `sensors`, `both`) and **stochasticity** (`sigma`) affect agent behaviour. It requires `SAVE_MODELS = True` in `config.py` so that saved model files are available for path rendering.

```bash
python -m evaluation.analyze_results_v2 results/experiments/<timestamp>
```

By default, the best-performing `obs_mode` per agent is used for the sigma and convergence plots. You can override this with `--obs_mode`:

```bash
python -m evaluation.analyze_results_v2 results/experiments/<timestamp> --obs_mode both
```

Other useful flags:

- `--obs_mode {xy,sensors,both}` — fix a specific observation mode for the sigma/convergence plots instead of using the best one per agent
- `--show_episodes_until N` — clip the x-axis of convergence plots at episode `N`

Output is saved to `results/experiments/<timestamp>/analysis/`:
- `paths_obs_modes.pdf` — overlaid paths for the best model per `obs_mode`, evaluated with σ=0
- `paths_sigma.pdf` — overlaid paths across all sigma values, using the best `obs_mode` per agent
- `convergence_sigma.pdf` — smoothed training reward per episode, one curve per sigma value
- `convergence_obs_modes.pdf` — smoothed training reward per episode, one curve per `obs_mode`
- `convergence_combined.pdf` — both sigma and obs_mode effects in one plot per agent

### 5. Symmetric Ablation — `evaluation/run_symmetric_ablation.py`, `aggregate_eval.py`, `plot_eval.py`

A symmetric room (`symmetric_2_space.pickle`) has two mirrored starting positions. An agent using only the LiDAR `sensors` observation can't tell the two starts apart, since the readings look identical from both sides. This checks whether that actually breaks the agent, by comparing it against `obs_mode="both"`, which can tell the difference.

Train all agent/obs_mode combinations for the ablation, alternating training episodes between the two mirrored starts:

```bash
python -m evaluation.run_symmetric_ablation --run --space spaces/symmetric_2_space.pickle --run-dir results/experiments/symmetric_ablation_2
```

Run without `--run` first to see the planned experiments without training anything. `evaluation/aggregate_eval.py` then evaluates a saved model from both starting positions (and multiple seeds) and collects the results, success rate, steps to goal, full paths, into a DataFrame. `evaluation/plot_eval.py` plots those trajectories on top of the room so you can see by eye whether the agent reaches the target from both starts:

```bash
python -m evaluation.plot_eval
```

