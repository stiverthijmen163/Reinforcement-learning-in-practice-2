import random
from pathlib import Path
from warnings import warn
from tqdm import trange
from datetime import datetime
from copy import deepcopy
from world.helpers import *
from agents.base_agent import BaseAgent
from world.path_visualizer import visualize_path, visualize_heatmap
from world.space import Space
from world.state import ObservationBuilder


class Environment:
    def __init__(
            self,
            space_path: Path,
            no_gui: bool = False,
            sigma: float = 0,
            agent_start_pos: tuple[float, float] = None,
            reward_fn: callable = None,
            target_fps: int = 30,
            random_seed: int | float | str | bytes | bytearray | None = 0,
            agent_radius: float = 0.1,
            target_radius: float = 0.2,
    ) -> None:
        """
        Initializes the overall space environment used for solving the delivering agent problem.

        :param space_path: path to the file containing the space to play in
        :param no_gui: whether to see the agents moves or not (simulation)
        :param sigma: stochasticity of the environment
        :param agent_start_pos: starting position of the agent within the space
        :param reward_fn: custom reward function to use
        :param target_fps: how fast the simulation should run if the GUI is shown
        :param random_seed: the random seed to use for this environment object
        :param agent_radius: the radius of the agent
        :param target_radius: the radius of the target
        """
        random.seed(random_seed)

        # Check if space layout exists
        if not space_path.exists():
            raise FileNotFoundError(f"Grid {space_path} does not exist.")
        else:
            self.space_fp = space_path

            self.x_max, self.y_max, self.obstacles, self.agent_start_pos, self.target_pos = None, None, None, None, None

        # Initialize up reward function
        if reward_fn is None:
            warn("No reward function provided. Using default reward.")
            self.reward_fn = self._default_reward_function
        else:
            self.reward_fn = reward_fn

        # GUI specific code: Set up the environment as a blank state.
        self.no_gui = no_gui
        if target_fps <= 0:
            self.target_spf = 0.
        else:
            self.target_spf = 1. / target_fps
        self.gui = None

        # Initialize other variables
        self.target_radius = target_radius
        self.terminal_state = False
        self.sigma = sigma
        self.agent_radius = agent_radius
        self.info = {}
        self.world_stats = {}


    def _reset_info(self) -> None:
        """
        Resets the info dictionary, containing information about the last move made.
        """
        self.info = {
            "target_reached": False,
            "agent_moved": False,
            "actual_action": None,
            "collided": False
        }


    def _reset_world_stats(self) -> None:
        """
        Resets the world stats dictionary, containing information about the environment since the last reset.
        """
        self.world_stats = {
            "cumulative_reward": 0,
            "total_steps": 0,
            "total_agent_moves": 0,
            "total_failed_moves": 0,
            "total_targets_reached": 0,
            "total_collision": 0
        }


    def _validate_start_pos(self, pos: tuple[float, float]):
        """
        Validates whether the starting position is valid.

        :param pos: starting position

        :raises ValueError: if the starting position is invalid
        """
        move, _ = get_pos_type(pos, self.agent_radius, self.target_pos, self.target_radius,
                             self.obstacles, (self.x_max, self.y_max))
        if move in [1, 2]:
            raise ValueError(f"Start position {pos} is {'in an obstacle' if move == 1 else 'out of bounds'}. "
                             f"The agent can only start in empty space.")


    def _initialize_agent_pos(self) -> None:
        """
        Initializes the agent's stating position, assigns a random
        starting position if no initial state is provided.
        """
        if self.agent_start_pos:  # Check if the starting position is valid
            self._validate_start_pos(self.agent_start_pos)
            self.agent_pos = self.agent_start_pos
        else:  # Generate random starting position
            # Generate positions until a valid position is found
            valid_starting_pos = False
            while not valid_starting_pos:
                random_pos = (random.uniform(0, self.x_max), random.uniform(0, self.y_max))
                if get_pos_type(random_pos, self.agent_radius, self.target_pos, self.target_radius,
                             self.obstacles, (self.x_max, self.y_max))[0] == 0:
                    valid_starting_pos = True
                    self.agent_pos = random_pos

            print(f"\nNo start position provided. "
                  f"Randomly placed agent at {self.agent_pos}.")
            print(f"To use this position next run: "
                  f"--start_pos {self.agent_pos[0]},{self.agent_pos[1]}")


    def reset(self, **kwargs) -> tuple[float, float]:
        """
        Resets the environment to its initial state, keyword arguments can be provided
        to overwrite the initial arguments provided when initializing the environment.

        :param kwargs: keyword options to overwrite provided initial arguments

        :return: the agent's initial state
        """
        for k, v in kwargs.items():
            # Go through each possible keyword argument.
            match k:
                case "space_fp":
                    self.space_fp = v
                case "agent_start_pos":
                    self.agent_start_pos = v
                case "no_gui":
                    self.no_gui = v
                case "target_fps":
                    self.target_spf = 1. / v
                case "sigma" | "reward_fn" | "random_seed":
                    raise ValueError(f"{k} cannot be changed after initialization.")
                case _:
                    raise ValueError(f"{k} is not one of the possible "
                                     f"keyword arguments.")

        # Reset variables
        space = Space.load_space(self.space_fp)  # Make sure to also set initial state
        self.obstacles = space["obstacles"]
        self.target_pos = space["target_pos"]
        self.x_max, self.y_max = space["bound"]
        self.agent_start_pos = self.agent_start_pos if self.agent_start_pos else space["starting_pos"]
        self.terminal_state = False
        # print("\nRESET\n")
        self._reset_info()
        self._reset_world_stats()

        ########################## IMPLEMENT GUI CODE IFF DESIRED ##########################

        self._initialize_agent_pos()

        return self.agent_pos


    def _move_agent(self, new_pos: tuple[float, float]) -> None:
        """
        Moves the agent to the next position, possibly a mid-move position if an obstacle was hit,
        and updates the corresponding stats.

        :param new_pos: aimed position of the agent after making a move in the form (x, y)
        """
        prev_pos = self.agent_pos
        move, pos_of_hit = get_pos_type(new_pos, self.agent_radius, self.target_pos, self.target_radius,
                               self.obstacles, (self.x_max, self.y_max), True, prev_pos)

        match move:
            case 0:  # Moved into free space
                self.agent_pos = new_pos
                self.info["agent_moved"] = True
                self.world_stats["total_agent_moves"] += 1
                self.info["collided"] = False
            case 1:  # Moved into/against an obstacle
                self.world_stats["total_collision"] += 1
                self.info["collided"] = True

                if prev_pos == pos_of_hit:  # Agent is stuck (something wrong in code)
                    warn(f"Agent seems to be stuck at position {pos_of_hit}!")
                    self.info["agent_moved"] = False
                else:  # Move on the line from prev to next position just before hitting the obstacle
                    self.agent_pos = pos_before_next_pos(prev_pos, pos_of_hit)
                    self.info["agent_moved"] = True
                    self.world_stats["total_agent_moves"] += 1
            case 2:  # Moved out of the space
                self.world_stats["total_collision"] += 1
                self.info["collided"] = True

                if prev_pos == pos_of_hit:  # Agent is stuck (something wrong in code)
                    warn(f"Agent seems to be stuck at position {pos_of_hit}!")
                    self.info["agent_moved"] = False
                else:  # Move on the line from prev to next position just before hitting the wall
                    self.agent_pos = pos_before_next_pos(prev_pos, pos_of_hit, 0.0001)
                    self.info["agent_moved"] = True
                    self.world_stats["total_agent_moves"] += 1
            case 3:  # Moved into the target
                self.agent_pos = new_pos
                self.terminal_state = True
                self.info["target_reached"] = True
                self.world_stats["total_targets_reached"] += 1
                self.info["agent_moved"] = True
                self.world_stats["total_agent_moves"] += 1
                self.info["collided"] = False
            case _:
                raise ValueError(f"Grid is badly formed. It has a value of "
                                 f"{move} at position "
                                 f"{new_pos}.")


    def step(self, action_id: int) -> tuple[tuple[float, float], float, bool, dict]:
        """
        Takes an action.

        :param action_id: id of the action to take

        :return: the position of the agent after making the move,
                 the reward received for this move,
                 whether a terminal state has been reached,
                 and a dictionary containing information regarding the step taken
        """
        self.world_stats["total_steps"] += 1

        ########################## IMPLEMENT GUI CODE IFF DESIRED ##########################

        # Calculate next position based on the action
        action = ACTIONS[action_id]
        next_pos = calc_next_position(self.agent_pos, action[0], action[1])
        actual_action = f"{action_id}"

        # Add stochasticity to the actions
        val = random.random()
        if val <= self.sigma:
            random_dir = (random.uniform(-1, 1), random.uniform(-1, 1))
            random_step_size = random.uniform(0, 0.2)  # Maximum deviation of 0.2
            next_pos = calc_next_position(next_pos, random_dir, random_step_size)
            actual_action += f", and with random move: (({random_dir}), {random_step_size})"

        self.info["actual_action"] = actual_action

        # Calculate the reward for the agent
        reward = self.reward_fn(next_pos, self.agent_pos)

        # Make the move
        self._move_agent(next_pos)
        self.world_stats["cumulative_reward"] += reward

        return self.agent_pos, reward, self.terminal_state, self.info


    # @staticmethod
    def _default_reward_function(self, agent_pos: tuple[float, float], prev_pos: tuple[float, float] = None) -> float:
        """
        Default reward function used to evaluate a move made by the agent within the space.

        :param agent_pos: position of the agent within the space
        :param prev_pos: position of the agent before making the move in the form (x, y)

        :return: reward or penalty received by reaching the current position
        """
        # Get the type of the current position
        if prev_pos:
            move, _ = get_pos_type(agent_pos, self.agent_radius, self.target_pos, self.target_radius,
                                   self.obstacles, (self.x_max, self.y_max), True, prev_pos)
        else:
            move, _ = get_pos_type(agent_pos, self.agent_radius, self.target_pos, self.target_radius,
                                self.obstacles, (self.x_max, self.y_max))

        match move:
            case 0:  # Moved to an empty tile
                reward = -1
            case 1 | 2:  # Moved out of bounds or to an obstacle
                reward = -10
                pass
            case 3:  # Moved to a target tile
                reward = 10 * self.x_max * self.y_max
            case _:  # "Illegal move"
                raise ValueError(f"Grid cell should not have value: {move}.",
                                 f"at position {agent_pos}")
        return reward


    @staticmethod
    def evaluate_agent(space_fp: Path, agent: BaseAgent, max_steps: int, sigma: float = 0.,
                       agent_start_pos: tuple[float, float] = None,
                       random_seed: int | float | str | bytes | bytearray = 0, reward_fn: callable = None,
                       save_path: Path = None, save_name: str = None,
                       save_image: bool = True, agent_radius: float = 0.1,
                       training_positions: list[tuple[float, float]] = None,
                       obs_mode: str = "both", sensor_range: float = 10.0) -> dict:
        """
        Evaluates an agent on a specified space layout, saves the results accordingly.

        :param space_fp: path to the space layout file
        :param agent: agent to use for evaluation
        :param max_steps: maximum number of steps allowed in an episode
        :param sigma: stochasticity of the environment
        :param agent_start_pos: starting position of the agent
        :param random_seed: seed for reproducibility
        :param reward_fn: reward function to use
        :param save_path: location to save results to
        :param save_name: name of files to save results to
        :param save_image: whether to save the figure containing the path taken by the agent
        :param agent_radius: radius of the agent
        :param training_positions: all (x, y) positions visited during training.
                                   Used to create a heatmap of the agent's positions
                                   during trianing
        :param obs_mode: observation mode ('xy', 'sensors', or 'both'); must match training
        :param sensor_range: maximum sensor distance, must match training

        :return: dictionary containing information about the entire run
        """
        env = Environment(space_path=space_fp,
                          no_gui=True,
                          sigma=sigma,
                          agent_start_pos=agent_start_pos,
                          reward_fn=reward_fn,
                          target_fps=-1,
                          random_seed=random_seed,
                          agent_radius=agent_radius)

        env.reset()
        obs_builder = ObservationBuilder(env, obs_mode, sensor_range)
        state = obs_builder.build(env.agent_pos)

        # Add initial agent position to the path
        agent_path = [env.agent_pos]
        collision_path = [False]

        info = {"targets_reached": False}

        for _ in trange(max_steps, desc="Evaluating agent"):
            action = agent.take_action(state)
            _, _, terminated, info = env.step(action)
            state = obs_builder.build(env.agent_pos)

            agent_path.append(env.agent_pos)
            collision_path.append(info["collided"])

            if terminated:
                break

        env.world_stats["targets_remaining"] = 0 if info["target_reached"] else 1

        env_snapshot = deepcopy(env)
        file_name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S-%f")[
                    :-3] if not save_name else save_name  # Milliseconds precision to avoid overwriting files

        path_plot = visualize_path(env_snapshot, agent_path, collision_path)
        save_results(file_name, env.world_stats, path_plot, save_path, save_image)

        # Save heatmap of agent positions
        if save_image and training_positions is not None:
            heatmap_plot = visualize_heatmap(
                env_snapshot,
                training_positions,
                overlay_path=agent_path,
                overlay_collision_path=collision_path,
                title="Training visit frequency + final greedy episode",
            )
            out_dir = Path("results/") if not save_path else save_path
            heatmap_plot.savefig(out_dir / f"{file_name}_heatmap.pdf")

        return dict(env.world_stats)  # Return stats so it can be used in evaluation