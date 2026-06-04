import random
from pathlib import Path
from warnings import warn
from helpers import *

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
            agent_radius: float = 1.0
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
        """
        random.seed(random_seed)

        # Load the space ########################## IMPLEMENT ##########################
        if not space_path.exists():
            raise FileNotFoundError(f"Grid {space_path} does not exist.")
        else:
            self.space_fp = space_path

            # Extract boundary, obstacles and starting position
            # For example usage ########################## SHOULD BE COMMENTED OUT WHEN FINISHED ##########################
            self.x_max, self.y_max = 20.0, 20.0  # Boundary of the space
            self.obstacles = [  # (x_left, y_bottom, w, h)
                (1.0, 2.2, 2.0, 2.0),
                (3.1, 4.0, 2.4, 2.0)
            ]
            self.agent_start_pos = (5.0, 6.0)  # x, y
            self.target_pos = (19.0, 18.0)  # x, y
            self.target_radius = agent_radius  # Should not be dependent on agent_radius
        ########################## END OF IMPLEMENT ##########################

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
        self.terminal_state = False
        self.sigma = sigma
        self.agent_radius = agent_radius
        self.info = {}
        self.world_stats = {}


    # def _reset_info(self) -> dict:
    #     pass


    # @staticmethod
    # def _reset_world_stats() -> dict:
    #     pass


    # @staticmethod
    # def _format_grid(grid, agent_pos=None) -> str:
    #     """
    #     Not possible for cont state space?
    #     """
    #     pass


    # def _validate_start_pos(self, pos: tuple[int, int]):
    #     pass


    def _initialize_agent_pos(self):
        ########################## IMPLEMENT ##########################
        self.agent_pos = self.agent_start_pos


    # def reset(self, **kwargs) -> tuple[int, int]:
    #     pass


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
            case 1:  # Moved into/against an obstacle
                self.world_stats["total_failed_moves"] += 1

                if prev_pos == new_pos:  # Agent is stuck (something wrong in code)
                    warn(f"Agent seems to be stuck at position {new_pos}!")
                    self.info["agent_moved"] = False
                else:  # Move on the line from prev to next position just before hitting the obstacle
                    self.agent_pos = pos_before_next_pos(prev_pos, new_pos)
            case 2:  # Moved out of the space
                self.world_stats["total_failed_moves"] += 1

                if prev_pos == new_pos:  # Agent is stuck (something wrong in code)
                    warn(f"Agent seems to be stuck at position {new_pos}!")
                    self.info["agent_moved"] = False
                else:  # Move on the line from prev to next position just before hitting the wall
                    self.agent_pos = pos_before_next_pos(prev_pos, new_pos, 0.0001 + self.agent_radius)
            case 3:  # Moved into the target
                self.agent_pos = new_pos
                self.terminal_state = True
                self.info["target_reached"] = True
                self.world_stats["total_targets_reached"] += 1
                self.info["agent_moved"] = True
                self.world_stats["total_agent_moves"] += 1
            case _:
                raise ValueError(f"Grid is badly formed. It has a value of "
                                 f"{move} at position "
                                 f"{new_pos}.")



    # def step(self, action: int) -> tuple[np.ndarray, float, bool]:
    #     pass


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


    # @staticmethod
    # def evaluate_agent():
    #     pass