from math import sqrt, dist
from pathlib import Path
from warnings import warn
import matplotlib.pyplot as plt

global ACTIONS


# Create dictionary including all actions in the form ((direction vector), length to travel in that direction)
directions = [
    (0, 1),    # North
    (1, 1),    # North-East
    (1, 0),    # East
    (1, -1),   # South-East
    (0, -1),   # South
    (-1, -1),  # South-West
    (-1, 0),   # West
    (-1, 1),   # North-West
]
step_sizes = [0.2, 0.5, 1.0, 3.0]
ACTIONS = {i: (dir, s) for i, (dir, s) in enumerate((dir, s) for dir in directions for s in step_sizes)}


def agent_bumped_obstacle(x_left: float, y_bottom: float, w: float, h: float,
                          x_center: float, y_center: float, radius: float) -> bool:
    """
    Calculates whether the agent intersects/touches with an obstacle (rectangle).

    :param x_left: left x-coordinate of the obstacle
    :param y_bottom: bottom y-coordinate of the obstacle
    :param w: width of the obstacle
    :param h: height of the obstacle
    :param x_center: center x-coordinate of the agent
    :param y_center: center y-coordinate of the agent
    :param radius: radius of the agent

    :return: True iff the agent touches/intersects with the obstacle, False otherwise
    """
    # Closest point on rectangle to circle center
    closest_x = max(x_left, min(x_center, x_left + w))
    closest_y = max(y_bottom, min(y_center, y_bottom + h))

    dx = x_center - closest_x
    dy = y_center - closest_y

    return dx * dx + dy * dy <= radius * radius


def agent_in_target(x: float, y: float, x_center: float, y_center: float, radius: float) -> bool:
    """
    Checks whether the agent's center is inside the target's area.

    :param x: x-coordinate of the agent
    :param y: y-coordinate of the agent
    :param x_center: center x-coordinate of the target
    :param y_center: center y-coordinate of the target
    :param radius: radius of the target

    :return: True iff the agent is inside the target's area, False otherwise
    """
    dx = x - x_center
    dy = y - y_center

    return dx * dx + dy * dy <= radius * radius


def agent_bumped_obstacle_during_move(prev_pos: tuple[float, float], agent_pos: tuple[float, float], radius: float,
                      obstacles: list[tuple[float, float, float, float]]) -> tuple[bool, tuple[float, float]]:
    """
    Check whether the agent intersects or touches an obstacles while moving from the previous position
    to the current position, and calculates the coordinates of the agent where the first obstacle was hit.

    :param prev_pos: position of the agent before making the move in the form (x, y)
    :param agent_pos: position of the agent after making the move in the form (x, y)
    :param radius: radius of the agent
    :param obstacles: list containing all the obstacles in the space in the form (x_left, y_bottom, w, h)

    :return: (True, position of first hit) if an obstacle has been hit, (False, None) otherwise
    """
    x_start, y_start = prev_pos

    dx = agent_pos[0] - x_start
    dy = agent_pos[1] - y_start

    best_t = float("inf")
    best_pos = None

    # Check for all obstacles
    for obst_x, obst_y, obst_w, obst_h in obstacles:
        left   = obst_x - radius
        right  = obst_x + obst_w + radius
        bottom = obst_y - radius
        top    = obst_y + obst_h + radius

        tmin = 0.0
        tmax = 1.0

        collision = True

        # Check whether the agent has touched the obstacle and at which position
        for p, q in [(-dx, x_start - left), ( dx, right - x_start), (-dy, y_start - bottom), ( dy, top - y_start)]:
            if p == 0:
                if q < 0:
                    collision = False
                    break
                continue

            t = q / p

            if p < 0:
                tmin = max(tmin, t)
            else:
                tmax = min(tmax, t)

            if tmin > tmax:
                collision = False
                break

        if not collision:
            continue

        # Gets the position of the first hit of this obstacle
        if tmin < best_t:
            best_t = tmin
            best_pos = (
                x_start + tmin * dx,
                y_start + tmin * dy
            )

    return best_pos is not None, best_pos


def get_pos_type(agent_pos: tuple[float, float], agent_radius: float, target_pos: tuple[float, float],
                 target_radius: float, obstacles: list[tuple[float, float, float, float]],
                 boundary: tuple[float, float], during_move: bool = False,
                 prev_pos: tuple[float, float] = None) -> tuple[int, tuple[float, float]|None]:
    """
    Gets the type of the agent's current position in the space:
    0: free, 1: hit obstacle, 2: out of bound, 3: reached target,
    also has the option to check for obstacles hits during a move from prev_pos to agent_pos,
    and will then calculate the first point of contact with an obstacle.

    :param agent_pos: current position of the agent in the space
    :param agent_radius: radius of the agent
    :param target_pos: position of the target in the space
    :param target_radius: radius of the target
    :param obstacles: list of obstacles in the form (x_left, y_bottom, w, h)
    :param boundary: boundary of the space in the form (x_max, y_max)
    :param during_move: whether obstacle hits during the move should be taken into account
    :param prev_pos: position of the agent before making the move in the form (x, y)

    :return: as a first element, 0 if the agent is in free space, 1 if it hit an obstacle
             2 if it's out of bounds, and 3 if it has reached the target,
             secondly, it will return the agent's position when first hitting an obstacle iff one was hit,
             and return None otherwise
    """
    # If checking on the move
    if during_move:
        if prev_pos is None:
            raise ValueError(f"No previous position was provided!")
        bound_to_obstacles = [(0.0, boundary[1], boundary[0], boundary[1]),
                              (boundary[0], 0.0, boundary[0], boundary[1]),
                              (0.0, -boundary[1], boundary[0], boundary[1]),
                              (-boundary[0], 0.0, boundary[0], boundary[1])]
        contact, pos = agent_bumped_obstacle_during_move(prev_pos, agent_pos, agent_radius, obstacles)
        contact_bound, pos_b = agent_bumped_obstacle_during_move(prev_pos, agent_pos, agent_radius, bound_to_obstacles)

        # Check whether contact was made
        if contact and contact_bound:  # If contact with obstacle and out of bound, check which was hit first
            if dist(prev_pos, pos) < dist(prev_pos, pos_b):
                return 1, pos
            return 2, pos_b
        elif contact:  # Contact with obstacle(s) only
            return 1, pos
        elif contact_bound:  # Out of bounds only
            return 2, pos_b


    move = 0
    # Check whether agent is in or touches an obstacles
    for obstacle in obstacles:
        if agent_bumped_obstacle(obstacle[0], obstacle[1], obstacle[2], obstacle[3], agent_pos[0], agent_pos[1], agent_radius):
            move = 1
            break

    # Check whether agent is out of bounds
    if not agent_bumped_obstacle(0.0, 0.0, boundary[0], boundary[1], agent_pos[0], agent_pos[1], agent_radius):
        move = 2

    # Check whether target is reached
    if agent_in_target(agent_pos[0], agent_pos[1], target_pos[0], target_pos[1], target_radius):
        move = 3

    return move, None


def pos_before_next_pos(prev_pos: tuple[float, float], next_pos: tuple[float, float],
                        distance: float = 0.0001) -> tuple[float, float]:
    """
    Calculates the position a distance before the next position of the agent,
    on the line from the previous position to the next position.

    :param prev_pos: previous position of the agent in the form (x, y)
    :param next_pos: next position of the agent in the form (x, y)
    :param distance: distance to stop before reaching the next position

    :return: the position of the agent after making the move in the form (x, y)
    """
    x1, y1 = prev_pos
    x2, y2 = next_pos

    dx = x2 - x1
    dy = y2 - y1

    length = sqrt(dx*dx + dy*dy)
    factor = (length - distance) / length

    return x1 + dx * factor, y1 + dy * factor


def calc_next_position(prev_pos: tuple[float, float], direction: tuple[float, float],
                       distance: float) -> tuple[float, float]:
    """
    Calculates the next position from prev_pos given a direction vector
    and a distance to travel into that direction.

    :param prev_pos: the position of the agent before making the move in the form (x, y)
    :param direction: the direction vector in the form (x, y)
    :param distance: distance to travel along the direction from prev_pos

    :return: the position of the agent after making the move in the form (x, y)
    """
    x, y = prev_pos
    dx, dy = direction
    length = dist((0.0, 0.0), direction)

    # Calculate the next position
    if length == 0:  # In case of no direction
        next_pos = prev_pos
    else:
        next_pos = (x + distance * dx / length, y + distance * dy / length)

    return next_pos


def save_results(file_name: str, world_stats: dict, path_figure: plt.Figure,
                 save_path: Path = None, save_image: bool = True) -> None:
    """
    Saves and prints the results of the simulation into a file and its corresponding path taken.

    :param file_name: name of the .txt and .pdf file
    :param world_stats: dict containing information about the entire run
    :param path_figure: figure containing the path
    :param save_path: location to save the files to
    :param save_image: whether to save the figure and results
    :return:
    """
    out_dir = Path("results/") if not save_path else save_path
    if not out_dir.exists():
        warn("Evaluation output directory does not exist. Creating the "
             "directory.")
        out_dir.mkdir(parents=True, exist_ok=True)

    # Print evaluation results
    print("Evaluation complete. Results:")
    # Text file
    out_fp = out_dir / f"{file_name}.txt"
    with open(out_fp, "w") as f:
        for key, value in world_stats.items():
            f.write(f"{key}: {value}\n")
            print(f"{key}: {value}")

    if save_image:
        # Image file
        out_fp = out_dir / f"{file_name}.pdf"
        path_figure.savefig(out_fp)