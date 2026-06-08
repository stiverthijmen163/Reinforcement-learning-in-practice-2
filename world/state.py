from math import sqrt
import numpy as np

from world.helpers import agent_bumped_obstacle_during_move, directions as SENSOR_DIRS


def get_state(env) -> np.ndarray:
    """Build the normalised 10-D state vector [x/W, y/H, d1..d8].

    d1..d8 are LiDAR readings in the 8 action directions, normalised to [0, 1]
    where 1.0 means the full diagonal range and 0.0 means obstacle right here.
    """
    x, y = env.agent_pos
    radius = env.agent_radius
    # Maximum possible diagonal distance in the environment, used for normalisation of LiDAR readings.
    max_range = sqrt(env.x_max**2 + env.y_max**2)

    sensors = []
    for dx, dy in SENSOR_DIRS: # 8 unit direction vectors defined in world/helpers.py
        norm = sqrt(dx*dx + dy*dy) # The length of the direction vector
        ux, uy = dx / norm, dy / norm # Unit vector (vector with magnitude 1) in the sensor direction
        out_bounds = (x + ux * max_range, y + uy * max_range) # A point far away in that direction, guaranteed to be out of bounds
        # Find where this ray first hits an obstacle (or the wall) when cast from the agent's position in this direction.
        hit, hit_pos = agent_bumped_obstacle_during_move(env.agent_pos, out_bounds, radius, env.obstacles)

        # Distance to the nearest boundary wall along this ray
        ts = []
        if ux > 0: ts.append((env.x_max - radius - x) / ux)
        elif ux < 0: ts.append((radius - x) / ux)
        if uy > 0: ts.append((env.y_max - radius - y) / uy)
        elif uy < 0: ts.append((radius - y) / uy)
        bound_dist = min(max(t, 0.0) for t in ts) # Minimum absolute distance to nearest boundary wall
        # Find if obstacle is closer than boundary wall. Distance readings are normalised to [0, 1]
        if hit:
            hit_dist = sqrt((hit_pos[0] - x)**2 + (hit_pos[1] - y)**2)
            sensors.append(min(hit_dist, bound_dist) / max_range)
        else:
            sensors.append(bound_dist / max_range)

    # Normalise x and y to [0, 1] by dividing by environment dimensions. TODO do we want to normalise x and y?
    return np.array([x / env.x_max, y / env.y_max] + sensors, dtype=np.float32)