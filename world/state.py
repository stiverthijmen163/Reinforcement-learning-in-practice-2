import numpy as np


SENSOR_DIRECTIONS = np.array(
    [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    ],
    dtype=np.float32,
)

SENSOR_DIRECTIONS = SENSOR_DIRECTIONS / np.linalg.norm(
    SENSOR_DIRECTIONS,
    axis=1,
    keepdims=True,
)


class ObservationBuilder:
    """Builds the observation vector for an agent based on obs_mode.

    :param env: Environment instance (used to access obstacles, bounds, radius).
    :param obs_mode: Which information to include: 'xy' (2-D), 'sensors' (8-D), or 'both' (10-D).
    :param sensor_range: Maximum sensor reading distance (raw units, not normalised).
    """

    def __init__(self, env, obs_mode="both", sensor_range=10.0):
        self.env = env
        self.obs_mode = obs_mode
        self.sensor_range = sensor_range

        if obs_mode not in ["xy", "sensors", "both"]:
            raise ValueError("obs_mode must be 'xy', 'sensors', or 'both'")

    def get_state_dim(self):
        """Return observation vector length for this obs_mode."""
        if self.obs_mode == "xy":
            return 2
        if self.obs_mode == "sensors":
            return 8
        return 10

    def build(self, state):
        """Build observation vector from environment state (agent position).

        :param state: Agent position as (x, y).
        :return: Observation vector as numpy array.
        """
        x, y = state

        if self.obs_mode == "xy":
            return np.array([x, y], dtype=np.float32)

        sensors = self.get_sensor_readings(x, y) / self.sensor_range

        if self.obs_mode == "sensors":
            return sensors

        return np.concatenate([np.array([x, y], dtype=np.float32), sensors])

    def get_sensor_readings(self, x, y):
        """Cast 8 LiDAR rays and return raw distances up to sensor_range.

        :param x: Agent x position.
        :param y: Agent y position.
        :return: Array of 8 distances.
        """
        readings = []

        for direction in SENSOR_DIRECTIONS:
            distance = self.sensor_range
            distance = min(distance, self.distance_to_wall(x, y, direction))

            for obstacle in self.env.obstacles:
                distance = min(distance, self.distance_to_obstacle(x, y, direction, obstacle))

            readings.append(distance)

        return np.array(readings, dtype=np.float32)

    def distance_to_wall(self, x, y, direction):
        dx, dy = direction
        eps = 1e-8

        left   = self.env.agent_radius
        right  = self.env.x_max - self.env.agent_radius
        bottom = self.env.agent_radius
        top    = self.env.y_max - self.env.agent_radius

        distances = []
        if dx > eps:  distances.append((right  - x) / dx)
        if dx < -eps: distances.append((left   - x) / dx)
        if dy > eps:  distances.append((top    - y) / dy)
        if dy < -eps: distances.append((bottom - y) / dy)

        distances = [d for d in distances if d >= 0]
        if not distances:
            return self.sensor_range
        return min(min(distances), self.sensor_range)

    def distance_to_obstacle(self, x, y, direction, obstacle):
        ox, oy, width, height = obstacle
        dx, dy = direction

        radius = self.env.agent_radius
        eps = 1e-8

        left   = ox - radius
        right  = ox + width + radius
        bottom = oy - radius
        top    = oy + height + radius

        t_min = -float("inf")
        t_max =  float("inf")

        for position, delta, low, high in [(x, dx, left, right), (y, dy, bottom, top)]:
            if abs(delta) < eps:
                if position < low or position > high:
                    return self.sensor_range
            else:
                t1 = (low  - position) / delta
                t2 = (high - position) / delta
                enter = min(t1, t2)
                exit  = max(t1, t2)
                t_min = max(t_min, enter)
                t_max = min(t_max, exit)
                if t_min > t_max:
                    return self.sensor_range

        if t_max < 0:
            return self.sensor_range

        return min(max(t_min, 0), self.sensor_range)
