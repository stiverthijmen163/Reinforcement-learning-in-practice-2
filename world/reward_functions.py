def dqn_reward(move, x_max, y_max):
    match move:
        case 0:  # Moved to an empty tile
            reward = -1
        case 1 | 2:  # Moved out of bounds or to an obstacle
            reward = -10 * (x_max + y_max)
            pass
        case 3:  # Moved to a target tile
            reward = 10 * x_max * y_max
        case _:  # "Illegal move"
            raise ValueError(f"Grid cell should not have value: {move}.")
    return reward

REWARD_FUNCTIONS = {
    "default": None,
    "dqn": dqn_reward
}