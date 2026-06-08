import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch


def visualize_path(env, agent_path: list[tuple[float, float]], collision_path: list[bool]) -> plt.Figure:
    """
    Visualizes the path taken by an agent.

    :param env: the Environment object containing all important information regarding the space
    :param agent_path: the path taken bij de agent
    :param collision_path: whether a collision was made at a certain step

    :return: figure containing the path
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    x_coords = [i[0] for i in agent_path]
    y_coords = [i[1] for i in agent_path]

    cmap = plt.get_cmap("Blues")
    colors = [
        cmap(0.30),  # lighter
        cmap(0.50),  # slightly darker
    ]

    # Plot the line with alternating colors, to distinct between moves
    for i in range(len(x_coords) - 1):
        ax.plot(x_coords[i:i + 2], y_coords[i:i + 2], color=colors[i % 2], linewidth=2)

    # Plot all the obstacles
    for obstacle in env.obstacles:
        x, y, w, h = obstacle
        obst = plt.Rectangle((x, y), w, h, facecolor="black")
        ax.add_patch(obst)

    # Add the starting position as a yellow marker
    ax.add_patch(plt.Circle((x_coords[0], y_coords[0]), radius=env.agent_radius, color="#f2d352"))

    # Plot the agent at each position of a collision
    for i, col in enumerate(collision_path):
        if col:
            agent = plt.Circle((x_coords[i], y_coords[i]), radius=env.agent_radius, facecolor="red", alpha=0.3, zorder=100)
            ax.add_patch(agent)

    # Plot the target
    target = plt.Circle(env.target_pos, env.target_radius, facecolor="green", alpha=0.5)
    ax.add_patch(target)

    # Plot the final agent's position as a black marker
    ax.add_patch(plt.Circle((x_coords[-1], y_coords[-1]), env.agent_radius, facecolor="purple", zorder=100))

    # Limit plot to the boundary of the environment
    plt.xlim([0, env.x_max])
    plt.ylim([0, env.y_max])

    legend_elements = [
        Circle((0, 0), radius=0.2, facecolor="#f2d352", label="start"),
        Circle((0, 0), radius=0.2, facecolor="green", alpha=0.5, label="target"),
        Circle((0, 0), radius=0.2, facecolor="purple", label="end"),
        Circle((0, 0), radius=0.2, facecolor="red", alpha=0.3, label="collision"),
        Patch(facecolor="black", edgecolor="black", label="obstacle"),
    ]
    ax.legend(handles=legend_elements)

    return fig