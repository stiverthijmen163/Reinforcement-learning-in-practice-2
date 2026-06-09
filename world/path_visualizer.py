import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D


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


def visualize_heatmap(
        env,
        all_positions: list[tuple[float, float]],
        overlay_path: list[tuple[float, float]] = None,
        overlay_collision_path: list[bool] = None,
        resolution: int = 50,
        title: str = None,
) -> plt.Figure:
    """
    Visualizes a heatmap of agent positions across all training episodes,
    with an optional path overlaid on top (typically the final greedy episode).

    Each cell shows how often the agent visited that region. The color scale is 
    normalized with square root so both hotspots and low-frequency areas are visible.
    Mainly done because certain areas otherwise dominate the scale and then the rest of
    the path is more difficult to see.

    The start cell is excluded from the scale maximum to prevent it from dominating
    (it is visited once per episode and is therefore often the most-visited cell).
    
    Unvisited cells are shown in white.

    :param env: Environment object (used for bounds, obstacles, target, agent_radius,
                and agent_start_pos for scale normalization)
    :param all_positions: flat list of (x, y) positions from all training episodes combined
    :param overlay_path: optional single path drawn on top of the heatmap
    :param overlay_collision_path: collision flags per step, aligned with overlay_path
    :param resolution: number of histogram bins per axis (lower = smoother, higher = more detail)
    :param title: optional figure title

    :return: figure containing the heatmap
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    xs = np.array([p[0] for p in all_positions])
    ys = np.array([p[1] for p in all_positions])

    # Build 2D visit-frequency histogram
    density, xedges, yedges = np.histogram2d(
        xs, ys,
        bins=resolution,
        range=[[0, env.x_max], [0, env.y_max]],
    )

    bin_width = env.x_max / resolution
    bin_height = env.y_max / resolution

    # Mask out obstacle cells so they don't pollute the color scale
    for obs_x, obs_y, obs_w, obs_h in env.obstacles:
        x_bin_min = max(0, int(obs_x / bin_width))
        x_bin_max = min(resolution, int(np.ceil((obs_x + obs_w) / bin_width)))
        y_bin_min = max(0, int(obs_y / bin_height))
        y_bin_max = min(resolution, int(np.ceil((obs_y + obs_h) / bin_height)))
        density[x_bin_min:x_bin_max, y_bin_min:y_bin_max] = np.nan

    # Transpose so rows=y, cols=x for imshow
    density_plot = density.T.copy()

    # Make sure the unvisited cells are shown as white (background color)
    density_plot[density_plot == 0] = np.nan

    # Exclude start cell by exact coordinates
    density_for_scale = density_plot.copy()
    if env.agent_start_pos is not None:
        start_x, start_y = env.agent_start_pos
        x_bin = int(np.clip(start_x / bin_width, 0, resolution - 1))
        y_bin = int(np.clip(start_y / bin_height, 0, resolution - 1))
        density_for_scale[y_bin, x_bin] = np.nan

    vmax = float(np.nanmax(density_for_scale)) if np.any(~np.isnan(density_for_scale)) else 2.0
    vmax = max(vmax, 1.0)

    # Square-root normalization:
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax)

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="white")

    img = ax.imshow(
        density_plot,
        extent=[0, env.x_max, 0, env.y_max],
        origin="lower",
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",  # avoids white lines between cells
        norm=norm,
    )
    plt.colorbar(img, ax=ax, label="Visit frequency (sqrt scale)", fraction=0.046, pad=0.04)

    # Draw obstacles as solid black on top of the heatmap
    for obs_x, obs_y, obs_w, obs_h in env.obstacles:
        ax.add_patch(plt.Rectangle((obs_x, obs_y), obs_w, obs_h, facecolor="black", zorder=5))

    # Draw target
    ax.add_patch(plt.Circle(env.target_pos, env.target_radius, facecolor="green", alpha=0.5, zorder=6))

    # Overlay a specific path (default final greedy episode)
    if overlay_path is not None:
        path_x = [p[0] for p in overlay_path]
        path_y = [p[1] for p in overlay_path]
        ax.plot(path_x, path_y, color="white", linewidth=1.5, alpha=0.85, zorder=10)

        ax.add_patch(plt.Circle(
            (path_x[0], path_y[0]), env.agent_radius,
            facecolor="#f2d352", edgecolor="black", linewidth=0.5, zorder=11,
        ))
        ax.add_patch(plt.Circle(
            (path_x[-1], path_y[-1]), env.agent_radius,
            facecolor="purple", edgecolor="none", zorder=11,
        ))

        if overlay_collision_path is not None:
            for i, col in enumerate(overlay_collision_path):
                if col:
                    ax.add_patch(plt.Circle(
                        (path_x[i], path_y[i]), env.agent_radius,
                        facecolor="red", alpha=0.5, zorder=12,
                    ))

    ax.set_xlim([0, env.x_max])
    ax.set_ylim([0, env.y_max])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)

    # Create legend:
    legend_elements = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=(0.0, 0.5, 0.0, 0.5), markeredgecolor="none",
               markersize=10, label="target"),
        Patch(facecolor="black", label="obstacle"),
    ]
    if overlay_path is not None:
        legend_elements += [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="#f2d352", markeredgecolor="black", markeredgewidth=0.5,
                   markersize=8, label="start"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="purple", markeredgecolor="none",
                   markersize=8, label="end"),
        ]
        if overlay_collision_path is not None and any(overlay_collision_path):
            legend_elements.append(
                Line2D([0], [0], marker="o", color="none",
                       markerfacecolor=(1.0, 0.0, 0.0, 0.5), markeredgecolor="none",
                       markersize=8, label="collision"),
            )
    ax.legend(handles=legend_elements, loc="upper right")

    return fig