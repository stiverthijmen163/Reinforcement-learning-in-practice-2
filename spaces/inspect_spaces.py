"""Print the contents of all space pickle files and show each layout as an image."""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def inspect(path: Path, ax) -> None:
    with open(path, "rb") as f:
        space = pickle.load(f)

    bound     = space["bound"]
    start     = space["starting_pos"]
    target    = space["target_pos"]
    obstacles = space["obstacles"]

    print(f"=== {path.name} ===")
    print(f"  Bounds      : {bound[0]} x {bound[1]}")
    print(f"  Start pos   : {start}")
    print(f"  Target pos  : {target}")
    print(f"  Obstacles   : {len(obstacles)}")
    for i, (x, y, w, h) in enumerate(obstacles):
        print(f"    [{i}] x={x}, y={y}, w={w}, h={h}")
    print()

    # Background
    ax.set_facecolor("#f5f5f5")
    ax.set_xlim(0, bound[0])
    ax.set_ylim(0, bound[1])
    ax.set_aspect("equal")
    ax.set_title(path.name, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Obstacles
    for (ox, oy, ow, oh) in obstacles:
        ax.add_patch(patches.Rectangle((ox, oy), ow, oh, color="#555555"))

    # Start and target
    if start:
        ax.plot(*start, marker="o", color="royalblue", markersize=8, label="Start")
        ax.annotate("S", xy=start, fontsize=8, color="royalblue",
                    xytext=(4, 4), textcoords="offset points")
    if target:
        ax.plot(*target, marker="*", color="crimson", markersize=12, label="Target")
        ax.annotate("T", xy=target, fontsize=8, color="crimson",
                    xytext=(4, 4), textcoords="offset points")

    ax.legend(fontsize=7, loc="upper right")


if __name__ == "__main__":
    here = Path(__file__).parent
    pickle_files = sorted(here.glob("*.pickle"))

    fig, axes = plt.subplots(1, len(pickle_files), figsize=(5 * len(pickle_files), 5))
    if len(pickle_files) == 1:
        axes = [axes]

    for ax, path in zip(axes, pickle_files):
        inspect(path, ax)

    plt.tight_layout()
    plt.show()
