import pickle
from pathlib import Path


class Space:
    """
    A simple object that is used to create, save and load the space to play in,
    note that this object does not check for validity and reachability.
    """

    def __init__(self):
        self.obstacles = []
        self.target_pos = None
        self.starting_pos = None
        self.bound = None


    def add_obstacle(self, x_left: float, y_bottom: float, w: float, h: float) -> None:
        self.obstacles.append((x_left, y_bottom, w, h))


    def remove_obstacle(self, x_left: float, y_bottom: float, w: float, h: float) -> None:
        for obstacle in self.obstacles:
            if obstacle == (x_left, y_bottom, w, h):
                self.obstacles.remove(obstacle)


    def change_target_pos(self, x: float, y: float) -> None:
        self.target_pos = (x, y)


    def change_starting_pos(self, x: float, y: float) -> None:
        self.starting_pos = (x, y)


    def change_bound(self, x: float, y: float) -> None:
        self.bound = (x, y)


    def save_space(self, save_path: Path) -> None:
        dct = {
            "starting_pos": self.starting_pos,
            "obstacles": self.obstacles,
            "target_pos": self.target_pos,
            "bound": self.bound
        }
        with open(save_path, "wb") as f:
            pickle.dump(dct, f)


    @ staticmethod
    def load_space(save_path: Path) -> dict:
        with open(save_path, "rb") as f:
            return pickle.load(f)