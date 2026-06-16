from space import *
import os

save_path = Path("../spaces/u_path_space.pickle")
if not os.path.isdir(save_path.parent):
    os.mkdir(save_path.parent)

s = Space()
s.change_bound(20.0, 20.0)

s.change_starting_pos(5.0, 4.0)

s.change_target_pos(15.0, 16.0)

s.add_obstacle(8.5, 0.0, 3.0, 14.0)

s.save_space(save_path)

print(Space.load_space(save_path))
