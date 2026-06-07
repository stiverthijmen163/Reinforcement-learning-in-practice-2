from space import *
import os

save_path = Path("../spaces/test_space.pickle")
if not os.path.isdir(save_path.parent):
    os.mkdir(save_path.parent)
s = Space()
s.change_starting_pos(7.0, 8.0)
s.change_target_pos(19.0, 18.0)
s.add_obstacle(1.0, 2.2, 2.0, 2.0)
s.add_obstacle(3.1, 4.0, 2.4, 2.0)
s.change_bound(20.0, 20.0)
s.save_space(save_path)

print(Space.load_space(save_path))
