from space import *
import os
import copy

save_path = Path("../spaces/restaurant_2_space.pickle")
if not os.path.isdir(save_path.parent):
    os.mkdir(save_path.parent)
s = Space()

# The entire space
s.change_bound(20.0, 20.0)

# Starting position
s.change_starting_pos(5.35, 5.9)

# Target
s.change_target_pos(10.0, 11.0)

# Obstacles
s.add_obstacle(2.0, 18.0, 1.0, 2.0)
s.add_obstacle(2.0, 17.0, 7.0, 1.0)

s.add_obstacle(3.0, 14.7, 6.5, 0.3)
s.add_obstacle(3.0, 13.5, 3.0, 1.0)
s.add_obstacle(6.5, 13.5, 3.0, 1.0)

s.add_obstacle(3.0, 10.5, 3.0, 1.0)
s.add_obstacle(6.5, 10.5, 3.0, 1.0)
s.add_obstacle(3.0, 10.0, 6.5, 0.3)

s.add_obstacle(8.7, 2.5, 0.3, 6.5)
s.add_obstacle(2.5, 8.7, 6.2, 0.3)

s.add_obstacle(7.2, 7.3, 1.0, 1.0)
s.add_obstacle(4.35, 7.3, 2.0, 1.0)
s.add_obstacle(2.5, 7.3, 1.0, 1.0)
s.add_obstacle(2.5, 2.5, 2.0, 2.0)
s.add_obstacle(6.2, 2.5, 2.0, 2.0)

for obst in copy.deepcopy(s.obstacles):
    x, y, w, h = obst
    s.add_obstacle(x + 10, y, w, h)

s.add_obstacle(0.0, 2.5, 0.5, 6.5)
s.add_obstacle(9.0, 2.5, 1.5, 6.5)

s.save_space(save_path)
print(Space.load_space(save_path))