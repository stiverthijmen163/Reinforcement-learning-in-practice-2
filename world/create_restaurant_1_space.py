from space import *
import os

save_path = Path("../spaces/restaurant_1_space.pickle")
if not os.path.isdir(save_path.parent):
    os.mkdir(save_path.parent)
s = Space()

# The entire space
s.change_bound(50.0, 50.0)

# Starting position
s.change_starting_pos(5.0, 5.0)

# Target
s.change_target_pos(35.5, 37.0)

# Obstacles
s.add_obstacle(2.0, 46.0, 46.0, 2.5)
s.add_obstacle(2.0, 1.0, 10.0, 2.0)
s.add_obstacle(22.0, 1.0, 6.0, 2.0)
s.add_obstacle(38.0, 1.0, 10.0, 2.0)
s.add_obstacle(23.0, 12.0, 4.0, 5.0)
s.add_obstacle(4.0, 40.5, 2.0, 2.0)
s.add_obstacle(28.0, 40.5, 2.0, 2.0)

for x in [14.0, 20.0, 38.0, 44.0]:
    for y in [16.0, 25.0, 36.0]:
        s.add_obstacle(x, y, 3.0, 2.0)

for x in [7.0, 31.0]:
    for y in [12.0, 22.0]:
        s.add_obstacle(x, y, 2.0, 3.0)

for x in [15.0, 35.0]:
    s.add_obstacle(x, 8.0, 4.0, 2.0)

s.save_space(save_path)

print(Space.load_space(save_path))
