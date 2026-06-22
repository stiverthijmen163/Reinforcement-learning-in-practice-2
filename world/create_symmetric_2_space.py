from space import *
import os

save_path = Path("../spaces/symmetric_2_space.pickle")
if not os.path.isdir(save_path.parent):
    os.mkdir(save_path.parent)
s = Space()

s.change_bound(20.0, 20.0)

# Mirror start is (15.0, 3.0) -- not stored here, sampled 50/50 at training time.
s.change_starting_pos(5.0, 3.0)

# Centered, not mirrored: this room isn't a perceptual-aliasing test.
s.change_target_pos(10.0, 17.0)

# Left half, reflected about x=10 via (x, y, w, h) -> (20 - x - w, y, w, h).
left_half = [
    (2.0, 0.0, 0.3, 7.0),
    (7.7, 0.0, 0.3, 7.0),
    (4.25, 5.0, 1.5, 0.3),
    (4.7, 5.3, 0.6, 9.7),
]
for x, y, w, h in left_half:
    s.add_obstacle(x, y, w, h)
    s.add_obstacle(20 - x - w, y, w, h)

s.save_space(save_path)
print(Space.load_space(save_path))
