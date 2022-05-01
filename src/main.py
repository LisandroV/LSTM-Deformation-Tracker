import os

from read_data.finger_force_reader import read_finger_forces_file
from read_data.finger_position_reader import read_finger_positions_file
from read_data.control_point_reader import read_control_points_file

DATA_DIR = "data/sponge_centre"


# READ FORCE FILE -------------------------------------------------
finger_force_file = os.path.join(DATA_DIR, "finger_force.txt")
forces = read_finger_forces_file(finger_force_file)
print(forces)


# READ FINGER POSITION FILE -------------------------------------------------
finger_positions_file = os.path.join(DATA_DIR, "finger_position.txt")
positions = read_finger_positions_file(finger_positions_file)
print(positions)


# CONTINUE: read control points
control_points_file = os.path.join(DATA_DIR, "control_points.hist")
print(read_control_points_file(control_points_file))

# create transformers
