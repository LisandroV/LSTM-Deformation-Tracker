import os
import numpy as np

from read_data.finger_force_reader import read_finger_forces_file
from read_data.finger_position_reader import read_finger_positions_file
from read_data.control_point_reader import read_control_points_file

import plots.data_plotter as plotter

DATA_DIR = "data/sponge_centre"


# READ FORCE FILE ----------------------------------------------------------
finger_force_file = os.path.join(DATA_DIR, "finger_force.txt")
forces = read_finger_forces_file(finger_force_file)


# READ FINGER POSITION FILE ------------------------------------------------
finger_positions_file = os.path.join(DATA_DIR, "finger_position.txt")
positions = read_finger_positions_file(finger_positions_file)


# READ CONTROL POINTS ------------------------------------------------------
control_points_file = os.path.join(DATA_DIR, "control_points.csv")
control_points_df = read_control_points_file(control_points_file)


# PLOT DATA ----------------------------------------------------------------
plotter.plot_control_point_history(control_points_df)
plotter.plot_finger_position(positions)
plotter.plot_finger_force(forces)


# create transformers
