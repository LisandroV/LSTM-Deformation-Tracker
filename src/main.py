import os
import numpy as np
import time

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
#plotter.plot_control_point_history(control_points_df)
#plotter.plot_finger_position(positions)
#plotter.plot_finger_force(forces)


# create transformers

control_points_df = control_points_df.set_index(['id', 'time_step']).sort_index()

n_degree = 10
segments = [] # contains the neighbors for every point in time
for key, row in control_points_df.iterrows():
    segment = [np.array([row[2],row[3]])]
    prev_id = row[4]
    next_id = row[5]
    for i in range(n_degree):
        prev = control_points_df.loc[prev_id, key[1]]
        next = control_points_df.loc[next_id, key[1]]
        segment.insert(0, np.array(
            [prev['x'], prev['y']]
        ))
        segment.append(np.array(
            [next['x'], next['y']]
        ))
        prev_id = prev['prev_id']
        next_id = next['next_id']
    segments.append(np.array(segment))
    print(str(key) + " segment... " + time.ctime())

segments = np.array(segments)
print(segments)
print(np.shape(segments))