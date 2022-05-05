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
#plotter.plot_control_point_history(control_points_df)
#plotter.plot_finger_position(positions)
#plotter.plot_finger_force(forces)


# create transformers

# continue: create slices
# control_points_by_time = control_points_df.groupby(
#     ["time_step"]
# ).apply(
#     lambda group: np.column_stack((
#         np.array(group["x"]),
#         np.array(group["y"]),
#         np.array(group["id"])
#     ))
# )
# print(control_points_by_time[0])

# 0 time_step      100
# 1 birth_time     100
# 2 death_time      -1
# 3 x              553
# 4 y              390
# 5 next_id       1046
# 6 prev_id       1076


n_degree = 1
for key, row in control_points_df.iterrows():
    print(row)
    #prev = next = control_point = self.control_points[cp_ident].get_control_point(time)
    #import pdb;pdb.set_trace();
    segment = [np.array([row[3],row[4]])]
    for i in range(n_degree):
        prev = control_points_df.loc[(control_points_df['id'] == row[6]) & (control_points_df['time_step'] == row[1])]
        next = control_points_df.loc[(control_points_df['id'] == row[6]) & (control_points_df['time_step'] == row[1])]
        print(prev)
        # prev = self.control_points[prev.prev_neighbour_index].get_control_point(time)
        # next = self.control_points[next.next_neighbour_index].get_control_point(time)
        # segment.insert(0, prev)
        # segment.append(next)
    break
