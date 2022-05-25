import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # to supress tf warnings
from tensorflow import keras

from read_data.control_point_reader import ContourHistory
from read_data.dataset_creator import create_dataset
from read_data.finger_force_reader import read_finger_forces_file
from read_data.finger_position_reader import read_finger_positions_file

import utils.normalization as normalization
import plots.dataset_plotter as plotter

DATA_DIR: str = "data/sponge_centre"
MODEL_NAME: str = "basic_recurrent"


# READ FORCE FILE --------------------------------------------------------------
finger_force_file: str = os.path.join(DATA_DIR, "finger_force.txt")
forces: np.ndarray = read_finger_forces_file(finger_force_file)


# READ FINGER POSITION FILE ----------------------------------------------------
control_points_file: str = os.path.join(DATA_DIR, "fixed_control_points.npz")
npzfile = np.load(control_points_file)
finger_positions = npzfile['X'] # Finger positions are in X


# READ CONTROL POINTS ----------------------------------------------------------
control_points_file: str = os.path.join(DATA_DIR, "fixed_control_points.npz")
npzfile = np.load(control_points_file)
flat_control_points = npzfile['Y'] # Finger positions are in X
polygons = np.reshape(flat_control_points, (np.shape(flat_control_points)[0],-1, 2))


# NORMALIZATION
norm_polygons = normalization.normalize_polygons(polygons)
polygons_means = normalization.get_polygons_centers(polygons)


# PLOT DATA --------------------------------------------------------------------
time_steps = polygons.shape[0]

mean_plot = lambda ax: ax.plot(range(time_steps),polygons_means[:,0],polygons_means[:,1])
plotter.plot_npz_control_points(polygons, mean_plot)

xy_axis_plot = lambda ax: ax.plot(range(time_steps),[0]*time_steps,[0]*time_steps)
plotter.plot_npz_control_points(norm_polygons, xy_axis_plot)

plotter.plot_finger_position(finger_positions)

plotter.plot_finger_force(forces)


# # CREATE DATASET ---------------------------------------------------------------
# X_train, Y_train = create_dataset(history, forces, positions, 2, 10)
# print(np.shape(X_train))
# print(X_train[0])
# print(np.shape(Y_train))
# print(Y_train[0])

# # CREATE MODEL -----------------------------------------------------------------
# input_ = keras.layers.Input(shape=X_train.shape[1:])
# norm = keras.layers.LayerNormalization(axis=1)(input_)
# hidden1 = keras.layers.Dense(30, activation="tanh")(norm)
# hidden2 = keras.layers.Dense(30, activation="tanh")(hidden1)
# output = keras.layers.Dense(2)(hidden2)
# model = keras.models.Model(inputs=[input_], outputs=[output])

# print(model.summary())

# model.compile(
#     loss="mean_squared_error", optimizer=keras.optimizers.SGD(learning_rate=1e-2)
# )

# # SETUP TENSORBOARD LOGS -------------------------------------------------------
# def get_log_filename(model_name: str):
#     import time

#     logdir = os.path.join(os.curdir, "logs")
#     log_name = time.strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S")
#     return os.path.join(logdir, log_name)


# log_name = get_log_filename("basic")
# tensorboard_cb = keras.callbacks.TensorBoard(log_name)


# # TRAIN ------------------------------------------------------------------------
# print("TRAINING ------ ")
# history = model.fit(
#     X_train,
#     Y_train,
#     epochs=30,
#     callbacks=[tensorboard_cb],
# )
# # validation_data=(X_valid, y_valid))

# print("ERROR")
# mse_test = model.evaluate(X_train, Y_train)
# print(mse_test)

# print("PREDICT ------ ")
# y_pred = model.predict([X_train[:]])
# print(y_pred)
