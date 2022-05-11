import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # to supress tf warnings
from tensorflow import keras

from read_data.control_point_reader import ContourHistory
from read_data.dataset_creator import create_dataset
from read_data.finger_force_reader import read_finger_forces_file
from read_data.finger_position_reader import read_finger_positions_file

import plots.dataset_plotter as plotter

DATA_DIR: str = "data/sponge_centre"


# READ FORCE FILE ----------------------------------------------------------
finger_force_file: str = os.path.join(DATA_DIR, "finger_force.txt")
forces: np.ndarray = read_finger_forces_file(finger_force_file)


# READ FINGER POSITION FILE ------------------------------------------------
finger_positions_file: str = os.path.join(DATA_DIR, "finger_position.txt")
positions: np.ndarray = read_finger_positions_file(finger_positions_file)


# READ CONTROL POINTS ------------------------------------------------------
control_points_file: str = os.path.join(DATA_DIR, "control_points.hist")
history: ContourHistory = ContourHistory(control_points_file)


# PLOT DATA ----------------------------------------------------------------
# plotter.plot_control_point_history(history)
# plotter.plot_finger_position(positions)
# plotter.plot_finger_force(forces)


# CREATE DATASET ----------------------------------------------------------------
X_train, Y_train = create_dataset(history, forces, positions, 2, 10)
print(np.shape(X_train))
print(X_train[0])
print(np.shape(Y_train))
print(Y_train[0])

# CREATE MODEL ----------------------------------------------------------------
input_ = keras.layers.Input(shape=X_train.shape[1:])
norm = keras.layers.LayerNormalization(axis=1)(input_)
hidden1 = keras.layers.Dense(30, activation="tanh")(norm)
hidden2 = keras.layers.Dense(30, activation="tanh")(hidden1)
output = keras.layers.Dense(2)(hidden2)
model = keras.models.Model(inputs=[input_], outputs=[output])

print(model.summary())

model.compile(
    loss="mean_squared_error", optimizer=keras.optimizers.SGD(learning_rate=1e-2)
)

# TRAIN----------------------------------------------------------------
print("TRAINING ------ ")
history = model.fit(
    X_train,
    Y_train,
    epochs=100,
)
# validation_data=(X_valid, y_valid))

print("ERROR")
mse_test = model.evaluate(X_train, Y_train)
print(mse_test)

print("PREDICT ------ ")
y_pred = model.predict([X_train[:]])
print(y_pred)
