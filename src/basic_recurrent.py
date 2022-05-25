import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"# to supress tf warnings
import tensorflow as tf
from tensorflow import keras

from read_data.finger_force_reader import read_finger_forces_file

import utils.normalization as normalization
import utils.logs as util_logs
import plots.dataset_plotter as plotter

np.random.seed(42)
tf.random.set_seed(42)

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
plotter.plot_npz_control_points(
    polygons,
    title="Control Points from NPZ file",
    plot_cb=mean_plot
)

origin_axis_plot = lambda ax: ax.plot(range(time_steps),[0]*time_steps,[0]*time_steps)
plotter.plot_npz_control_points(
    norm_polygons,
    title="Normalized Data",
    plot_cb=origin_axis_plot
)

plotter.plot_finger_position(finger_positions)

plotter.plot_finger_force(forces)


# CREATE DATASET ---------------------------------------------------------------
X_train = np.reshape(norm_polygons,(norm_polygons.shape[0],-1))
y_train = np.zeros((100,94))
for index, row in enumerate(X_train):
    y_train[index,:-2] = X_train[index,2:]
    y_train[index,-2:] = X_train[index,:2]


# CREATE RECURRENT MODEL -------------------------------------------------------
model = keras.models.Sequential([
    keras.layers.SimpleRNN(1, return_sequences=True, input_shape=[94, 1]),
    #keras.layers.SimpleRNN(1, return_sequences=True, input_shape=[94, 1]), # deep recurent neural network
])

print(model.summary())

model.compile(loss="mse", optimizer="adam")


# SETUP TENSORBOARD LOGS -------------------------------------------------------
log_name = util_logs.get_log_filename(MODEL_NAME)
tensorboard_cb = keras.callbacks.TensorBoard(log_name)


# EARLY STOPPING
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20,min_delta=0.0001)


# TRAIN ------------------------------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_train,y_train),#must be a different validation set, needed for early stopping
    epochs=1000,
    callbacks=[early_stopping_cb, tensorboard_cb],
)

mse_test = model.evaluate(X_train, y_train)
print(f"FINAL ERROR: {mse_test}")


# MULTIPLE-STEP PREDICTION
to_predict = X_train[0:1]
predictions = []
for time_step in range(time_steps):
    y_pred = model.predict(to_predict)
    predictions.append(np.reshape(y_pred,-1))
    to_predict = np.reshape(y_pred,(1,94))
flat_predictions = np.array(predictions)
predicted_polygons = np.reshape(flat_predictions, (np.shape(flat_predictions)[0],-1, 2))
plotter.plot_npz_control_points(
    predicted_polygons,
    title="Multiple-Step Prediction",
    plot_cb=origin_axis_plot
)


# ONE-STEP PREDICTION
predictions = []
for to_predict in X_train:
    y_pred = model.predict(np.reshape(to_predict,(1,94)))
    predictions.append(np.reshape(y_pred,-1))
flat_predictions = np.array(predictions)
predicted_polygons = np.reshape(flat_predictions, (np.shape(flat_predictions)[0],-1, 2))
plotter.plot_npz_control_points(
    predicted_polygons,
    title="One-Step Prediction",
    plot_cb=origin_axis_plot
)