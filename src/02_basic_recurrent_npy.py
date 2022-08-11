"""
In this file a basic recurrent neural network is trained with the npy data using only the control points.
"""

import os
import numpy as np
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # to supress tf warnings
import tensorflow as tf
from tensorflow import keras

from read_data.finger_force_reader import read_finger_forces_file
from read_data.finger_position_reader import read_finger_positions_file
from utils.model_updater import save_best_model
from utils.script_arguments import get_script_args
from utils.dataset_creation import create_basic_dataset
import plots.dataset_plotter as plotter
import utils.logs as util_logs
import utils.normalization as normalization

np.random.seed(42)
tf.random.set_seed(42)

script_args = get_script_args()

TRAIN_DATA_DIR: str = "data/sponge_centre"
VALIDATION_DATA_DIR: str = "data/sponge_longside"
MODEL_NAME: str = "basic_recurrent_npy"
SAVED_MODEL_FILE: str = f"saved_models/best_{MODEL_NAME}_model.h5"
SHOULD_TRAIN_MODEL: bool = script_args.train


# READ FORCE FILE --------------------------------------------------------------
finger_force_file: str = os.path.join(TRAIN_DATA_DIR, "finger_force.txt")
forces: np.ndarray = read_finger_forces_file(finger_force_file)


# READ FINGER POSITION FILE ----------------------------------------------------
finger_positions_file: str = os.path.join(TRAIN_DATA_DIR, "finger_position.txt")
finger_positions: np.ndarray = read_finger_positions_file(finger_positions_file)


# READ CONTROL POINTS ----------------------------------------------------------
train_cp_file: str = os.path.join(TRAIN_DATA_DIR, "fixed_control_points.npy")
train_polygons = np.load(train_cp_file)

valid_cp_file: str = os.path.join(VALIDATION_DATA_DIR, "fixed_control_points.npy")
validation_polygons = np.load(valid_cp_file)

# NORMALIZATION
norm_train_polygons = normalization.normalize_polygons(train_polygons)
polygons_means = normalization.get_polygons_centers(train_polygons)

norm_valid_polygons = normalization.normalize_polygons(validation_polygons)

# PLOT DATA --------------------------------------------------------------------
time_steps = train_polygons.shape[0]

mean_plot = lambda ax: ax.plot(
    range(time_steps), polygons_means[:, 0], polygons_means[:, 1]
)

origin_axis_plot = lambda ax: ax.plot(
    range(time_steps), [0] * time_steps, [0] * time_steps
)
plotter.plot_npz_control_points(
    norm_train_polygons, title="Training Control Points", plot_cb=origin_axis_plot
)
plotter.plot_npz_control_points(
    norm_valid_polygons, title="Validation Control Points", plot_cb=origin_axis_plot
)
plotter.plot_finger_position(finger_positions)

plotter.plot_finger_force(forces)


# CREATE DATASET ---------------------------------------------------------------
X_train, y_train = create_basic_dataset(norm_train_polygons)
X_valid, y_valid = create_basic_dataset(norm_valid_polygons)

# CREATE RECURRENT MODEL -------------------------------------------------------
model = keras.models.Sequential(
    [
        keras.layers.SimpleRNN(94, return_sequences=True, input_shape=[None, 94]),
    ]
)

print(model.summary())

model.compile(loss="mse", optimizer="adam")


# SETUP TENSORBOARD LOGS -------------------------------------------------------
log_name = util_logs.get_log_filename(MODEL_NAME)
tensorboard_cb = keras.callbacks.TensorBoard(log_name)


# EARLY STOPPING
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, min_delta=0.0001)


# TRAIN ------------------------------------------------------------------------
if SHOULD_TRAIN_MODEL:
    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_valid,
            y_valid,
        ),
        epochs=1000,
        callbacks=[tensorboard_cb],
    )

    save_best_model(model, SAVED_MODEL_FILE, X_train, y_train)
else:
    try:
        model = keras.models.load_model(SAVED_MODEL_FILE)
        print("Using stored model.")
    except:
        sys.exit(
            "Error:  There is no model saved.\n\tTo use the flag --train, the model has to be trained before."
        )


# PREDICTION -------------------------------------------------------------------

# MULTIPLE-STEP PREDICTION
to_predict = X_train[:, :1, :]
predictions = []
for step in range(time_steps):
    y_pred = model.predict(to_predict)
    to_predict = y_pred
    predictions.append(np.reshape(y_pred, -1))
predicted_polygons = np.reshape(np.array(predictions), (100, 47, 2))

plotter.plot_npz_control_points(
    predicted_polygons[1:], title="Multiple-Step Prediction", plot_cb=origin_axis_plot
)

# ONE-STEP PREDICTION
y_pred = model.predict(X_train)
predicted_polygons = np.reshape(y_pred, (100, 47, 2))

plotter.plot_npz_control_points(
    predicted_polygons[1:],
    title="One-Step Prediction On Train Set",
    plot_cb=origin_axis_plot,
)

# PREDICT ON VALIDATION SET
y_pred = model.predict(X_valid)
predicted_polygons = np.reshape(y_pred, (100, 47, 2))

plotter.plot_npz_control_points(
    predicted_polygons[1:],
    title="One-Step Prediction On Validation Set",
    plot_cb=origin_axis_plot,
)
