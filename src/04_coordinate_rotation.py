"""
in this file the basic model is trained with the data rotating coordinates. This means every neuron learns from all the coordinates

results: The result on the validation set was very good, and the validation was constantly decreasing until the 2000 epoch. Multiple-step is terrible, it look like spaghetti.
* using LSTM the result on the multiple-step prediction was better, but on the validation se was worst than using the simple ones.
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
from utils.npy_helpers import create_rotating_coordinates_dataset
import plots.dataset_plotter as plotter
import utils.logs as util_logs
import utils.normalization as normalization

np.random.seed(42)
tf.random.set_seed(42)

script_args = get_script_args()

TRAIN_DATA_DIR: str = "data/sponge_centre"
VALIDATION_DATA_DIR: str = "data/sponge_longside"
MODEL_NAME: str = "04_coordinate_rotation"
SAVED_MODEL_FILE: str = f"saved_models/best_{MODEL_NAME}_model.h5"
TRAIN_MODEL: bool = script_args.train


# READ FORCE FILE --------------------------------------------------------------
train_finger_force_file: str = os.path.join(TRAIN_DATA_DIR, "finger_force.txt")
train_forces: np.ndarray = read_finger_forces_file(train_finger_force_file)

validation_finger_force_file: str = os.path.join(
    VALIDATION_DATA_DIR, "finger_force.txt"
)
validation_forces: np.ndarray = read_finger_forces_file(validation_finger_force_file)


# READ FINGER POSITION FILE ----------------------------------------------------
train_finger_positions_file: str = os.path.join(TRAIN_DATA_DIR, "finger_position.txt")
train_finger_positions: np.ndarray = read_finger_positions_file(
    train_finger_positions_file
)

valid_finger_positions_file: str = os.path.join(
    VALIDATION_DATA_DIR, "finger_position.txt"
)
validation_finger_positions: np.ndarray = read_finger_positions_file(
    valid_finger_positions_file
)


# READ CONTROL POINTS ----------------------------------------------------------
train_cp_file: str = os.path.join(TRAIN_DATA_DIR, "fixed_control_points.npy")
train_polygons = np.load(train_cp_file)

valid_cp_file: str = os.path.join(VALIDATION_DATA_DIR, "fixed_control_points.npy")
validation_polygons = np.load(valid_cp_file)

# NORMALIZATION
norm_train_polygons = normalization.normalize_polygons(train_polygons)
norm_train_finger_positions = normalization.normalize_finger_position(
    train_polygons, train_finger_positions
)
norm_train_forces = normalization.normalize_force(train_forces)

norm_valid_polygons = normalization.normalize_polygons(validation_polygons)
norm_valid_finger_positions = normalization.normalize_finger_position(
    validation_polygons, validation_finger_positions
)
norm_valid_forces = normalization.normalize_force(validation_forces)


# PLOT DATA --------------------------------------------------------------------
time_steps = train_polygons.shape[0]

origin_axis_plot = lambda ax: ax.plot(
    range(time_steps), [0] * time_steps, [0] * time_steps
)

# curryfied
finger_position_plot = lambda positions: lambda ax: ax.scatter(
    range(time_steps), positions[:, 0], positions[:, 1], s=10
)

plotter.plot_npz_control_points(
    norm_train_polygons,
    title="Normalized Training Control Points",
    plot_cb=finger_position_plot(norm_train_finger_positions),
)
plotter.plot_npz_control_points(
    norm_valid_polygons,
    title="Normalized Validation Control Points",
    plot_cb=finger_position_plot(norm_valid_finger_positions),
)

plotter.plot_finger_force(norm_train_forces, title="Normalized Training Finger Force")

plotter.plot_finger_force(norm_valid_forces, title="Normalized Validation Finger Force")


# CREATE DATASET ---------------------------------------------------------------
# CONTINUE: create dataset
X_train, y_train = create_rotating_coordinates_dataset(
    norm_train_polygons, norm_train_finger_positions, norm_train_forces
)
X_valid, y_valid = create_rotating_coordinates_dataset(
    norm_valid_polygons, norm_valid_finger_positions, norm_valid_forces
)

# CREATE RECURRENT MODEL -------------------------------------------------------
model = keras.models.Sequential(
    [
        keras.layers.SimpleRNN(94, return_sequences=True, input_shape=[None, 97]),
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
if TRAIN_MODEL:
    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_valid,
            y_valid,
        ),
        epochs=3000,
        callbacks=[tensorboard_cb],
    )

    save_best_model(model, SAVED_MODEL_FILE, X_train, y_train)
else:
    try:
        model = keras.models.load_model(SAVED_MODEL_FILE)
        print("Using stored model.")
    except:
        sys.exit(
            "Error:  There is no model saved.\n\tTo use the option train_model=False, the model has to be trained before."
        )


# PREDICTION -------------------------------------------------------------------

# MULTIPLE-STEP PREDICTION
to_predict = X_train[:1, :1, :]
predictions = []
for step in range(time_steps):
    y_pred = model.predict(to_predict)
    to_predict = np.append(
        np.append(y_pred.reshape(94), norm_train_finger_positions[step]),
        [norm_train_forces[step]],
    ).reshape(1, 1, 97)
    predictions.append(np.reshape(y_pred, -1))
predicted_polygons = np.reshape(np.array(predictions), (100, 47, 2))

plotter.plot_npz_control_points(
    predicted_polygons[1:],
    title="Multiple-Step Prediction",
    plot_cb=finger_position_plot(norm_train_finger_positions),
)

# ONE-STEP PREDICTION
y_pred = model.predict(X_train[:1])
predicted_polygons = np.reshape(y_pred, (100, 47, 2))

plotter.plot_npz_control_points(
    predicted_polygons[1:],
    title="One-Step Prediction On Train Set",
    plot_cb=finger_position_plot(norm_train_finger_positions),
)

# PREDICT ON VALIDATION SET
y_pred = model.predict(X_valid[:1])
predicted_polygons = np.reshape(y_pred, (100, 47, 2))

plotter.plot_npz_control_points(
    predicted_polygons[1:],
    title="One-Step Prediction On Validation Set",
    plot_cb=finger_position_plot(norm_valid_finger_positions),
)
