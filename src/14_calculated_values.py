"""
In this experiment we will give the model calculated values to learn from, like the distance between the finger and the control point

This experiment is based on Experiment-10

Results: ?
"""

import os
import numpy as np
import sys
from functools import reduce

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # to supress tf warnings
import tensorflow as tf
from tensorflow import keras

from read_data.finger_force_reader import read_finger_forces_file
from read_data.finger_position_reader import read_finger_positions_file
from subclassing_models import DeformationTrackerBiFlowModel as DeformationTrackerModel
from utils.dataset_creation import create_calculated_values_dataset, mirror_data_x_axis
from utils.model_updater import save_best_model
from utils.script_arguments import get_script_args
from utils.weight_plot_callback import PlotWeightsCallback
import plots.dataset_plotter as plotter
import utils.logs as util_logs
import utils.normalization as normalization

np.random.seed(42)
tf.random.set_seed(42)

script_args = get_script_args()

TRAIN_DATA_DIR: str = "data/sponge_centre"
VALIDATION_DATA_DIR: str = "data/sponge_longside"
MODEL_NAME: str = "14_e_tune_1_6"
SAVED_MODEL_DIR: str = f"saved_models/best_{MODEL_NAME}"
SHOULD_TRAIN_MODEL: bool = script_args.train


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
norm_train_forces = np.array(
    [0] * 11 + [1] * 38 + [-1] * 35 + [0] * 16
)  # use a discrete function instead

norm_valid_polygons = normalization.normalize_polygons(validation_polygons)
norm_valid_finger_positions = normalization.normalize_finger_position(
    validation_polygons, validation_finger_positions
)
norm_valid_forces = normalization.normalize_force(validation_forces)
norm_valid_forces = np.array(
    [0] * 14 + [1] * 36 + [-1] * 36 + [0] * 14
)  # use a discrete function instead


# PLOT DATA --------------------------------------------------------------------
time_steps = train_polygons.shape[0]

origin_axis_plot = lambda ax: ax.plot(
    range(time_steps), [0] * time_steps, [0] * time_steps
)

# curryfied
finger_position_plot = lambda positions: lambda ax: ax.scatter(
    range(time_steps), positions[:, 0], positions[:, 1], s=10
)

# CREATE DATASET ---------------------------------------------------------------
mirrored_polygons, mirrored_finger_positions, mirrored_forces = mirror_data_x_axis(
    norm_valid_polygons, norm_valid_finger_positions, norm_valid_forces
)

(
    X_train_mirror_cp,
    X_train_mirror_finger,
    y_train_mirror,
) = create_calculated_values_dataset(
    mirrored_polygons, mirrored_finger_positions, mirrored_forces
)


(
    X_train_center_sponge_cp,
    X_train_center_sponge_finger,
    y_train_center_sponge,
) = create_calculated_values_dataset(
    norm_train_polygons, norm_train_finger_positions, norm_train_forces
)

# Data augmentation
X_train_cp = np.concatenate((X_train_center_sponge_cp, X_train_mirror_cp))
X_train_finger = np.concatenate((X_train_center_sponge_finger, X_train_mirror_finger))
y_train = np.concatenate((y_train_center_sponge, y_train_mirror))

X_valid_cp, X_valid_finger, y_valid = create_calculated_values_dataset(
    norm_valid_polygons, norm_valid_finger_positions, norm_valid_forces
)


# SETUP TENSORBOARD LOGS -------------------------------------------------------
log_name = util_logs.get_log_filename(MODEL_NAME)
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir=log_name, histogram_freq=100, write_graph=True
)


# EARLY STOPPING
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, min_delta=0.0001)

# MODEL CREATION --------------------------------------------------------------

model = DeformationTrackerModel(log_dir=log_name)


model.compile(loss="mse", optimizer="adam")


# TRAIN ------------------------------------------------------------------------
if SHOULD_TRAIN_MODEL:
    model.setTeacherForcing(True)
    history = model.fit(
        [X_train_cp, X_train_finger],
        y_train,
        validation_data=(
            [X_valid_cp, X_valid_finger],
            y_valid,
        ),
        epochs=6000,
        callbacks=[tensorboard_cb, PlotWeightsCallback(plot_freq=50)],
        workers=4,
    )

    save_best_model(model, SAVED_MODEL_DIR, [X_valid_cp, X_valid_finger], y_valid)
else:
    try:
        prev_model = keras.models.load_model(
            SAVED_MODEL_DIR,
            custom_objects={"DeformationTrackerModel": DeformationTrackerModel},
        )
        model.build(input_shape=[(None, 100, 2), (None, 100, 4)])  # init model weights
        model.set_weights(prev_model.get_weights())
        print("Using stored model.")
        model.setTeacherForcing(True)
        print(
            f"Stored model train loss: {model.evaluate([X_train_cp, X_train_finger],y_train)}"
        )
        print(
            f"Stored model valid loss: {model.evaluate([X_valid_cp, X_valid_finger], y_valid)}"
        )
    except:
        sys.exit(
            "Error:  There is no model saved.\n\tTo use the flag --train, the model has to be trained before."
        )


# PREDICTION -------------------------------------------------------------------

# ONE-STEP PREDICTION
y_pred = model.predict([X_train_center_sponge_cp, X_train_center_sponge_finger])
predicted_polygons = y_pred.swapaxes(0, 1)

plotter.plot_npz_control_points(
    predicted_polygons[1:],
    title="E14: One-Step Prediction On Train Set",
    plot_cb=finger_position_plot(norm_train_finger_positions),
)


# PREDICT ON VALIDATION SET
y_pred = model.predict([X_valid_cp, X_valid_finger])
predicted_polygons = y_pred.swapaxes(0, 1)

plotter.plot_npz_control_points(
    predicted_polygons[1:],
    title="E14: One-Step Prediction On Validation Set",
    plot_cb=finger_position_plot(norm_valid_finger_positions),
)


# MULTIPLE-STEP PREDICTION
model.setTeacherForcing(False)

y_pred = model.predict(
    [X_train_center_sponge_cp[:, :1, :], X_train_center_sponge_finger]
)
predicted_polygons = y_pred.swapaxes(0, 1)

plotter.plot_npz_control_points(
    predicted_polygons[1:],
    title="E14: Multiple-Step Prediction On Train Set",
    plot_cb=finger_position_plot(norm_train_finger_positions),
)

# PREDICT ON VALIDATION SET
y_pred = model.predict([X_valid_cp[:, :1, :], X_valid_finger])
predicted_polygons = y_pred.swapaxes(0, 1)

plotter.plot_npz_control_points(
    predicted_polygons[1:],
    title="E14: Multiple-Step Prediction On Validation Set",
    plot_cb=finger_position_plot(norm_valid_finger_positions),
)
