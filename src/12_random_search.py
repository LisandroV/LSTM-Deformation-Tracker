"""
In this file the hyperparameters of the previous experiment will be fine tunned with KerasTuner

Results: ?
"""

import os
import numpy as np
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # to supress tf warnings
import tensorflow as tf
from tensorflow import keras
import keras_tuner

from read_data.finger_force_reader import read_finger_forces_file
from read_data.finger_position_reader import read_finger_positions_file
from subclassing_models import DeformationTrackerModel
from utils.dataset_creation import (
    create_teacher_forcing_dataset,
    create_calculated_values_dataset,
    mirror_data_x_axis,
)
from utils.model_updater import save_best_model
from utils.script_arguments import get_script_args
import plots.dataset_plotter as plotter
import utils.logs as util_logs
import utils.normalization as normalization

np.random.seed(42)
tf.random.set_seed(42)

script_args = get_script_args()

TRAIN_DATA_DIR: str = "data/sponge_centre"
VALIDATION_DATA_DIR: str = "data/sponge_longside"

MODEL_NAME: str = "12_rs_15_50n_discrete_2"
SAVED_MODEL_DIR: str = f"saved_models/best_{MODEL_NAME}"
CHECKPOINT_TRAIN_MODEL_DIR: str = f"{SAVED_MODEL_DIR}/checkpoint/train/"
CHECKPOINT_VALID_MODEL_DIR: str = f"{SAVED_MODEL_DIR}/checkpoint/valid/"

PREV_MODEL_NAME: str = "12_rs_15_50n_discrete_cdmx"
# PREV_MODEL_NAME: str = "10_subclassing_api_50n"
PREV_MODEL_DIR: str = f"saved_models/best_{PREV_MODEL_NAME}"
PREV_CHECKPOINT_MODEL_DIR: str = f"{PREV_MODEL_DIR}/checkpoint/"
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
# norm_train_forces = np.array([0]*11 + [1]*38 + [-1]*35 + [0]*16) # use a discrete function instead

norm_valid_polygons = normalization.normalize_polygons(validation_polygons)
norm_valid_finger_positions = normalization.normalize_finger_position(
    validation_polygons, validation_finger_positions
)
norm_valid_forces = normalization.normalize_force(validation_forces)
# norm_valid_forces = np.array([0]*14 + [1]*36 + [-1]*36 + [0]*14) # use a discrete function instead


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

# SAVE BEST CALLBACK
checkpoint_train_cb = keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_TRAIN_MODEL_DIR,
    save_weights_only=True,
    monitor="val_loss",  # TODO: change to loss
    mode="min",
    save_best_only=True,
)

checkpoint_valid_cb = keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_VALID_MODEL_DIR,
    save_weights_only=True,
    monitor="loss",  # TODO: change to val_loss
    mode="min",
    save_best_only=True,
)


def load_model() -> DeformationTrackerModel:
    model = DeformationTrackerModel(log_dir=log_name)
    model.setTeacherForcing(True)
    model.build(input_shape=[(None, 100, 2), (None, 100, 4)])  # init model weights
    # model.compile(loss="mse", optimizer="adam")
    model.setTeacherForcing(False)
    prev_model = keras.models.load_model(
        PREV_MODEL_DIR,
        custom_objects={"DeformationTrackerModel": DeformationTrackerModel},
    )
    model.set_weights(
        prev_model.get_weights()
    )  # to use last model of previous training
    # model.load_weights(PREV_CHECKPOINT_MODEL_DIR)  # to use best model of previous training
    print("Using stored model.")

    return model


def build_model(hp):
    model: DeformationTrackerModel = load_model()
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=2e-2, sampling="log")
    epsilon = hp.Float("epsilon", min_value=1e-7, max_value=1e-5, sampling="log")
    beta_1 = hp.Float("beta_1", min_value=0.7, max_value=0.95)
    model.setTeacherForcing(False)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=beta_1, epsilon=epsilon
        ),
        loss="mse",
    )

    return model


tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=60,
    executions_per_trial=1,
    overwrite=True,
    directory="saved_models/random_search",
    project_name="first_test",
)
tuner.search_space_summary()

tuner.search(
    [X_train_cp[:, :1, :], X_train_finger],
    y_train,
    epochs=10000,
    validation_data=(
        [X_valid_cp, X_valid_finger],
        y_valid,
    ),
    callbacks=[tensorboard_cb, checkpoint_train_cb, checkpoint_valid_cb],
)


tuner.results_summary()
# Get the top 2 hyperparameters.
best_hps = tuner.get_best_hyperparameters(2)
# TODO CONTINUE: fix model saving
# Build the model with the best hp.
# model = build_model(best_hps[0])
# model.setTeacherForcing(False)
# y_pred = model.predict([X_train_center_sponge_cp[:,:1,:], X_train_center_sponge_finger])
# print("pred DONE")

models = tuner.get_best_models(num_models=2)
best_model = models[0]
save_best_model(best_model, SAVED_MODEL_DIR, [X_valid_cp, X_valid_finger], y_valid)


# PREDICTION -------------------------------------------------------------------

# # MULTIPLE-STEP PREDICTION
# model.setTeacherForcing(False)

# y_pred = model.predict([X_train_center_sponge_cp[:,:1,:], X_train_center_sponge_finger])
# predicted_polygons = y_pred.swapaxes(0,1)
# polygons_to_show = np.append(norm_train_polygons[:1], predicted_polygons[1:]).reshape(100,47,2)

# plotter.plot_npz_control_points(
#     polygons_to_show,
#     title="E12: Multiple-Step Prediction On Train Set",
#     plot_cb=finger_position_plot(norm_train_finger_positions),
# )

# # PREDICT ON VALIDATION SET
# y_pred = model.predict([X_valid_cp[:,:1,:], X_valid_finger])
# predicted_polygons = y_pred.swapaxes(0,1)
# polygons_to_show = np.append(norm_valid_polygons[:1], predicted_polygons[1:]).reshape(100,47,2)

# plotter.plot_npz_control_points(
#     polygons_to_show,
#     title="E12: Multiple-Step Prediction On Validation Set",
#     plot_cb=finger_position_plot(norm_valid_finger_positions),
# )
