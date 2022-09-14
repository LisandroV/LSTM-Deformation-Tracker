"""
In this experiment the model will be trained only with the control points with the most significant deformation, this means the closest to the finger.
With this we expect the model to lear the deformation, which is what the model is missing.

Results: poor
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
from utils.dataset_creation import create_teacher_forcing_dataset, mirror_data_x_axis
import plots.dataset_plotter as plotter
import utils.logs as util_logs
import utils.normalization as normalization
from subclassing_models import DeformationTrackerModel

np.random.seed(42)
tf.random.set_seed(42)

script_args = get_script_args()

TRAIN_DATA_DIR: str = "data/sponge_centre"
VALIDATION_DATA_DIR: str = "data/sponge_longside"

MODEL_NAME: str = "13_subset_train_50n"
SAVED_MODEL_DIR: str = f"saved_models/best_{MODEL_NAME}"
CHECKPOINT_MODEL_DIR: str = f"{SAVED_MODEL_DIR}/checkpoint/"

PREV_MODEL_NAME: str = "11_no_teacher_subclassing_50n"
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

# CREATE TRAINING SUBSET: It contains only the cp that are deformed the most by the finger

mirrored_polygons, mirrored_finger_positions, mirrored_forces = mirror_data_x_axis(
    norm_valid_polygons, norm_valid_finger_positions, norm_valid_forces
)

(
    X_subset_train_mirror_cp,
    X_subset_train_mirror_finger,
    y_subset_train_mirror,
) = create_teacher_forcing_dataset(
    mirrored_polygons.take([0,6,7,9,16,20,24,25,33,34,38,39,43], axis=1),
    mirrored_finger_positions, mirrored_forces
)

(
    X_subset_train_center_sponge_cp,
    X_subset_train_center_sponge_finger,
    y_subset_train_center_sponge,
) = create_teacher_forcing_dataset(
    norm_train_polygons.take([0,3,12,16,19,22,26,28,29,37,39,40,46], axis=1),
    norm_train_finger_positions, norm_train_forces
)

# Data augmentation
X_subset_train_cp = np.concatenate((X_subset_train_center_sponge_cp, X_subset_train_mirror_cp))
X_subset_train_finger = np.concatenate((X_subset_train_center_sponge_finger, X_subset_train_mirror_finger))
y_subset_train = np.concatenate((y_subset_train_center_sponge, y_subset_train_mirror))


X_valid_cp, X_valid_finger, y_valid = create_teacher_forcing_dataset(
    norm_valid_polygons, norm_valid_finger_positions, norm_valid_forces
)

# X_subset_valid_cp, X_subset_valid_finger, y_subset_valid = create_teacher_forcing_dataset(
#     norm_valid_polygons.take([0,6,7,9,16,20,24,25,33,34,38,39,43], axis=1),
#     norm_valid_finger_positions, norm_valid_forces
# )


model = DeformationTrackerModel()

# print(model.summary())

model.compile(loss="mse", optimizer="adam")
model.setTeacherForcing(False)


# SETUP TENSORBOARD LOGS -------------------------------------------------------
log_name = util_logs.get_log_filename(MODEL_NAME)
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir=log_name, histogram_freq=100, write_graph=True
)

tensorboard_subset_cb = keras.callbacks.TensorBoard(
    log_dir=(log_name + '_subset'), histogram_freq=100, write_graph=True
)

# EARLY STOPPING
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, min_delta=0.0001)

# SAVE BEST CALLBACK
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_MODEL_DIR,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
)

# TRAIN ------------------------------------------------------------------------
if SHOULD_TRAIN_MODEL:
    prev_model = keras.models.load_model(
        PREV_MODEL_DIR,
        custom_objects={"DeformationTrackerModel": DeformationTrackerModel},
    )
    model.build(input_shape=[(None, 100, 2), (None, 100, 3)])  # init model weights
    #model.set_weights(prev_model.get_weights()) # to load from model at the end of training
    model.load_weights(PREV_CHECKPOINT_MODEL_DIR) # to load from checkpoint, best model found

    # Train with subset containing only cp closest to the finger
    history = model.fit(
        [X_subset_train_cp[:, :1, :], X_subset_train_finger],
        y_subset_train,
        validation_data=(
            [X_valid_cp[:, :1, :], X_valid_finger],
            y_valid,
        ),
        epochs=1000,
        callbacks=[tensorboard_subset_cb, checkpoint_cb],
    )

    save_best_model(model, SAVED_MODEL_DIR, [X_valid_cp, X_valid_finger], y_valid)
else:
    try:
        prev_model = keras.models.load_model(
            SAVED_MODEL_DIR,
            custom_objects={"DeformationTrackerModel": DeformationTrackerModel},
        )
        model.setTeacherForcing(True)
        model.build(input_shape=[(None, 100, 2), (None, 100, 3)])  # init model weights
        model.set_weights(prev_model.get_weights())
        print("Using stored model.")
        model.setTeacherForcing(False)
        print(
            f"Stored model valid loss: {model.evaluate([X_valid_cp[:,:1,:], X_valid_finger], y_valid)}"
        )
    except:
        sys.exit(
            "Error:  There is no model saved.\n\tTo use the flag --train, the model has to be trained before."
        )


# PREDICTION -------------------------------------------------------------------

# # ONE-STEP PREDICTION
# model.setTeacherForcing(True)
# y_pred = model.predict([X_train_center_sponge_cp, X_train_center_sponge_finger])
# predicted_polygons = y_pred.swapaxes(0, 1)

# plotter.plot_npz_control_points(
#     predicted_polygons[1:],
#     title="E11: One-Step Prediction On Train Set",
#     plot_cb=finger_position_plot(norm_train_finger_positions),
# )

# # PREDICT ON VALIDATION SET
# y_pred = model.predict([X_valid_cp, X_valid_finger])
# predicted_polygons = y_pred.swapaxes(0, 1)

# plotter.plot_npz_control_points(
#     predicted_polygons[1:],
#     title="E11: One-Step Prediction On Validation Set",
#     plot_cb=finger_position_plot(norm_valid_finger_positions),
# )


# # MULTIPLE-STEP PREDICTION
# model.setTeacherForcing(False)

# y_pred = model.predict(
#     [X_train_center_sponge_cp[:, :1, :], X_train_center_sponge_finger]
# )
# predicted_polygons = y_pred.swapaxes(0, 1)
# polygons_to_show = np.append(norm_train_polygons[:1], predicted_polygons[1:]).reshape(
#     100, 47, 2
# )

# plotter.plot_npz_control_points(
#     polygons_to_show,
#     title="E11: Multiple-Step Prediction On Train Set",
#     plot_cb=finger_position_plot(norm_train_finger_positions),
# )

# # PREDICT ON VALIDATION SET
# y_pred = model.predict([X_valid_cp[:, :1, :], X_valid_finger])
# predicted_polygons = y_pred.swapaxes(0, 1)
# polygons_to_show = np.append(norm_valid_polygons[:1], predicted_polygons[1:]).reshape(
#     100, 47, 2
# )

# plotter.plot_npz_control_points(
#     polygons_to_show,
#     title="E11: Multiple-Step Prediction On Validation Set",
#     plot_cb=finger_position_plot(norm_valid_finger_positions),
# )
