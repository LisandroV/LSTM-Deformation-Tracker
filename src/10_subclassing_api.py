"""
In this experiment we will use the subclassing API from keras to build the model so it can do both, teacher and no-teacher forcing.

This experiment is based on Experiment-08

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
from utils.model_updater import save_best_model
from utils.script_arguments import get_script_args
from utils.dataset_creation import create_teacher_forcing_dataset, mirror_data_x_axis
import plots.dataset_plotter as plotter
import utils.logs as util_logs
import utils.normalization as normalization

np.random.seed(42)
tf.random.set_seed(42)

script_args = get_script_args()

TRAIN_DATA_DIR: str = "data/sponge_centre"
VALIDATION_DATA_DIR: str = "data/sponge_longside"
MODEL_NAME: str = "10_subclassing_api"
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

# plotter.plot_npz_control_points(
#     norm_train_polygons,
#     title="Normalized Training Control Points",
#     plot_cb=finger_position_plot(norm_train_finger_positions),
# )

# plotter.plot_npz_control_points(
#     norm_valid_polygons,
#     title="Normalized Validation Control Points",
#     plot_cb=finger_position_plot(norm_valid_finger_positions),
# )

# plotter.plot_finger_force(norm_train_forces, title="Normalized Training Finger Force")

# plotter.plot_finger_force(norm_valid_forces, title="Normalized Validation Finger Force")


# CREATE DATASET ---------------------------------------------------------------
mirrored_polygons, mirrored_finger_positions, mirrored_forces = mirror_data_x_axis(
    norm_valid_polygons, norm_valid_finger_positions, norm_valid_forces
)

# plotter.plot_npz_control_points(
#     mirrored_polygons,
#     title="Mirrored Data for training",
#     plot_cb=finger_position_plot(mirrored_finger_positions),
# )

X_train_mirror_cp, X_train_mirror_finger, y_train_mirror = create_teacher_forcing_dataset(
    mirrored_polygons, mirrored_finger_positions, mirrored_forces
)

X_train_center_sponge_cp, X_train_center_sponge_finger, y_train_center_sponge = create_teacher_forcing_dataset(
    norm_train_polygons, norm_train_finger_positions, norm_train_forces
)

# Data augmentation
X_train_cp = np.concatenate((X_train_center_sponge_cp, X_train_mirror_cp))
X_train_finger = np.concatenate((X_train_center_sponge_finger, X_train_mirror_finger))
y_train = np.concatenate((y_train_center_sponge, y_train_mirror))

X_valid_cp, X_valid_finger, y_valid = create_teacher_forcing_dataset(
    norm_valid_polygons, norm_valid_finger_positions, norm_valid_forces
)


# CREATE RECURRENT MODEL -------------------------------------------------------
class DeformationTrackerModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.hidden1 = keras.layers.SimpleRNN(15, return_sequences=True, input_shape=[None, 5])
        self.hidden2 = keras.layers.SimpleRNN(15, return_sequences=True)
        self.output_layer = keras.layers.Dense(2)
        self.__use_teacher_forcing__ = True

    def setTeacherForcing(self, useTeacherForcing: bool):
        self.__use_teacher_forcing__ = useTeacherForcing

    # inputs = (cp_input, finger_input)
    def call(self, model_input):
        control_point_input, finger_input = model_input

        if self.__use_teacher_forcing__: # With teacher forcing
            print("Using teacher forcing")
            layer_input = tf.keras.layers.Concatenate()([control_point_input, finger_input])
            hidden1 = self.hidden1(layer_input)
            hidden2 = self.hidden2(hidden1)
            model_output = self.output_layer(hidden2)
            return model_output

        else: # No teacher forcing
            print("Not using teacher forcing")
            layer_output = control_point_input[:,:1,:] # first control point of the seq
            layer_outputs=[]
            for i in range(100):
                next_layer_input = tf.keras.layers.Concatenate()([layer_output, finger_input[:,i:i+1,:]])# init layer input
                hidden1 = self.hidden1(next_layer_input)
                hidden2 = self.hidden2(hidden1)
                layer_output = self.output_layer(hidden2)
                layer_outputs.append(layer_output)
            concat_func = lambda x, y: tf.keras.layers.Concatenate(axis=1)([x, y])
            model_output = reduce(concat_func, layer_outputs)
            return model_output

        

model = DeformationTrackerModel()

# print(model.summary())

model.compile(loss="mse", optimizer="adam")


# SETUP TENSORBOARD LOGS -------------------------------------------------------
log_name = util_logs.get_log_filename(MODEL_NAME)
tensorboard_cb = keras.callbacks.TensorBoard(log_dir=log_name, histogram_freq=100, write_graph=True)


# EARLY STOPPING
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, min_delta=0.0001)


# TRAIN ------------------------------------------------------------------------
if SHOULD_TRAIN_MODEL:
    # history = model.fit(
    #     [X_train_cp, X_train_finger],
    #     y_train,
    #     validation_data=(
    #         [X_valid_cp, X_valid_finger],
    #         y_valid,
    #     ),
    #     epochs=20,
    #     callbacks=[tensorboard_cb],
    # )

    model.setTeacherForcing(False)

    history = model.fit(
        [X_train_cp, X_train_finger],
        y_train,
        validation_data=(
            [X_valid_cp, X_valid_finger],
            y_valid,
        ),
        epochs=20,
        callbacks=[tensorboard_cb],
    )

    save_best_model(model, SAVED_MODEL_DIR, [X_valid_cp, X_valid_finger], y_valid)
else:
    try:
        model = keras.models.load_model(SAVED_MODEL_DIR)
        print("Using stored model.")
    except:
        sys.exit(
            "Error:  There is no model saved.\n\tTo use the flag --train, the model has to be trained before."
        )


# PREDICTION -------------------------------------------------------------------

# ONE-STEP PREDICTION
y_pred = model.predict([X_train_center_sponge_cp, X_train_center_sponge_finger])
predicted_polygons = y_pred.swapaxes(0,1)

plotter.plot_npz_control_points(
    predicted_polygons[1:],
    title="One-Step Prediction On Train Set",
    plot_cb=finger_position_plot(norm_train_finger_positions),
)


# PREDICT ON VALIDATION SET
y_pred = model.predict([X_valid_cp, X_valid_finger])
predicted_polygons = y_pred.swapaxes(0,1)

plotter.plot_npz_control_points(
    predicted_polygons[1:],
    title="One-Step Prediction On Validation Set",
    plot_cb=finger_position_plot(norm_valid_finger_positions),
)


sys.exit()

# MULTIPLE-STEP PREDICTION
cp_start_index=0 # from which control point start plotting
offstet=47 # how many control points after cp_start_index will be plotted
to_predict = X_valid[cp_start_index:cp_start_index+offstet, :1, :]
predictions = []
for step in range(time_steps):
    y_pred = model.predict(to_predict)
    predictions.append(y_pred)
    to_predict = np.append(y_pred,X_train[cp_start_index:cp_start_index+offstet, step:step+1, 2:],axis=2)

print(np.array(predictions).shape)
predicted_polygons = np.array(predictions).reshape((100, offstet, 2))

plotter.plot_npz_control_points(
    predicted_polygons[:100],
    title="Multiple-Step Prediction on Validation Set",
    plot_cb=finger_position_plot(norm_valid_finger_positions),
)