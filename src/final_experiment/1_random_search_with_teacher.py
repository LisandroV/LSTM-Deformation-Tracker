"""
    Random search using teacher forcing to find the best parameters for Adam.
"""
import os
import numpy as np
import sys
import time
sys.path.append('./src')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # to supress tf warnings
import tensorflow as tf
from tensorflow import keras
import keras_tuner

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
MODEL_NAME: str = "random_search_with_teacher"
SAVED_MODEL_DIR: str = f"src/final_experiment/saved_models/best_{MODEL_NAME}"
TRIAL_NAME = time.strftime("experiment_%Y_%m_%d-%H_%M_%S")
LOGS_DIR = f"src/final_experiment/logs/random_search_with_teacher/{TRIAL_NAME}"
SHOULD_TRAIN_MODEL: bool = script_args.train
TRAINING_EPOCHS = 12000


def create_datasets():
    """
        Returns training and validation datasets
    """
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




    # NORMALIZATION ----------------------------------------------------------------
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

    plotter.plot_finger_force(norm_train_forces, title="Normalized Training Finger Force")
    plotter.plot_finger_force(train_forces, title="Training Finger Force")

    plotter.plot_finger_force(norm_valid_forces, title="Normalized Validation Finger Force")
    plotter.plot_finger_force(validation_forces, title="Validation Finger Force")


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

    # DATA AUGMENTATION ------------------------------------------------------------
    train_dataset = {}
    train_dataset['X_control_points'] = np.concatenate((X_train_center_sponge_cp, X_train_mirror_cp))
    train_dataset['X_finger'] = np.concatenate((X_train_center_sponge_finger, X_train_mirror_finger))
    train_dataset['Y'] = np.concatenate((y_train_center_sponge, y_train_mirror))

    validation_dataset = {}
    (
        validation_dataset['X_control_points'],
        validation_dataset['X_finger'],
        validation_dataset['Y']
    ) = create_calculated_values_dataset(
        norm_valid_polygons, norm_valid_finger_positions, norm_valid_forces
    )

    return train_dataset, validation_dataset

train_dataset, validation_dataset = create_datasets()

# SETUP TENSORBOARD LOGS -------------------------------------------------------
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir=LOGS_DIR,
    histogram_freq=100,
    write_graph=True
)
# EARLY STOPPING
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, min_delta=0.0001)


# RANDOM SEARCH ----------------------------------------------------------------------------
def create_model() -> DeformationTrackerModel:
    model = DeformationTrackerModel(log_dir=LOGS_DIR)
    model.setTeacherForcing(True)

    return model

def build_model(hp):
    model: DeformationTrackerModel = create_model()
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=2e-2, sampling="log")
    epsilon = hp.Float("epsilon", min_value=1e-7, max_value=1e-5, sampling="log")
    beta_1 = hp.Float("beta_1", min_value=0.7, max_value=0.95)
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(
            learning_rate=learning_rate, beta_1=beta_1, epsilon=epsilon
        ),
        loss="mse",
    )
    model.setTeacherForcing(True)

    return model


tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=120,
    executions_per_trial=1,
    overwrite=True,
    directory="src/final_experiment/saved_models/random_search_with_teacher",
    project_name=TRIAL_NAME
)
tuner.search_space_summary()

tuner.search(
    [train_dataset['X_control_points'], train_dataset['X_finger']],
    train_dataset['Y'],
    epochs=TRAINING_EPOCHS,
    validation_data=(
        [validation_dataset['X_control_points'], validation_dataset['X_finger']],
        validation_dataset['Y'],
    ),
    callbacks=[tensorboard_cb,] #checkpoint_train_cb, checkpoint_valid_cb],
)
models = tuner.get_best_models(num_models=2)
best_model = models[0]
save_best_model(
    best_model,
    SAVED_MODEL_DIR,
    [validation_dataset['X_control_points'], validation_dataset['X_finger']],
    validation_dataset['Y'],
)

# Get Ratings for analysis graph ----------
models = tuner.get_best_models(num_models=120)
ratings = []
for i,m in enumerate(models):
   ratings.append({'rating': i, **m.get_compile_config()['optimizer']['config']})
print(ratings)
print(tuner.results_summary()) # To get the score of the 10 best models
