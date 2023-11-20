"""
    Random search without teacher forcing.
    This file is to get the best model on training set,
    this means the model with the lowest error on the training set.
"""

import os
import numpy as np
import sys
import time
sys.path.append('./src')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # to supress tf warnings
import tensorflow as tf
from tensorflow import keras

from subclassing_models import DeformationTrackerBiFlowModel as DeformationTrackerModel
from utils.model_updater import save_best_model
from utils.script_arguments import get_script_args
from utils.weight_plot_callback import PlotWeightsCallback
import plots.dataset_plotter as plotter
import utils.logs as util_logs
from dataset import create_datasets

script_args = get_script_args()

TRAIN_DATA_DIR: str = "data/sponge_centre"
VALIDATION_DATA_DIR: str = "data/sponge_longside"

MODEL_NAME: str = "best_params_without_teacher"
SAVED_MODEL_DIR: str = f"src/final_experiment/saved_models/best_{MODEL_NAME}"
CHECKPOINT_MODEL_DIR: str = f"{SAVED_MODEL_DIR}/checkpoint/"
TRIAL_NAME = time.strftime("experiment_%Y_%m_%d-%H_%M_%S")
LOGS_DIR = f"src/final_experiment/logs/{MODEL_NAME}/{TRIAL_NAME}"

# Model trained with teacher forcing
PREV_MODEL_DIR: str = "src/final_experiment/saved_models/best_best_params_with_teacher_BEST"
PREV_CHECKPOINT_MODEL_DIR: str = f"{PREV_MODEL_DIR}/checkpoint/"
TRAINING_EPOCHS = 4000

train_dataset, validation_dataset = create_datasets()

# SETUP TENSORBOARD LOGS -------------------------------------------------------
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir=LOGS_DIR,
    histogram_freq=100,
    write_graph=True
)

# SAVE BEST CALLBACK ON TRAINING
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_MODEL_DIR,
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

# CREATE MODEL
model = DeformationTrackerModel(log_dir=LOGS_DIR)

# print(model.summary())

model.compile(
    optimizer=keras.optimizers.legacy.Adam(
        learning_rate=0.001078203282079985,
        beta_1=0.9136968841686723,
        epsilon=8.652341013580072e-06
    ),
    loss="mse",
)
model.setTeacherForcing(False)


# TRAIN ------------------------------------------------------------------------
prev_model = keras.models.load_model(
    PREV_MODEL_DIR,
    custom_objects={"DeformationTrackerModel": DeformationTrackerModel},
)

model.build(input_shape=[(None, None, 2), (None, None, 4)])  # init model weights
# model.set_weights(prev_model.get_weights()) # to use last model
model.load_weights(PREV_CHECKPOINT_MODEL_DIR)  # to use checkpoint
print("Using stored model.")

model.setTeacherForcing(False)
history = model.fit(
    [train_dataset['X_control_points'], train_dataset['X_finger']],
    train_dataset['Y'],
    validation_data=(
        [validation_dataset['X_control_points'], validation_dataset['X_finger']],
        validation_dataset['Y'],
    ),
    epochs=TRAINING_EPOCHS,
    callbacks=[tensorboard_cb, checkpoint_cb] #, PlotWeightsCallback(plot_freq=50)],
)

save_best_model(
    model,
    SAVED_MODEL_DIR,
    [validation_dataset['X_control_points'], validation_dataset['X_finger']],
    validation_dataset['Y']
)
