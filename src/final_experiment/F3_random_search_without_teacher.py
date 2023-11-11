"""
    Random search without teacher forcing.
"""

import os
import numpy as np
import sys
import time
sys.path.append('./src')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # to supress tf warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
import keras_tuner

from subclassing_models import DeformationTrackerBiFlowModel as DeformationTrackerModel
from utils.model_updater import save_best_model
from utils.script_arguments import get_script_args
from utils.weight_plot_callback import PlotWeightsCallback
import plots.dataset_plotter as plotter
from dataset import create_datasets

script_args = get_script_args()

TRAIN_DATA_DIR: str = "data/sponge_centre"
VALIDATION_DATA_DIR: str = "data/sponge_longside"

MODEL_NAME: str = "e4_rs"
SAVED_MODEL_DIR: str = f"src/final_experiment/saved_models/best_{MODEL_NAME}"
CHECKPOINT_MODEL_DIR: str = f"{SAVED_MODEL_DIR}/checkpoint/"
TRIAL_NAME = time.strftime("experiment_%Y_%m_%d-%H_%M_%S")
LOGS_DIR = f"src/final_experiment/logs/random_search_without_teacher/{MODEL_NAME}/{TRIAL_NAME}"

# Model trained with teacher forcing
PREV_MODEL_DIR: str = "src/final_experiment/saved_models/best_best_params_with_teacher_BEST"
PREV_CHECKPOINT_MODEL_DIR: str = f"{PREV_MODEL_DIR}/checkpoint/"

RANDOM_SEARCH_TRIALS = 30
TRAINING_EPOCHS = 18000

train_dataset, validation_dataset = create_datasets()

# SETUP TENSORBOARD LOGS -------------------------------------------------------
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir=LOGS_DIR,
    histogram_freq=100,
    write_graph=True
)

# SETUP RANDOM SEARCH ----------------------------------------------------------------------------
def load_weights(model):
    """Loads the weights of the training with teacher."""
    prev_model = keras.models.load_model(
        PREV_MODEL_DIR,
        custom_objects={"DeformationTrackerModel": DeformationTrackerModel},
    )

    model.build(input_shape=[(None, None, 2), (None, None, 4)])  # init model weights
    # model.set_weights(prev_model.get_weights()) # to use last model
    model.load_weights(PREV_CHECKPOINT_MODEL_DIR)  # to use checkpoint

    return model


def build_model(hp):
    model = DeformationTrackerModel(log_dir=LOGS_DIR)
    learning_rate = hp.Float("lr", min_value=0.005, max_value=0.01, sampling="log")
    epsilon = hp.Float("epsilon", min_value=1e-6, max_value=1e-5, sampling="log")
    beta_1 = hp.Float("beta_1", min_value=0.7, max_value=0.95)
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(
            learning_rate=learning_rate, beta_1=beta_1, epsilon=epsilon
        ),
        loss="mse",
    )
    loaded_model = load_weights(model)
    loaded_model.setTeacherForcing(False)

    return loaded_model

# RUN RANDOM SEARCH ----------------------------------------------------------------------
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=RANDOM_SEARCH_TRIALS,
    executions_per_trial=1,
    overwrite=True,
    directory=f"src/final_experiment/saved_models/random_search_without_teacher/{MODEL_NAME}",
    project_name=TRIAL_NAME
)
tuner.search_space_summary()

tuner.search(
    [train_dataset['X_control_points'], train_dataset['X_finger']],
    train_dataset['Y'],
    validation_data=(
        [validation_dataset['X_control_points'], validation_dataset['X_finger']],
        validation_dataset['Y'],
    ),
    epochs=TRAINING_EPOCHS,
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
models = tuner.get_best_models(num_models=RANDOM_SEARCH_TRIALS)
ratings = []
print("Ratings ---- ########################################################")
for i,m in enumerate(models):
   ratings.append({'rating': i, **m.get_compile_config()['optimizer']['config']})
print(ratings)
print("RESULTS_SUMMARY ---- ########################################################")
print(tuner.results_summary()) # To get the score of the 10 best models
print("SEARCH_SPACE_SUMMARY ---- ########################################################")
tuner.search_space_summary()
