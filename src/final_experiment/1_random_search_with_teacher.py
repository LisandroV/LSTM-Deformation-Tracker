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

from subclassing_models import DeformationTrackerBiFlowModel as DeformationTrackerModel
from utils.model_updater import save_best_model
from utils.script_arguments import get_script_args
from utils.weight_plot_callback import PlotWeightsCallback
import plots.dataset_plotter as plotter
from dataset import create_datasets

np.random.seed(42)
tf.random.set_seed(42)

script_args = get_script_args()

TRAIN_DATA_DIR: str = "data/sponge_centre"
VALIDATION_DATA_DIR: str = "data/sponge_longside"
MODEL_NAME: str = "random_search_with_teacher"
SAVED_MODEL_DIR: str = f"src/final_experiment/saved_models/best_{MODEL_NAME}"
TRIAL_NAME = time.strftime("experiment_%Y_%m_%d-%H_%M_%S")
LOGS_DIR = f"src/final_experiment/logs/{MODEL_NAME}/{TRIAL_NAME}"
SHOULD_TRAIN_MODEL: bool = script_args.train
TRAINING_EPOCHS = 12000

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
