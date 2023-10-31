"""
    Training with teacher forcing using the best parameters obtained from the random search.
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
from dataset import create_datasets
import plots.dataset_plotter as plotter

np.random.seed(42)
tf.random.set_seed(42)

script_args = get_script_args()

TRAIN_DATA_DIR: str = "data/sponge_centre"
VALIDATION_DATA_DIR: str = "data/sponge_longside"
MODEL_NAME: str = "best_params_with_teacher"
SAVED_MODEL_DIR: str = f"src/final_experiment/saved_models/best_{MODEL_NAME}"
CHECKPOINT_MODEL_DIR: str = f"{SAVED_MODEL_DIR}/checkpoint/"
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

# SAVE BEST CALLBACK
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_MODEL_DIR,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
)


# CREATE MODEL
model = DeformationTrackerModel(log_dir=LOGS_DIR)
model.setTeacherForcing(True)

model.compile(
    optimizer=keras.optimizers.legacy.Adam(
        learning_rate=0.004773385573290485,
        beta_1=0.9177266177579712,
        epsilon=1.5462747018467993e-06
    ),
    loss="mse",
)

history = model.fit(
    [train_dataset['X_control_points'], train_dataset['X_finger']],
    train_dataset['Y'],
    epochs=TRAINING_EPOCHS,
    validation_data=(
        [validation_dataset['X_control_points'], validation_dataset['X_finger']],
        validation_dataset['Y'],
    ),
    callbacks=[tensorboard_cb, checkpoint_cb]
)

save_best_model(
    model,
    SAVED_MODEL_DIR,
    [validation_dataset['X_control_points'], validation_dataset['X_finger']],
    validation_dataset['Y'],
)
