"""
    The only purpose of this file is to see the performance of the trained models.
    Plots the 3d graph.
"""

import os
import numpy as np
import sys
sys.path.append('./src')
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # to supress tf warnings
import time
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from concave_hull import concave_hull_indexes


from utils.script_arguments import get_script_args
from dataset import create_datasets
import plots.dataset_plotter as plotter
from subclassing_models import DeformationTrackerBiFlowModel as DeformationTrackerModel


STORED_MODEL_DIR: str = "src/final_experiment/saved_models/best_best_params_without_teacher"
CHECKPOINT_MODEL_DIR: str = f"{STORED_MODEL_DIR}/checkpoint/"


if __name__ == "__main__":
    train_dataset, validation_dataset = create_datasets()

    model = DeformationTrackerModel()

    # print(model.summary())

    model.compile(loss="mse", optimizer="adam")
    model.setTeacherForcing(False)

    # try:
    prev_model = keras.models.load_model(
        STORED_MODEL_DIR,
        custom_objects={"DeformationTrackerModel": DeformationTrackerModel},
    )

    model.setTeacherForcing(True)
    model.build(input_shape=[(None, 100, 2), (None, 100, 4)])  # init model weights
    model.set_weights(prev_model.get_weights())  # to use last model
    #model.load_weights(CHECKPOINT_MODEL_DIR)  # to use checkpoint
    print("Using stored model.")

    # EVALUACION --------------------------------------------------------------------

    print("EVALUACION SIN FORZAMIENTO:")
    model.setTeacherForcing(False)
    train_loss = model.evaluate(
        [train_dataset['X_control_points'][:,:1,:], train_dataset['X_finger']],
        train_dataset['Y']
    )
    print(f"Stored model loss on training set: {train_loss}")
    validation_loss = model.evaluate(
        [validation_dataset['X_control_points'][:,:1,:], validation_dataset['X_finger']],
        validation_dataset['Y']
    )
    print(f"Stored model loss on validation set: {validation_loss}")



    # PREDICTION -------------------------------------------------------------------
    model.setTeacherForcing(False)

    finger_position_plot = lambda positions: lambda ax: ax.scatter(
        range(100), positions[:, 0], positions[:, 1]*-1, s=10
    )


    # MULTIPLE PREDICTION TRAINING SET
    finger_data = validation_dataset['X_finger'][1,:,:2]


    y_pred = model.predict([validation_dataset['X_control_points'][:,:1,:], validation_dataset['X_finger']])


    predicted_polygons = y_pred.swapaxes(0, 1)
    polygons_to_show = predicted_polygons.reshape(
        100, 47, 2
    )

    plotter.plot_npz_control_points(
        polygons_to_show,
        title="Multiple-Step Prediction On Validation Set",
        plot_cb=finger_position_plot(finger_data),
    )

    # MULTIPLE PREDICTION TRAINING SET
    finger_data = train_dataset['X_finger'][1,:,:2]

    y_pred = model.predict([train_dataset['X_control_points'][:47,:1,:], train_dataset['X_finger'][:47,:,:]])


    predicted_polygons = y_pred.swapaxes(0, 1)
    polygons_to_show = predicted_polygons.reshape(
        100, 47, 2
    )

    plotter.plot_npz_control_points(
        polygons_to_show,
        title="Multiple-Step Prediction On Validation Set",
        plot_cb=finger_position_plot(finger_data),
    )
