"""
    The only purpose of this file is to see the performance of the trained models.
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


STORED_MODEL_DIR: str = "src/final_experiment/saved_models/best_best_params_with_teacher_BEST"
#STORED_MODEL_DIR: str = "src/final_experiment/saved_models/best_random_search_with_teacher/" #5.74
CHECKPOINT_MODEL_DIR: str = f"{STORED_MODEL_DIR}/checkpoint/"

def save_prediction_images(prediction, finger_positions):
    poligon_indexes = list(concave_hull_indexes(prediction[:,0,:], length_threshold=0.05,))
    poligon_indexes.append(poligon_indexes[0])
    trial_name = time.strftime("%Y_%m_%d-%H_%M_%S")
    for i in range(100):
        save_scatter_plot(
            prediction[:,i,:].take(poligon_indexes,axis=0),
            trial_name,
            finger_position = finger_positions[i,:],
            name=i+1
        )

def save_scatter_plot(control_points, trial_name, name='X', finger_position=None):
    """Plots the trajectories of the control points through time."""
    title = f"Prediction #{str(name)}"
    fig = plt.figure(title)
    fig.suptitle(title)
    ax = fig.add_subplot(111)

    ax.scatter(control_points[:,0], control_points[:,1]*-1, color='cyan', s=30)

    if finger_position is not None:
        ax.scatter(finger_position[0], finger_position[1]*-1, color='lime', s=130)

    # link points with line
    ax.plot(control_points[:,0], control_points[:,1]*-1, color='cyan')

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    try:
        os.makedirs(f"./src/final_experiment/tmp/2_prediction/{trial_name}")
    except:
        pass

    plt.savefig(f"./src/final_experiment/tmp/2_prediction/{trial_name}/{name}.png")
    plt.close(fig)

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
    #model.set_weights(prev_model.get_weights())  # to use last model
    model.load_weights(CHECKPOINT_MODEL_DIR)  # to use checkpoint
    print("Using stored model.")

    # EVALUACION --------------------------------------------------------------------
    print("EVALUACION CON FORZAMIENTO")
    model.setTeacherForcing(True)
    train_loss = model.evaluate(
        [train_dataset['X_control_points'], train_dataset['X_finger']],
        train_dataset['Y']
    )
    print(f"Stored model loss on training set: {train_loss}")
    validation_loss = model.evaluate(
        [validation_dataset['X_control_points'], validation_dataset['X_finger']],
        validation_dataset['Y']
    )
    print(f"Stored model loss on validation set: {validation_loss}")


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


    exit()
    # DATASET PLOT -----------------------------------------------------------------
    #LAST BEST
    # best_params_with_teacher/experiment_2023_10_30-21_26_05
    # Stored model loss on training set: 5.08580487803556e-05
    # Stored model loss on validation set: 5.714035069104284e-05

    # PREDICTION -------------------------------------------------------------------

    # ONE-STEP PREDICTION
    model.setTeacherForcing(True)

    finger_position_plot = lambda positions: lambda ax: ax.scatter(
        range(100), positions[:, 0], positions[:, 1]*-1, s=10
    )


    y_pred = model.predict([validation_dataset['X_control_points'], validation_dataset['X_finger']])
    # y_pred (47,100,2)
    save_prediction_images(y_pred, validation_dataset['X_finger'][1,:,:2])

    #predicted_polygons = y_pred.swapaxes(0, 1)



    # MULTIPLE-STEP PREDICTION
    model.setTeacherForcing(False)

    y_pred = model.predict([validation_dataset['X_control_points'][:,:1,:], validation_dataset['X_finger']])
    save_prediction_images(y_pred, validation_dataset['X_finger'][1,:,:2])
