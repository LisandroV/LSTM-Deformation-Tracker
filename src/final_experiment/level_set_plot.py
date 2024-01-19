"""This file plots the training dataset as a level set."""

import os
import numpy as np
import sys
sys.path.append('./src')
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

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


#STORED_MODEL_DIR: str = "src/final_experiment/saved_models/best_best_params_without_teacher" # BEST ON TRAINING (Has over fitting)
STORED_MODEL_DIR: str = "src/final_experiment/saved_models/best_e2_rs" # FINAL MODEL
CHECKPOINT_MODEL_DIR: str = f"{STORED_MODEL_DIR}/checkpoint/"

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
        os.makedirs(f"./src/final_experiment/tmp/F4_prediction_best_trining_set/{trial_name}")
    except:
        pass

    plt.savefig(f"./src/final_experiment/tmp/F4_prediction_best_trining_set/{trial_name}/{name}.png")
    plt.close(fig)

#prediction shape: (47, 100, 2)
#finger_positions: (100,2)
def save_prediction_images(prediction, finger_positions):
    poligon_indexes = list(concave_hull_indexes(prediction[:,0,:], length_threshold=0.05,))
    poligon_indexes.append(poligon_indexes[0])
    trial_name = time.strftime("%Y_%m_%d-%H_%M_%S")
    #for i in range(100):
        #import ipdb;ipdb.set_trace();
        # save_scatter_plot(
        #     prediction[:,i,:].take(poligon_indexes,axis=0),#shape: (48,2)
        #     trial_name,
        #     finger_position = finger_positions[i,:],
        #     name=i+1
        # )

    title = "Level Set"
    fig = plt.figure(title)
    fig.suptitle(title)
    ax = fig.add_subplot(111, projection="3d")

    NCURVES = 100
    values = range(NCURVES)
    jet = plt.get_cmap("nipy_spectral")
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

    # plots every control point history separately
    for i in values:
        try:
            colorVal = scalarMap.to_rgba(values[i])
            polygon = prediction[:,i,:].take(poligon_indexes,axis=0)#shape: (48,2)
            t = np.array([i]*48)
            #import ipdb;ipdb.set_trace();
            ax.plot(t, polygon[:, 0], polygon[:, 1]*-1, color=colorVal, alpha=0.5)
        except:
            import ipdb;ipdb.set_trace();

    ax.set_xlabel("tiempo")
    ax.set_ylabel("x")
    ax.set_zlabel("y")

    # Plot finger
    #ax.scatter(
    #    range(100), finger_positions[:, 0], finger_positions[:, 1]*-1, s=10
    #)
    plt.show()


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

    # MULTIPLE-STEP PREDICTION -------------------------------------------------------------------

    finger_data = train_dataset['X_finger'][1,:,:2]
    y_pred = model.predict([train_dataset['X_control_points'][:47,:1,:], train_dataset['X_finger'][:47,:,:]])

    save_prediction_images(y_pred, finger_data)
