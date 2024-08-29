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
from concave_hull import concave_hull_indexes


from dataset import create_datasets


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
    #poligon_indexes.append(poligon_indexes[0])
    trial_name = time.strftime("%Y_%m_%d-%H_%M_%S")

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
    new_level_set = []
    for i in values:

        colorVal = scalarMap.to_rgba(values[i])
        polygon = prediction[:,i,:].take(poligon_indexes,axis=0)# shape: (48,2)
        polygon *= np.array([1,-1])
        t = np.array([i]*47)
        ax.plot(t, polygon[:, 0], polygon[:, 1], color=colorVal, alpha=0.5)
        new_level_set.append(polygon)

    print("level_set = " + str(np.array(new_level_set).tolist()))
    # Plot finger position
    ax.plot(np.array(values), finger_positions[:, 0], finger_positions[:, 1]*-1, color='blue', alpha=0.5)

    ax.set_xlabel("tiempo")
    ax.set_ylabel("x")
    ax.set_zlabel("y")

    plt.show()


if __name__ == "__main__":
    train_dataset, validation_dataset = create_datasets()

    to_extract = validation_dataset

    #np.shape(validation_dataset['X_control_points']) # (47, 100, 2)
    #np.shape(y_pred) #(47, 100, 2)

    save_prediction_images(to_extract['X_control_points'], to_extract['X_finger'][0,:,:2]) # FIXME: missing force

    finger_data = to_extract['X_finger'][0,:,:]*np.array([1,-1,1,1])
    print("finger_data = " + str(finger_data.tolist()))
