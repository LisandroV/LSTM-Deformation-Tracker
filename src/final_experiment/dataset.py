"""
    Function to create the training and validation dataset.
"""

import os
import numpy as np
import sys
sys.path.append('./src')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # to supress tf warnings

from read_data.finger_force_reader import read_finger_forces_file
from read_data.finger_position_reader import read_finger_positions_file
from utils.dataset_creation import create_calculated_values_dataset, mirror_data_x_axis
import plots.dataset_plotter as plotter
import utils.normalization as normalization

TRAIN_DATA_DIR: str = "data/sponge_centre"
VALIDATION_DATA_DIR: str = "data/sponge_longside"
TEST_DATA_DIR: str = "data/sponge_shortside"

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
    train_polygons = np.flip(np.load(train_cp_file), axis=0)

    valid_cp_file: str = os.path.join(VALIDATION_DATA_DIR, "fixed_control_points.npy")
    validation_polygons = np.flip(np.load(valid_cp_file),axis=0)


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

    #plotter.plot_finger_force(norm_train_forces, title="Normalized Training Finger Force")
    #plotter.plot_finger_force(train_forces, title="Training Finger Force")

    #plotter.plot_finger_force(norm_valid_forces, title="Normalized Validation Finger Force")
    #plotter.plot_finger_force(validation_forces, title="Validation Finger Force")


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
    train_dataset['finger_position'] = np.concatenate((norm_train_finger_positions, mirrored_finger_positions))

    validation_dataset = {}
    (
        validation_dataset['X_control_points'],
        validation_dataset['X_finger'],
        validation_dataset['Y']
    ) = create_calculated_values_dataset(
        norm_valid_polygons, norm_valid_finger_positions, norm_valid_forces
    )
    validation_dataset['finger_position'] = norm_valid_finger_positions

    return train_dataset, validation_dataset

def create_test_dataset():
    """
        Returns training and validation datasets
    """
    # READ FORCE FILE --------------------------------------------------------------
    test_finger_force_file: str = os.path.join(TEST_DATA_DIR, "finger_force.txt")
    test_forces: np.ndarray = read_finger_forces_file(test_finger_force_file)

    # READ FINGER POSITION FILE ----------------------------------------------------
    test_finger_positions_file: str = os.path.join(TEST_DATA_DIR, "finger_position.txt")
    test_finger_positions: np.ndarray = read_finger_positions_file(
        test_finger_positions_file
    )

    # READ CONTROL POINTS ----------------------------------------------------------
    test_cp_file: str = os.path.join(TEST_DATA_DIR, "fixed_control_points.npy")
    test_polygons = np.flip(np.load(test_cp_file),axis=0)


    # NORMALIZATION ----------------------------------------------------------------
    norm_test_polygons = normalization.normalize_polygons(test_polygons)
    norm_test_finger_positions = normalization.normalize_finger_position(
        test_polygons, test_finger_positions
    )
    norm_test_forces = normalization.normalize_force(test_forces)


    # DATA AUGMENTATION ------------------------------------------------------------


    test_dataset = {}
    (
        test_dataset['X_control_points'],
        test_dataset['X_finger'],
        test_dataset['Y']
    ) = create_calculated_values_dataset(
        norm_test_polygons, norm_test_finger_positions, norm_test_forces
    )
    test_dataset['finger_position'] = norm_test_finger_positions

    return test_dataset