import numpy as np


def create_basic_dataset(polygons: np.ndarray):
    """Creates dataset with data from a npy file"""
    X_data = np.reshape(polygons, (polygons.shape[0], -1))
    y_data = np.zeros(X_data.shape)
    for index, row in enumerate(X_data):
        y_data[index, :-2] = X_data[index, 2:]
        y_data[index, -2:] = X_data[index, :2]

    # We wrap it since there is only one data sample, there would be more if there were more videos.
    X_data = np.array([X_data])
    y_data = np.array([y_data])

    return X_data, y_data


def create_dataset(polygons: np.ndarray, finger_positions: np.ndarray):
    pass
