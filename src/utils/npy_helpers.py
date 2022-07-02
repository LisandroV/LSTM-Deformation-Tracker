import numpy as np


def create_basic_dataset(polygons: np.ndarray):
    """Creates dataset with data from a npy file"""
    X_data = np.reshape(polygons, (polygons.shape[0], -1))

    # FIX: this is wrong, its rotating the polygon, not the sequence
    y_data = np.zeros(X_data.shape)
    for index, row in enumerate(X_data):
        y_data[index, :-2] = X_data[index, 2:]
        y_data[index, -2:] = X_data[index, :2]

    # We wrap it since there is only one data sample, there would be more if there were more videos.
    X_data = np.array([X_data])
    y_data = np.array([y_data])

    return X_data, y_data


def create_dataset(
    polygons: np.ndarray, finger_positions: np.ndarray, finger_force: np.ndarray
):
    flat_polygons = np.reshape(polygons, (polygons.shape[0], -1))

    # create X_data
    X_data = np.zeros((flat_polygons.shape[0], flat_polygons.shape[1] + 3))
    for index, flat_polygon in enumerate(flat_polygons):
        X_data[index, 0 : flat_polygon.shape[0]] = flat_polygon
        X_data[index, flat_polygon.shape[0] : -1] = finger_positions[index]
        X_data[index, -1] = finger_force[index]

    # create y_data
    y_data = np.zeros(flat_polygons.shape)
    y_data[:-1] = flat_polygons[1:]
    y_data[-1] = flat_polygons[-1]

    # We wrap it since there is only one data sample, there would be more if there were more videos.
    X_data = np.array([X_data])
    y_data = np.array([y_data])

    return X_data, y_data
