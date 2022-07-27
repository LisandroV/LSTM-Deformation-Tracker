import numpy as np


def create_basic_dataset(polygons: np.ndarray):
    """Creates dataset with data from a npy file"""
    X_data = np.reshape(polygons, (polygons.shape[0], -1))

    y_data = np.zeros(X_data.shape)
    y_data[:-1] = X_data[1:]
    y_data[-1] = X_data[-1]

    # We wrap it since there is only one data sample, there would be more if there were more videos.
    X_data = np.array([X_data])
    y_data = np.array([y_data])

    return X_data, y_data


def create_polygon_datapoint(
    polygons: np.ndarray, finger_positions: np.ndarray, finger_force: np.ndarray
):
    """returns one data instance by merging the polygons, finger_positions and finger_force together along with the expected result"""
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

    return X_data, y_data


def create_dataset(
    polygons: np.ndarray, finger_positions: np.ndarray, finger_force: np.ndarray
):
    """
    Creates the dataset with only one video instance, which means only one polygon sequence with finger position and force
    """
    X_instance, y_instance = create_polygon_datapoint(
        polygons, finger_positions, finger_force
    )

    # We wrap it since there is only one data sample, there would be more if there were more videos.
    X_data = np.array([X_instance])
    y_data = np.array([y_instance])

    return X_data, y_data


def rotate_plygons(polygons: np.ndarray):
    """
    Takes the first coordinate of the polygon and puts it in the end of the list
    """
    rotated_polygons = np.zeros(polygons.shape)
    for index, polygon in enumerate(polygons):
        rotated_polygons[index, :-1] = polygon[1:]
        rotated_polygons[index, -1] = polygon[0]
    return rotated_polygons


def create_rotating_coordinates_dataset(
    polygons: np.ndarray, finger_positions: np.ndarray, finger_force: np.ndarray
):
    """
    Data multiplication by rotating the coordinates of the initial polygon sequence
    """
    num_points = polygons.shape[1]
    rotated_polygons = polygons
    X_data, y_data = [], []
    for _ in range(num_points):
        X_instance, y_instance = create_polygon_datapoint(
            rotated_polygons, finger_positions, finger_force
        )
        X_data.append(X_instance)
        y_data.append(y_instance)
        rotated_polygons = rotate_plygons(rotated_polygons)

    return np.array(X_data), np.array(y_data)

def mirror_data_x_axis(
    polygons: np.ndarray, finger_positions: np.ndarray, finger_force: np.ndarray
):
    """
    Apply a transformation on the polygon and finger position
    """
    mirrored_polygons  = polygons.copy()
    mirrored_finger_positions  = finger_positions.copy()

    # Move and scale a bit so its a bit different from the validation data
    mirrored_polygons[:,:,0] *= -0.9
    mirrored_finger_positions[:,0] *= -0.9

    mirrored_polygons[:,:,1] *= 0.93
    mirrored_finger_positions[:,1] *= 0.93

    mirrored_polygons[:,:,0] += 0.005
    mirrored_finger_positions[:,0] += 0.005

    mirrored_polygons[:,:,1] += 0.005
    mirrored_finger_positions[:,1] += 0.005

    return mirrored_polygons, mirrored_finger_positions, finger_force

def create_multiple_step_dataset(
    polygons: np.ndarray, finger_positions: np.ndarray, finger_force: np.ndarray, step_size: int=10
):
    X_data, y_data = create_dataset(polygons, finger_positions, finger_force)
    y_step_data = []
    for i in range(polygons.shape[0] - step_size):
        y_step_data.append(y_data[0,i:i+step_size].reshape(-1))
    return X_data[:,:-step_size], np.array([y_step_data])
