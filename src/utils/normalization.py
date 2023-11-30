import numpy as np


def get_polygon_center(polygon):
    return np.mean(polygon[:, 0:2], axis=0, keepdims=True)[0]


def get_polygons_centers(polygons):
    "the center is given by the mean of all the points in the polygon"
    geom_means = []
    for index in range(np.shape(polygons)[0]):
        geom_means.append(get_polygon_center(polygons[index]))
    return np.array(geom_means)


def get_scale(polygons):
    geom_means = get_polygons_centers(polygons)
    min_maxs = []

    for index, polygon in enumerate(polygons):
        centred_polygon = polygon[..., :2] - geom_means[index]
        min_maxs.append([np.min(centred_polygon), np.max(centred_polygon)])
    std = np.max(min_maxs)  # VERSION 3: use np.max instead of np.std

    return std


# VERSION: 1
# substracts the center of every polygon respectively
# def normalize_polygons(polygons):
#     transformed_polygons = np.copy(polygons)
#     means = get_polygons_centers(transformed_polygons)
#     scale = get_scale(transformed_polygons)

#     for index, polygon in enumerate(transformed_polygons):
#         polygon[..., :2] -= means[index]
#         polygon[..., :2] /= scale

#     return transformed_polygons

# VERSION: 2
# substracts only the center of the first polygon
# this is the best version
def normalize_polygons(polygons):
    transformed_polygons = np.copy(polygons)
    means = get_polygons_centers(transformed_polygons)
    scale = get_scale(transformed_polygons)

    print("NORMALIZATION")
    print(means[0])
    print(scale)

    for polygon in transformed_polygons:
        polygon[..., :2] -= means[0]
        polygon[..., :2] /= scale

    return transformed_polygons


def normalize_finger_position(polygons, finger_positions):
    norm_finger_positions = np.copy(finger_positions)
    means = get_polygons_centers(polygons)
    scale = get_scale(polygons)

    norm_finger_positions[..., :2] -= means[0]
    norm_finger_positions[..., :2] /= scale

    return norm_finger_positions


def normalize_force(forces):
    norm = np.min(forces)
    if norm == 0:
        raise Exception("Is not possible to normalize with norm equal to zero.")
    return forces / norm
