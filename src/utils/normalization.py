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
        min_maxs.append([
            np.min(centred_polygon),
            np.max(centred_polygon)
        ])
    std = np.std(min_maxs)
    print(f"Normalization Params:[\n\tmax_coord_val:{np.max(min_maxs)}\n\tmin_coord_val:{np.min(min_maxs)}\n\tstd:{std}\n]")
    return std

def get_normalized_polygons(polygons):
    transformed_polygons = np.copy(polygons)
    means = get_polygons_centers(transformed_polygons)
    scale = get_scale(transformed_polygons)

    for index, polygon in enumerate(transformed_polygons):
        polygon[..., :2] -= means[index]
        polygon[..., :2] /= scale

    return transformed_polygons