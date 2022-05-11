import numpy as np
import functools

from read_data.control_point_reader import ContourHistory, ControlPoint


def create_dataset(
    history: ContourHistory,
    finger_force_data: np.ndarray,
    finger_position_data: np.ndarray,
    time_degree: int,  # number of segments in the past (considering the current), it consider 0 as a natural
    neighbour_degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fills supervised data matrices X and Y."""
    dimension = 2
    segment_size = (neighbour_degree * 2 + 1) * dimension
    t_plus_one = time_degree + 1

    data_X = []
    data_Y = []

    for cp_history in history.cp_histories:
        if (  # be careful with this condition, if an error happens, add one more
            len(cp_history.history) >= t_plus_one
        ):  # need control points with enough history according to T
            start_cp = cp_history.birth_time + t_plus_one
            for current_time, current_cp in enumerate(
                cp_history.history[t_plus_one:], start_cp
            ):
                t = current_time - 1
                time_segments = np.array([])
                for delta_t in range(
                    0, time_degree
                ):  # Goes back in time to get the segments before the current control point
                    segment: list[ControlPoint] = history.get_contour_segment(
                        cp_history.ident, neighbour_degree, t - delta_t
                    )
                    flat_segment = np.zeros(segment_size, dtype=np.float64)
                    for j, cp in enumerate(segment):
                        flat_segment[j * 2] = cp.x
                        flat_segment[j * 2 + 1] = cp.y

                    time_segments = np.append(time_segments, flat_segment)

                x = functools.reduce(
                    lambda x, y: np.append(x, y),
                    [
                        np.array([finger_force_data[current_time - 1] * 100]),
                        finger_position_data[current_time - 1],
                        time_segments,
                    ],
                )
                data_X.append(x)

                data_Y.append(np.array([current_cp.x, current_cp.y]))

    return np.array(data_X), np.array(data_Y)
