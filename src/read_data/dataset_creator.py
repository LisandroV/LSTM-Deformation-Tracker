import numpy as np

from read_data.control_point_reader import ContourHistory, ControlPoint

def create_dataset(history: ContourHistory, time_degree: int, neighbour_degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Fills supervised data matrices X and Y."""
    num_time_steps = time_degree + 2
    t_plus_one = time_degree + 1

    data_x = []
    data_y = []

    for cp_history in history.cp_histories:
        if (
            len(cp_history.history) >= num_time_steps
        ):  # need control points with enough history according to T
            start_cp = cp_history.birth_time + t_plus_one
            for t_p_1, current_cp in enumerate(
                cp_history.history[t_plus_one:], start_cp
            ):  # ASK: why those cp before t_plus_one are not considered?
                # x = np.zeros(n_characteristics, dtype=np.float64)
                # y = np.zeros(n_output, dtype=np.float64)

                # ADD FINGER FORCES

                # y[:] = (control_point.x, control_point.y)    # position at t + 1

                # ADD FINGER POSITIONS

                dimension = 2
                pos_size = (neighbour_degree * 2 + 1) * dimension
                t = t_p_1 - 1
                time_segments = []
                for delta_t in range(0, t_plus_one):  # goes back in time
                    segment: list[ControlPoint] = history.get_contour_segment(
                        cp_history.ident, neighbour_degree, t - delta_t
                    )

                    pos_row = np.zeros(pos_size, dtype=np.float64)
                    for j, cp in enumerate(segment):
                        pos_row[j * 2] = cp.x
                        pos_row[j * 2 + 1] = cp.y
                    time_segments.append(pos_row)
                data_x.append(np.array(time_segments))

                data_y.append(np.array([current_cp.x, current_cp.y]))

    return np.array(data_x), np.array(data_y)