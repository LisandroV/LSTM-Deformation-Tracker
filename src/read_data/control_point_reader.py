import os
import numpy as np


class ControlPoint:
    """Coordinates and neighbours of a control point within the history at a certain time."""

    def __init__(
        self, x: int, y: int, prev_neighbour_index: int, next_neighbour_index: int
    ):
        self.x: int = x
        self.y: int = y
        self.prev_neighbour_index: int = prev_neighbour_index
        self.next_neighbour_index: int = next_neighbour_index

    def __str__(self):
        return "[{0} {1} {2} {3}]".format(
            self.x, self.y, self.prev_neighbour_index, self.next_neighbour_index
        )


class ControlPointHistory:
    """Keeps the coordinates and indices of neighbours in global history for each time frame during which a control
    point was alive.
    """

    UNDEAD = -1

    def __init__(
        self, ident: int, birth_time: int, death_time: int, history: list[ControlPoint]
    ):
        """
        Creates a register with the coordinates of the control point at each time step beginning at birth_time.
        :param ident:       int
        :param birth_time:  int time index when the control point was created
        :param death_time:  int first time index when the control point was no longer seen.
                            UNDEAD if it lasted till the end of the video
        :param history:     [[x:int, y:int, prev:int, next:int]]
        """
        self.ident: int = ident
        self.birth_time: int = birth_time
        self.death_time: int = death_time
        self.history: list[ControlPoint] = history

    def get_history_as_array(self):
        """
        Returns array with numbers instead of control point instances
        [[x, y, prev_neighbour_index, next_neighbour_index]]
        """
        if not hasattr(self, "history_array"):
            harray = np.zeros((len(self.history), 4))
            for i, cp in enumerate(self.history):
                harray[i] = [
                    cp.x,
                    cp.y,
                    cp.prev_neighbour_index,
                    cp.next_neighbour_index,
                ]
            self.history_array = harray
        return self.history_array

    def get_control_point(self, time: int) -> ControlPoint:
        """
        Returns control point at given time or throws an exception
        if this point was not alive at that time.
        :param time:
        :return:
        """
        if time < self.birth_time or (
            self.death_time != ControlPointHistory.UNDEAD and time >= self.death_time
        ):
            raise Exception(
                "Control point was not alive at time "
                + str(time)
                + os.linesep
                + str(self)
            )
        return self.history[time - self.birth_time]

    def __str__(self):
        return "{0}\t{1}\t{2}\t[{3}]".format(
            self.ident,
            self.birth_time,
            self.death_time,
            " ".join(str(cp) for cp in self.history),
        )


def parse_control_point_history(str_line: str) -> ControlPointHistory:
    """Parses dodata from str_line and initializes a control point history."""
    tokens = str_line.split()
    ident = int(tokens[0])
    birth_time = int(tokens[1])
    death_time = int(tokens[2])

    # History
    rest = tokens[3:]
    if rest[0][0] != "[":
        raise Exception("'[' was expected but found " + rest[0] + " instead.")
    if rest[-1][-1] != "]":
        raise Exception("']' was expected but found " + rest[-1] + " instead.")
    # Remove '[' and ']'
    rest[0] = rest[0][1:]
    rest[-1] = rest[-1][:-1]

    hist = []
    iterator = iter(rest)
    x_str = next(iterator, False)
    while x_str:
        x = int(x_str[1:])
        y = int(next(iterator))
        prev_neighbour_index = int(next(iterator))
        next_neighbour_index = int(next(iterator)[:-1])
        x_str = next(iterator, False)
        hist.append(ControlPoint(x, y, prev_neighbour_index, next_neighbour_index))

    return ControlPointHistory(ident, birth_time, death_time, hist)


class ContourHistory:
    """Keeps the history of all control points for the entire duration of the video."""

    def __init__(self, history_file_name: str):
        self._header = None
        self.cp_histories: list[ControlPointHistory] = []
        self.load(history_file_name)

    def load(self, file_name: str):
        with open(file_name, "r") as hist_file:
            header = next(hist_file)
            self._header = header.split()
            print(self._header)
            for line in hist_file:
                self.cp_histories.append(parse_control_point_history(line))

    def get_contour_segment(
        self, cp_ident: int, n_degree: int, time: int
    ) -> list[ControlPoint]:
        """
        Returns a list with the control point at cp_ident in the center, n_degree previous neighbours to its left and
        n_degree next neighbours to its right at the indicated time.
        :param cp_ident: identifiyer of control point at the center.
        :param n_degree: neighbour degree: number of neighbours to each side of this control point.
        :param time: time index when the contour must be reconstructed.
        :return:
        """
        prev = next = control_point = self.cp_histories[cp_ident].get_control_point(
            time
        )
        segment: list[ControlPoint] = [control_point]
        for i in range(n_degree):
            prev = self.cp_histories[prev.prev_neighbour_index].get_control_point(time)
            next = self.cp_histories[next.next_neighbour_index].get_control_point(time)
            segment.insert(0, prev)
            segment.append(next)
        return segment

    def reconstruct_contour(self, time: int):
        """
        Reconstructs a list with the pairs of coordinates of all control points at time t.
        :param time:
        :return:
        """
        contour = []
        first = -1
        for i, cp_hist in enumerate(self.cp_histories):
            if cp_hist.birth_time >= time and (
                cp_hist.death_time > time
                or cp_hist.death_time == ControlPointHistory.UNDEAD
            ):
                first = i
                break
        if first == -1:
            raise Exception("Failed to find first point in contour.")
        point = self.cp_histories[first].get_control_point(time)
        contour.append([point.x, point.y])
        index_of_next = point.next_neighbour_index
        while index_of_next != first:
            point = self.cp_histories[index_of_next].get_control_point(time)
            contour.append([point.x, point.y])
            index_of_next = point.next_neighbour_index
        return contour

    def __str__(self):
        return "\n".join([str(cph) for cph in self.cp_histories])
