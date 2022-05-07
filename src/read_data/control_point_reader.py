#!/usr/bin/python3.5
from itertools import islice
import os, sys
import numpy as np

# OPENCV_HOME = os.environ['OPENCV_HOME']
# sys.path.append(OPENCV_HOME + '/lib/python3.5/dist-packages')
sys.path.append("./")

import cv2


class ControlPoint:
    """ Coordinates and neighbours of a control point within the history at a certain time. """
    def __init__(self, x, y, prev_neighbour_index, next_neighbour_index):
        self.x = x
        self.y = y
        self.prev_neighbour_index = prev_neighbour_index
        self.next_neighbour_index = next_neighbour_index

    def __str__(self):
        return "[{0} {1} {2} {3}]".format(self.x, self.y, self.prev_neighbour_index, self.next_neighbour_index)


class ControlPointHistory:
    """ Keeps the coordinates and indices of neighbours in global history for each time frame during which a control
    point was alive.
    """
    UNDEAD = -1

    def __init__(self, ident, birth_time, death_time, history):
        """
        Creates a register with the coordinates of the control point at each time step beginning at birth_time.
        :param ident:       int
        :param birth_time:  int time index when the control point was created
        :param death_time:  int first time index when the control point was no longer seen.
                            UNDEAD if it lasted till the end of the video
        :param history:     [[x:int, y:int, prev:int, next:int]]
        """
        self.ident = ident
        self.birth_time = birth_time
        self.death_time = death_time
        self.history = history

    def get_history_as_array(self):
        """
        Returns array with numbers instead of control point instances
        [[x, y, prev_neighbour_index, next_neighbour_index]]
        """
        if not hasattr(self, 'history_array'):
            harray = np.zeros((len(self.history), 4))
            for i, cp in enumerate(self.history):
                harray[i] = [cp.x, cp.y, cp.prev_neighbour_index, cp.next_neighbour_index]
            self.history_array = harray
        return self.history_array

    def get_control_point(self, time):
        """
        Returns control point at given time or throws an exception
        if this point was not alive at that time.
        :param time:
        :return:
        """
        if time < self.birth_time or (self.death_time != ControlPointHistory.UNDEAD and time >= self.death_time):
            raise Exception("Control point was not alive at time " + str(time) + os.linesep + str(self))
        return self.history[time - self.birth_time];

    def __str__(self):
        return "{0}\t{1}\t{2}\t[{3}]".format(self.ident, self.birth_time, self.death_time, " ".join(str(cp) for cp in self.history))


def parse_control_point_history(str_line):
    """ Parses dodata from str_line and initializes a control point history."""
    tokens = str_line.split()
    ident = int(tokens[0])
    birth_time = int(tokens[1])
    death_time = int(tokens[2])

    # History
    rest = tokens[3:]
    if rest[0][0] != '[':
        raise Exception("'[' was expected but found " + rest[0] + " instead.")
    if rest[-1][-1] != ']':
          raise Exception("']' was expected but found " + rest[-1] + " instead.")
    # Remove '[' and ']'
    rest[0] = rest[0][1:]
    rest[-1] = rest[-1][:-1]
    #print(rest)

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
    """ Keeps the history of all control points for the entire duration of the video. """
    def __init__(self):
        self._header = None
        self.control_points = []

    def load(self, file_name):
        with open(file_name, 'r') as hist_file:
            header = next(hist_file)
            self._header = header.split()
            print(self._header)
            #for line in islice(hist_file, 2):
            for line in hist_file:
                #print(line)
                self.control_points.append(parse_control_point_history(line))
                #print(self.control_points[-1])

    def get_contour_segment(self, cp_ident, n_degree, time):
        """
        Returns a list with the control point at cp_ident in the center, n_degree previous neighbours to its left and
        n_degree next neighbours to its right at the indicated time.
        :param cp_ident: identifiyer of control point at the center.
        :param n_degree: neighbour degree: number of neighbours to each side of this control point.
        :param time: time index when the contour must be reconstructed.
        :return:
        """
        prev = next = control_point = self.control_points[cp_ident].get_control_point(time)
        segment = [control_point]
        for i in range(n_degree):
            prev = self.control_points[prev.prev_neighbour_index].get_control_point(time)
            next = self.control_points[next.next_neighbour_index].get_control_point(time)
            segment.insert(0, prev)
            segment.append(next)
        return segment

    def reconstruct_contour(self, time):
        """
        Reconstructs a list with the pairs of coordinates of all control points at time t.
        :param time:
        :return:
        """
        contour = []
        first = -1
        #print("First before ", str(first))
        #print("First in cps ", self.control_points[0])
        for i, cp_hist in enumerate(self.control_points):
            #print("Point ", str(i), cp_hist.birth_time, cp_hist.death_time)
            if cp_hist.birth_time >= time and (cp_hist.death_time > time or cp_hist.death_time == ControlPointHistory.UNDEAD):
                first = i
                break
        if first == -1:
            raise Exception("Failed to find first point in contour.")
        #print("First ", str(first))
        point = self.control_points[first].get_control_point(time)
        #print("Starting with ", str(point))
        contour.append([point.x, point.y])
        index_of_next = point.next_neighbour_index
        while index_of_next != first:
            point = self.control_points[index_of_next].get_control_point(time)
            contour.append([point.x, point.y])
            index_of_next = point.next_neighbour_index
        return contour

    def __str__(self):
        return "\n".join([str(cph) for cph in self.control_points])


def draw_contour(img, contour, colours=((0, 255, 255), (0, 0, 255))):
#def draw_contour(img, contour, colours=((255, 40, 20), (13, 128, 53))):
    """
    Draw a contour using opencv
    :param img:
    :param contour: array of numpy opencv points with shape (-1, 1, 2)
    :param colours: line and control point colours
    :return:
    """
    cv2.polylines(img, [contour], True, colours[0], 1)
    for pair in contour:
        cv2.circle(img, (pair[0][0], pair[0][1]), 2, colours[1], -1)


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


def plot_history(history):
    """ Plots the trajectories of the control points through time. """
    fig = plt.figure("Control points' history")
    fig.suptitle("Control points' history")
    ax = fig.add_subplot(111, projection='3d')

    NCURVES = len(history.control_points)
    values = range(NCURVES)
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

    # plots every control point history separately
    for i, cp_history in enumerate(history.control_points[0:NCURVES]):
        end_t = cp_history.death_time if cp_history.death_time != ControlPointHistory.UNDEAD else cp_history.birth_time + len(cp_history.history)
        t = np.arange(cp_history.birth_time, end_t)
        harray = cp_history.get_history_as_array()
        x = harray[:,0]
        y = -1 * harray[:,1]

        colorVal = scalarMap.to_rgba(values[i])
        ax.plot(t, x, y, color=colorVal)

    ax.set_xlabel('time (steps)')
    ax.set_ylabel('x (pixels)')
    ax.set_zlabel('y (pixels)')
    plt.show()


# CONTINUE HERE
def create_data(history: ContourHistory, time_degree: int, neighbour_degree: int):
    """ Fills supervised data matrices X and Y. """
    # ASK: why these constants?
    num_time_steps = time_degree + 2
    t_plus_one = time_degree + 1  # index of data for t + 1 for the first step

    data_x = []
    data_y = []

    for cp_history in history.control_points:
        if len(cp_history.history) >= num_time_steps: # need control points with enough history according to T
            start_cp = cp_history.birth_time + t_plus_one
            for t_p_1, control_point in enumerate(cp_history.history[t_plus_one:], start_cp): # ASK: why those cp before t_plus_one are not considered?
                # x = np.zeros(n_characteristics, dtype=np.float64)
                # y = np.zeros(n_output, dtype=np.float64)

                # ADD FINGER FORCES

                # y[:] = (control_point.x, control_point.y)    # position at t + 1

                # ADD FINGER POSITIONS

                dimension = 2
                pos_size = (neighbour_degree * 2 + 1) * dimension
                t = t_p_1 - 1
                time_segments = []
                for delta_t in range(0, t_plus_one): # goes back in time
                    segment = history.get_contour_segment(cp_history.ident, neighbour_degree, t - delta_t)
                    #pos_size = len(segment) * dimensions

                    pos_row = np.zeros(pos_size, dtype=np.float64)
                    for j, cp in enumerate(segment):
                        pos_row[j*2] = cp.x
                        pos_row[j*2+1] = cp.y
                    time_segments.append(pos_row)
                data_x.append(np.array(time_segments))

                data_y.append(np.array([control_point.x, control_point.y]))

    return np.array(data_x), np.array(data_y)


if __name__ == '__main__':
    ##
    ## Plots a 3D graph with the evolution in time of every control point
    ##
    history_file_name = "data/sponge_centre/control_points.hist"
    print("Loading file ", history_file_name)

    history = ContourHistory()
    history.load(history_file_name)
    X_train, Y_train = create_data(history, 2, 10)
    print(np.shape(X_train))
    print(X_train[0])
    print(np.shape(Y_train))
    print(Y_train[0])

    # print("Testing reconstruction at time 0")
    # print(history.reconstruct_contour(0))

    #plot_history(history)
