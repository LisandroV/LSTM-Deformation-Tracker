import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

from read_data.control_point_reader import ControlPointHistory, ContourHistory


def plot_control_point_history(history: ContourHistory) -> None:
    """Plots the trajectories of the control points through time."""
    fig = plt.figure("Control points' history")
    fig.suptitle("Control points' history")
    ax = fig.add_subplot(111, projection="3d")

    NCURVES = len(history.cp_histories)
    values = range(NCURVES)
    jet = plt.get_cmap("jet")
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

    # plots every control point history separately
    for i, cp_history in enumerate(history.cp_histories[0:NCURVES]):
        end_t = (
            cp_history.death_time
            if cp_history.death_time != ControlPointHistory.UNDEAD
            else cp_history.birth_time + len(cp_history.history)
        )
        t = np.arange(cp_history.birth_time, end_t)
        harray = cp_history.get_history_as_array()
        x = harray[:, 0]
        y = -1 * harray[:, 1]

        colorVal = scalarMap.to_rgba(values[i])
        ax.plot(t, x, y, color=colorVal)

    ax.set_xlabel("time (steps)")
    ax.set_ylabel("x (pixels)")
    ax.set_zlabel("y (pixels)")
    plt.show()


def plot_finger_position(finger_position_data: np.ndarray) -> None:
    """Plots the finger position through time."""
    fig = plt.figure("Finger position (y axis)")
    fig.suptitle("Finger position in time")
    ax = fig.add_subplot(111)

    x = np.arange(finger_position_data[:, 1].size)
    y = finger_position_data[:, 1]
    ax.scatter(x, y, s=10)

    ax.set_xlabel("time (steps)")
    ax.set_ylabel("y (pixels)")
    plt.show()


def plot_finger_force(finger_force_data) -> None:
    """Plots the finger force through time."""
    fig = plt.figure("Finger force")
    fig.suptitle("Finger force in time")
    ax = fig.add_subplot(111)

    x = np.arange(finger_force_data.size)
    y = finger_force_data
    ax.plot(x, y)

    ax.set_xlabel("time (steps)")
    ax.set_ylabel("force")
    plt.show()

def plot_npz_control_points(control_points, extraplot_cb = None)->None:
    """Plots the trajectories of the control points through time."""
    fig = plt.figure("Control points' history")
    fig.suptitle("Control points' history")
    ax = fig.add_subplot(111, projection="3d")

    NCURVES = np.shape(control_points)[1]
    values = range(NCURVES)
    jet = plt.get_cmap("jet")
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

    # plots every control point history separately
    t = np.arange(0, np.shape(control_points)[0])
    for i in values:
        colorVal = scalarMap.to_rgba(values[i])
        ax.plot(t, control_points[:,i,0], control_points[:,i,1], color=colorVal)

    if(extraplot_cb is not None):
        extraplot_cb(ax)

    ax.set_xlabel("time (steps)")
    ax.set_ylabel("x (pixels)")
    ax.set_zlabel("y (pixels)")
    plt.show()
