import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


def plot_control_point_history(points_data_frame):
    """Plots the trajectories of the control points through time."""
    control_points_by_id = points_data_frame.groupby(
        ["id", "birth_time", "death_time"]
    ).apply(lambda group: list((group["x"], group["y"], group["time_step"])))

    fig = plt.figure("Control points' history")
    fig.suptitle("Control points' history")
    ax = fig.add_subplot(111, projection="3d")

    NCURVES = control_points_by_id.size
    jet = plt.get_cmap("jet")
    cNorm = colors.Normalize(vmin=0, vmax=NCURVES)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

    # plots every control point history separately
    color_i = 0
    for key, row in control_points_by_id.iteritems():
        x = row[0].to_numpy()
        y = row[1].to_numpy()
        time = row[2].to_numpy()
        colorVal = scalarMap.to_rgba(color_i)
        ax.plot(time, x, y, color=colorVal)
        color_i = color_i + 1

    ax.set_xlabel("time (steps)")
    ax.set_ylabel("x (pixels)")
    ax.set_zlabel("y (pixels)")
    plt.show()


def plot_finger_position(finger_position_data):
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


def plot_finger_force(finger_force_data):
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
