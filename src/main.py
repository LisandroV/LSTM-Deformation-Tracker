# READ FORCE FILE -------------------------------------------------
import numpy as np

# import numpy.typing as npt
import nptyping as npt
import os

DATA_DIR = "data/sponge_centre"


def read_finger_forces_file(forces_file: str) -> npt.NDArray[npt.Shape["*"], npt.Float]:
    """
    Reads the finger forces file and returns the Fz column.
    Arguments:
        forces_file: path of the forces file to read
    Returns:
        np.float64 numpy ndarray of the forces in Fz.
    """
    forces = np.loadtxt(
        forces_file,
        dtype={
            "names": ("time", "fx", "fy", "fz", "tx", "ty", "tz", "d"),
            "formats": ("f", "f", "f", "f", "f", "f", "f", "f"),
        },
    )
    # only_forces = forces[["fx", "fy", "fz", "tx", "ty", "tz"]]
    return forces[["fz"][0]]


finger_force_file = os.path.join(DATA_DIR, "finger_force.txt")
forces = read_finger_forces_file(finger_force_file)
print(forces)


# READ FINGER POSITION FILE -------------------------------------------------
def read_finger_positions_file(
    finger_position_file: str,
) -> npt.NDArray[npt.Shape["*, [x,y]"], npt.Int]:
    """
    Reads the finger positions file and returns array of (x,y) coordinates.
    Arguments:
        finger_position_file: path of the forces file to read
    Returns:
        (np.float32, np.float32) numpy ndarray
    """
    return np.loadtxt(
        finger_position_file, dtype={"names": ("cx", "cy"), "formats": ("f", "f")}
    )


finger_positions_file = os.path.join(DATA_DIR, "finger_position.txt")
positions = read_finger_positions_file(finger_positions_file)
print(positions)

# crear transformers
